import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import t
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import FusionDataset
from utils.config import ABLATIONS, apply_ablation, create_model, load_config
from utils.metrics import compute_fusion_metrics


def _grid_sample(image, grid):
    return F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)


def perturb_pair(ir, vi, kind, level):
    if kind == 'lowlight':
        scale = max(0.1, 1.0 - 0.2 * level)
        return ir, (vi * scale).clamp(0, 1)
    if kind == 'noise':
        sigma = 0.02 * level
        return (ir + torch.randn_like(ir) * sigma).clamp(0, 1), (vi + torch.randn_like(vi) * sigma).clamp(0, 1)

    batch, _, height, width = vi.shape
    if kind == 'shift':
        theta = vi.new_zeros(batch, 2, 3)
        theta[:, 0, 0] = 1
        theta[:, 1, 1] = 1
        theta[:, 0, 2] = 2.0 * level / max(width - 1, 1)
        grid = F.affine_grid(theta, vi.shape, align_corners=True)
        return ir, _grid_sample(vi, grid)
    if kind == 'rotation':
        angle = math.radians(level)
        theta = vi.new_zeros(batch, 2, 3)
        theta[:, 0, 0] = math.cos(angle)
        theta[:, 0, 1] = -math.sin(angle)
        theta[:, 1, 0] = math.sin(angle)
        theta[:, 1, 1] = math.cos(angle)
        grid = F.affine_grid(theta, vi.shape, align_corners=True)
        return ir, _grid_sample(vi, grid)
    if kind == 'elastic':
        coarse_h, coarse_w = max(2, height // 16), max(2, width // 16)
        displacement = torch.randn(batch, 2, coarse_h, coarse_w, device=vi.device, dtype=vi.dtype)
        displacement = F.interpolate(displacement, size=(height, width), mode='bilinear', align_corners=False)
        displacement = displacement / displacement.abs().mean().clamp_min(1e-6) * level
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, height, device=vi.device, dtype=vi.dtype),
            torch.linspace(-1, 1, width, device=vi.device, dtype=vi.dtype),
            indexing='ij',
        )
        base = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        normalized = displacement.permute(0, 2, 3, 1) * vi.new_tensor(
            [2.0 / max(width - 1, 1), 2.0 / max(height - 1, 1)]
        )
        return ir, _grid_sample(vi, base + normalized)
    raise ValueError(f'Unknown perturbation: {kind}')


@torch.no_grad()
def evaluate_perturbation(model, loader, device, kind, level, max_batches=None):
    model.eval()
    values = {key: [] for key in ('psnr', 'ssim', 'vif', 'entropy', 'sd', 'ag', 'sf', 'mi', 'qabf')}
    for index, batch in enumerate(tqdm(loader, desc=f'{kind}-{level}', leave=False)):
        if max_batches is not None and index >= max_batches:
            break
        ir, vi = batch['ir'].to(device), batch['vi'].to(device)
        ir, vi = perturb_pair(ir, vi, kind, level)
        fused = model(ir, vi)
        for i in range(fused.shape[0]):
            metrics = compute_fusion_metrics(
                fused[i].cpu().numpy(), ir[i].cpu().numpy(), vi[i].cpu().numpy()
            )
            for key, value in metrics.items():
                values[key].append(value)
    return {key: float(np.mean(items)) for key, items in values.items()}


def summarize(runs):
    result = {}
    for key in runs[0]:
        values = np.array([run[key] for run in runs], dtype=np.float64)
        mean = values.mean()
        std = values.std(ddof=1) if len(values) > 1 else 0.0
        confidence = float(t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values))) if len(values) > 1 else 0.0
        result[key] = {'mean': float(mean), 'std': float(std), 'ci95': confidence}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ablation', choices=ABLATIONS)
    parser.add_argument('--alignment_mode', choices=('deformable', 'transport'))
    parser.add_argument('--kinds', nargs='+', default=['shift', 'rotation', 'elastic', 'lowlight', 'noise'])
    parser.add_argument('--levels', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--max_batches', type=int)
    parser.add_argument('--output', default='robustness.json')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get('config', load_config(args.config))
    if args.ablation:
        cfg = apply_ablation(cfg, args.ablation)
    if args.alignment_mode:
        cfg['model'].setdefault('dfd', {})['alignment_mode'] = args.alignment_mode
        if args.alignment_mode == 'transport':
            cfg['model']['dfd'].update({
                'transport_grid': 16, 'sinkhorn_iters': 30,
                'transport_epsilon': 0.05, 'transport_spatial_weight': 0.01,
            })
    model = create_model(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    dataset = FusionDataset(cfg['data']['root'], split=cfg['data']['test_split'], size=cfg['data']['size'])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg['data'].get('num_workers', 0))

    output = {'runs': args.runs, 'levels': args.levels, 'results': {}}
    for kind in args.kinds:
        output['results'][kind] = {}
        for level in args.levels:
            runs = []
            for run in range(args.runs):
                torch.manual_seed(cfg['train']['seed'] + run)
                runs.append(evaluate_perturbation(model, loader, device, kind, level, args.max_batches))
            output['results'][kind][str(level)] = {'runs': runs, 'summary': summarize(runs)}
            print(f'{kind} level {level}: SSIM={output["results"][kind][str(level)]["summary"]["ssim"]}')
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f'Saved robustness results: {args.output}')


if __name__ == '__main__':
    main()
