import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from train import make_loaders
from utils.config import create_model
from utils.metrics import compute_fusion_metrics


def calibrate(fused, ir, vi, mode):
    if mode == 'identity':
        return fused
    if mode == 'minmax':
        low, high = fused.min(), fused.max()
        return (fused - low) / (high - low).clamp_min(1e-6)
    if mode == 'percentile':
        low = torch_quantile(fused, 0.01)
        high = torch_quantile(fused, 0.99)
        return ((fused - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
    if mode == 'meanstd':
        target_mean = 0.5 * (ir.mean() + vi.mean())
        target_std = torch.maximum(ir.std(), vi.std())
        return ((fused - fused.mean()) * target_std / fused.std().clamp_min(1e-6) + target_mean).clamp(0, 1)
    raise ValueError(f'Unknown calibration mode: {mode}')


def torch_quantile(tensor, q):
    return torch.quantile(tensor.flatten().float(), q)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints/ablation/region_lifting_miqabf_blend/best.pth')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_root')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--max_batches', type=int)
    parser.add_argument('--output', default='results/research/intensity_calibration/e4_blend')
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.ckpt, map_location=device)
    cfg = checkpoint['config']
    if args.data_root:
        cfg['data']['root'] = args.data_root
    model = create_model(cfg).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    if args.test:
        from data.dataset import FusionDataset
        from torch.utils.data import DataLoader
        loader = DataLoader(
            FusionDataset(cfg['data']['root'], split=cfg['data']['test_split'], size=cfg['data']['size']),
            batch_size=1, shuffle=False, num_workers=0,
        )
    else:
        _, loader = make_loaders(cfg)
    modes = ('identity', 'minmax', 'percentile', 'meanstd')
    results = {mode: [] for mode in modes}
    for index, batch in enumerate(tqdm(loader, desc='Intensity calibration')):
        if args.max_batches is not None and index >= args.max_batches:
            break
        ir, vi = batch['ir'].to(device), batch['vi'].to(device)
        fused = model(ir, vi)
        for mode in modes:
            calibrated = calibrate(fused, ir, vi, mode)
            results[mode].append(compute_fusion_metrics(
                calibrated[0].cpu().numpy(), ir[0].cpu().numpy(), vi[0].cpu().numpy()
            ))
    if not results['identity']:
        raise ValueError('No samples were evaluated')
    summary = {
        'checkpoint': args.ckpt,
        'data_root': cfg['data']['root'],
        'split': 'test' if args.test else 'validation',
        'results': {
            mode: {
                key: float(np.mean([row[key] for row in rows]))
                for key in ('entropy', 'sd', 'sf', 'mi', 'vif', 'qabf', 'ssim', 'psnr')
            }
            for mode, rows in results.items()
        },
    }
    with open(os.path.join(args.output, 'analysis.json'), 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
