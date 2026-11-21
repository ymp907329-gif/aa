import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from scripts.evaluate_high_frequency_compensation import quantize
from scripts.search_two_stage_compensation import compensate
from train import make_loaders
from utils.config import create_model
from utils.metrics import compute_fusion_metrics


METRICS = ('entropy', 'sd', 'sf', 'vif', 'qabf')
CONDITIONS = ('clean', 'gaussian_0.01', 'gaussian_0.03', 'poisson')


def add_noise(ir, vi, condition, seed):
    if condition == 'clean':
        return ir, vi
    rng = np.random.default_rng(seed)
    if condition.startswith('gaussian'):
        sigma = float(condition.split('_')[1])
        ir = np.clip(ir + rng.normal(0, sigma, ir.shape), 0, 1)
        vi = np.clip(vi + rng.normal(0, sigma, vi.shape), 0, 1)
    else:
        ir = np.clip(rng.poisson(ir * 255) / 255.0, 0, 1)
        vi = np.clip(rng.poisson(vi * 255) / 255.0, 0, 1)
    return ir, vi


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--data_root')
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_batches', type=int)
    parser.add_argument('--skip_minmax', action='store_true')
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = checkpoint['config']
    if args.data_root:
        cfg['data']['root'] = args.data_root
    model = create_model(cfg).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    _, loader = make_loaders(cfg)
    results = {}
    for condition in CONDITIONS:
        results[condition] = {'reconstruction': [], 'compensated': []}
    for index, batch in enumerate(tqdm(loader, desc='Noise robustness')):
        if args.max_batches is not None and index >= args.max_batches:
            break
        clean_ir = batch['ir'][0, 0].cpu().numpy()
        clean_vi = batch['vi'][0, 0].cpu().numpy()
        for condition in CONDITIONS:
            ir, vi = add_noise(clean_ir, clean_vi, condition, 1000 + index)
            fused = model(
                torch.from_numpy(ir).float()[None, None].to(device),
                torch.from_numpy(vi).float()[None, None].to(device),
            )[0, 0].cpu().numpy()
            if args.skip_minmax:
                fused = np.clip(fused, 0, 1)
                compensated = np.clip(compensate(fused, ir, vi, 0.0, 0.5), 0, 1)
            else:
                fused = quantize((fused - fused.min()) / max(float(fused.max() - fused.min()), 1e-6))
                compensated = quantize(compensate(fused, ir, vi, 0.0, 0.5))
            for mode, output in (('reconstruction', fused), ('compensated', compensated)):
                values = compute_fusion_metrics(output, ir, vi)
                results[condition][mode].append({key: values[key] for key in METRICS})
    summary = {}
    for condition in CONDITIONS:
        summary[condition] = {}
        for mode in ('reconstruction', 'compensated'):
            rows = results[condition][mode]
            summary[condition][mode] = {
                key: float(np.mean([row[key] for row in rows])) for key in METRICS
            }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        'checkpoint': args.ckpt, 'split': 'validation',
        'sample_count': len(loader), 'conditions': summary,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    for condition in CONDITIONS:
        print(condition)
        for mode in ('reconstruction', 'compensated'):
            values = summary[condition][mode]
            print(mode, ' '.join(f'{key}={values[key]:.4f}' for key in METRICS))
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
