import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from train import make_loaders
from utils.config import create_model
from utils.metrics import compute_fusion_metrics


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints/ablation/region_aid_only/best.pth')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--thresholds', nargs='+', type=float,
                        default=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5))
    parser.add_argument('--max_batches', type=int)
    parser.add_argument('--soft_region', action='store_true')
    parser.add_argument('--soft_temperature', type=float, default=0.05)
    parser.add_argument('--output', default='results/diagnostics/stop_thresholds/region_aid_only_validation.json')
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.ckpt, map_location=device)
    cfg = checkpoint['config']
    model = create_model(cfg).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    if not model.aid.region_adaptive:
        raise ValueError('Threshold analysis requires region-adaptive AID')
    model.aid.soft_region = args.soft_region
    model.aid.soft_temperature = args.soft_temperature
    _, loader = make_loaders(cfg)
    batches = []
    for index, batch in enumerate(loader):
        if args.max_batches is not None and index >= args.max_batches:
            break
        batches.append((batch['ir'].to(device), batch['vi'].to(device)))
    results = []
    for threshold in args.thresholds:
        model.aid.residual_tol = threshold
        rows = []
        step_counts = {}
        for ir, vi in tqdm(batches, desc=f'Threshold {threshold}'):
            out, aux = model(ir, vi, return_aux=True)
            steps = aux['aid']['steps_map'].float().cpu()
            values, counts = torch.unique(steps, return_counts=True)
            for value, count in zip(values.tolist(), counts.tolist()):
                key = str(int(value))
                step_counts[key] = step_counts.get(key, 0) + count
            for index in range(out.shape[0]):
                metrics = compute_fusion_metrics(
                    out[index].cpu().numpy(), ir[index].cpu().numpy(), vi[index].cpu().numpy()
                )
                rows.append({'mean_steps': float(steps[index].mean()), **metrics})
        summary = {
            'threshold': threshold,
            'sample_count': len(rows),
            'mean_steps': float(np.mean([row['mean_steps'] for row in rows])),
            'step_counts_pixels': step_counts,
            'step_fraction_pixels': {
                key: count / sum(step_counts.values()) for key, count in sorted(step_counts.items())
            },
            'metrics': {
                key: float(np.mean([row[key] for row in rows]))
                for key in ('entropy', 'sd', 'sf', 'mi', 'vif', 'qabf', 'ssim', 'psnr')
            },
        }
        results.append(summary)
        print(json.dumps(summary, ensure_ascii=False))
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as file:
        json.dump({'checkpoint': args.ckpt, 'split': 'validation', 'results': results}, file, indent=2)
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
