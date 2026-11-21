import argparse
import json
import math
import os

import numpy as np
import torch
from tqdm import tqdm

from scripts.analyze_iteration_value import fusion_value
from train import make_loaders
from utils.config import create_model
from utils.metrics import compute_fusion_metrics


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints/ablation/region_lifting_miqabf/best.pth')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output', default='results/research/band_shapley/e4')
    parser.add_argument('--max_batches', type=int, default=1)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    device = torch.device(
        'cpu' if args.gpu < 0 else f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    )
    checkpoint = torch.load(args.ckpt, map_location=device)
    model = create_model(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    dfd = model.fusion
    _, val_loader = make_loaders(checkpoint['config'])
    band_names = ('LL', 'LH', 'HL', 'HH')
    subset_samples = {}
    subset_metrics = {}
    for index, batch in enumerate(tqdm(val_loader, desc='Band Shapley')):
        if index >= args.max_batches:
            break
        ir, vi = batch['ir'].to(device), batch['vi'].to(device)
        f0_ir, _, feature_ir = model.encoder_ir(ir)
        f0_vi, _, feature_vi = model.encoder_vi(vi)
        feature_ir, feature_vi, _ = model.aid(feature_ir, feature_vi)
        aligned_ir, aligned_vi, _ = dfd._align(feature_ir, feature_vi)
        bands_ir = dfd._forward_wavelet(aligned_ir)
        bands_vi = dfd._forward_wavelet(aligned_vi)
        for subset in range(16):
            ll = dfd._fuse_band(bands_ir[0], bands_vi[0], dfd.low_weight, dfd.low_fusion)
            if subset & 1 == 0:
                ll = torch.zeros_like(ll)
            high = []
            for band_index in range(1, 4):
                band = dfd._fuse_band(
                    bands_ir[band_index], bands_vi[band_index],
                    dfd.high_weight, dfd.high_fusion, mode=dfd.high_fusion_mode,
                )
                if subset & (1 << band_index) == 0:
                    band = torch.zeros_like(band)
                high.append(band)
            fused_feature = dfd.output(dfd._inverse_wavelet((ll, *high), output_size=feature_ir.shape[-2:]))
            fused = model.decoder(fused_feature, f0_ir, f0_vi)
            value, _ = fusion_value(fused, ir, vi)
            subset_samples.setdefault(subset, []).append(value.mean().item())
            subset_metrics.setdefault(subset, []).append(compute_fusion_metrics(
                fused[0].cpu().numpy(), ir[0].cpu().numpy(), vi[0].cpu().numpy()
            ))

    subset_scores = {key: float(np.mean(values)) for key, values in subset_samples.items()}
    subset_metric_means = {
        key: {
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in ('entropy', 'sd', 'sf', 'mi', 'vif', 'qabf')
        }
        for key, rows in subset_metrics.items()
    }
    shapley = {}
    for band in range(4):
        value = 0.0
        for subset in range(16):
            if subset & (1 << band):
                continue
            size = subset.bit_count()
            weight = math.factorial(size) * math.factorial(3 - size) / math.factorial(4)
            value += weight * (subset_scores[subset | (1 << band)] - subset_scores[subset])
        shapley[band_names[band]] = value
    residual = sum(shapley.values()) - (subset_scores[15] - subset_scores[0])
    summary = {
        'checkpoint': args.ckpt,
        'samples': len(subset_samples[0]),
        'subset_scores': {format(key, '04b'): value for key, value in subset_scores.items()},
        'subset_metrics': {format(key, '04b'): value for key, value in subset_metric_means.items()},
        'shapley': shapley,
        'full_minus_empty': subset_scores[15] - subset_scores[0],
        'conservation_residual': residual,
        'interpretation': 'Subset bit order: LL, LH, HL, HH; 1 means the band is retained.',
    }
    with open(os.path.join(args.output, 'analysis.json'), 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
