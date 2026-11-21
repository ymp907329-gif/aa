import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import FusionDataset
from scripts.run_value_controlled import ControlledFusion
from train import make_loaders
from utils.config import create_model
from utils.metrics import compute_fusion_metrics
from utils.value_control import ValueDepthController


METRICS = ('entropy', 'sd', 'sf', 'mi', 'vif', 'qabf')
ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
MODES = ('high_mix', 'high_add', 'full_mix')


def haar(image):
    a = image[..., 0::2, 0::2]
    b = image[..., 0::2, 1::2]
    c = image[..., 1::2, 0::2]
    d = image[..., 1::2, 1::2]
    return (
        (a + b + c + d) * 0.5,
        (a - b + c - d) * 0.5,
        (a + b - c - d) * 0.5,
        (a - b - c + d) * 0.5,
    )


def inverse_haar(bands):
    ll, lh, hl, hh = bands
    a = (ll + lh + hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    out = np.zeros((ll.shape[0] * 2, ll.shape[1] * 2), dtype=np.float64)
    out[0::2, 0::2] = a
    out[0::2, 1::2] = b
    out[1::2, 0::2] = c
    out[1::2, 1::2] = d
    return out


def compensate(fused, ir, vi, mode, alpha):
    target = np.maximum(ir, vi)
    if mode == 'full_mix':
        return (1 - alpha) * fused + alpha * target
    fused_bands = haar(fused)
    target_bands = haar(target)
    high_fused = np.array(fused_bands[1:])
    high_target = np.array(target_bands[1:])
    if mode == 'high_mix':
        high = (1 - alpha) * high_fused + alpha * high_target
    elif mode == 'high_add':
        high = high_fused + alpha * high_target
    else:
        raise ValueError(mode)
    return inverse_haar((fused_bands[0], *high))


def quantize(image):
    return (np.clip(image, 0, 1) * 255).round().astype(np.uint8).astype(np.float64) / 255.0


@torch.no_grad()
def main():
    global MODES, ALPHAS
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--controller')
    parser.add_argument('--data_root')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_batches', type=int)
    parser.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--alphas', nargs='+', type=float, default=list(ALPHAS))
    parser.add_argument('--skip_minmax', action='store_true')
    args = parser.parse_args()
    MODES = tuple(args.modes)
    ALPHAS = tuple(args.alphas)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = checkpoint['config']
    if args.data_root:
        cfg['data']['root'] = args.data_root
    model = create_model(cfg).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    if args.controller:
        model = ControlledFusion(model, ValueDepthController(args.controller)).to(device)
    if args.test:
        dataset = FusionDataset(cfg['data']['root'], split=cfg['data']['test_split'], size=cfg['data']['size'])
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        split = 'test'
    else:
        _, loader = make_loaders(cfg)
        split = 'validation'
    rows = []
    for index, batch in enumerate(tqdm(loader, desc='High-frequency compensation')):
        if args.max_batches is not None and index >= args.max_batches:
            break
        ir = batch['ir'][0, 0].cpu().numpy()
        vi = batch['vi'][0, 0].cpu().numpy()
        fused = model(batch['ir'].to(device), batch['vi'].to(device))[0, 0].cpu().numpy()
        if args.skip_minmax:
            fused = np.clip(fused, 0, 1)
        else:
            fused = quantize((fused - fused.min()) / max(float(fused.max() - fused.min()), 1e-6))
        for mode in MODES:
            for alpha in ALPHAS:
                calibrated = compensate(fused, ir, vi, mode, alpha)
                calibrated = np.clip(calibrated, 0, 1) if args.skip_minmax else quantize(calibrated)
                values = compute_fusion_metrics(calibrated, ir, vi)
                rows.append({
                    'name': batch['name'][0], 'mode': mode, 'alpha': alpha,
                    **{key: values[key] for key in METRICS},
                })
    if not rows:
        raise ValueError('No samples evaluated')
    summary = {}
    for mode in MODES:
        summary[mode] = {}
        for alpha in ALPHAS:
            selected = [row for row in rows if row['mode'] == mode and row['alpha'] == alpha]
            summary[mode][str(alpha)] = {
                key: float(np.mean([row[key] for row in selected])) for key in METRICS
            }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        'checkpoint': args.ckpt, 'controller': args.controller,
        'split': split, 'sample_count': len(loader), 'alphas': ALPHAS,
        'summary': summary,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    with open(output.with_suffix('.csv'), 'w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=('name', 'mode', 'alpha', *METRICS))
        writer.writeheader()
        writer.writerows(rows)
    for mode in MODES:
        print(mode)
        for alpha in ALPHAS:
            values = summary[mode][str(alpha)]
            print(alpha, ' '.join(f'{key}={values[key]:.4f}' for key in METRICS))
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
