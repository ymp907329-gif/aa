import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import FusionDataset
from models.modules.frequency import haar_forward, haar_inverse
from utils.config import create_model
from utils.metrics import compute_fusion_metrics


METRICS = ('psnr', 'ssim', 'vif', 'entropy', 'sd', 'ag', 'sf', 'mi', 'qabf')


def calibrate(image):
    image = image.clamp(0, 1)
    return (image - image.min()) / (image.max() - image.min()).clamp_min(1e-6)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--data_root')
    parser.add_argument('--output', required=True)
    parser.add_argument('--save_dir')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_batches', type=int)
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--beta_scale', type=float, default=1.0)
    parser.add_argument('--condition', choices=('clean', 'gaussian_0.01', 'gaussian_0.03', 'poisson'), default='clean')
    parser.add_argument('--noise_seed', type=int, default=1000)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = checkpoint['config']
    if args.data_root:
        cfg['data']['root'] = args.data_root
    model = create_model(cfg).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    if model.adaptive_compensation is None:
        raise ValueError('The checkpoint does not enable adaptive compensation')
    model.adaptive_compensation.output_scale = args.beta_scale
    if args.validation:
        from train import make_loaders
        _, loader = make_loaders(cfg)
        split = 'validation'
    else:
        dataset = FusionDataset(cfg['data']['root'], split=cfg['data']['test_split'], size=cfg['data']['size'])
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        split = 'test'

    rows = []
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        for variant in ('none', 'fixed', 'adaptive', 'beta'):
            (save_dir / variant).mkdir(parents=True, exist_ok=True)
    for index, batch in enumerate(tqdm(loader, desc='Adaptive compensation evaluation')):
        if args.max_batches is not None and index >= args.max_batches:
            break
        if args.condition != 'clean':
            from scripts.evaluate_compensation_noise import add_noise
            ir_array, vi_array = add_noise(
                batch['ir'][0, 0].numpy(), batch['vi'][0, 0].numpy(),
                args.condition, args.noise_seed + index,
            )
            batch['ir'] = torch.from_numpy(ir_array).float()[None, None]
            batch['vi'] = torch.from_numpy(vi_array).float()[None, None]
        ir = batch['ir'].to(device)
        vi = batch['vi'].to(device)
        adaptive, aux = model(ir, vi, return_aux=True)
        uncompensated = aux['uncompensated_output']
        target = torch.maximum(ir, vi)
        fused_bands = haar_forward(uncompensated)
        target_bands = haar_forward(target)
        fixed = haar_inverse(
            fused_bands[0],
            *(band + 0.5 * reference for band, reference in zip(fused_bands[1:], target_bands[1:])),
            output_size=uncompensated.shape[-2:],
        )
        variants = {'none': uncompensated, 'fixed': fixed, 'adaptive': adaptive}
        images = {name: calibrate(image)[0, 0].cpu().numpy() for name, image in variants.items()}
        ir_image = batch['ir'][0, 0].numpy()
        vi_image = batch['vi'][0, 0].numpy()
        name = Path(batch['name'][0]).stem
        row = {'name': batch['name'][0]}
        for variant, image in images.items():
            values = compute_fusion_metrics(image, ir_image, vi_image)
            row.update({f'{variant}_{key}': values[key] for key in METRICS})
            if save_dir:
                Image.fromarray((image * 255).round().astype(np.uint8)).save(save_dir / variant / f'{name}.png')
        beta = aux['compensation_beta'][0, 0].cpu().numpy()
        saturation = aux['compensation_saturation'][0, 0].cpu().numpy().astype(bool)
        row['beta_mean'] = float(beta.mean())
        row['beta_std'] = float(beta.std())
        row['beta_saturated_mean'] = float(beta[saturation].mean()) if saturation.any() else float('nan')
        row['beta_unsaturated_mean'] = float(beta[~saturation].mean()) if (~saturation).any() else float('nan')
        rows.append(row)
        if save_dir:
            beta_max = model.adaptive_compensation.beta_max * model.adaptive_compensation.output_scale
            beta_display = beta.mean(axis=0) if beta.ndim == 3 else beta
            Image.fromarray((beta_display / beta_max * 255).round().astype(np.uint8)).save(save_dir / 'beta' / f'{name}.png')
    if not rows:
        raise ValueError('No samples evaluated')

    summary = {}
    for variant in ('none', 'fixed', 'adaptive'):
        summary[variant] = {
            key: float(np.mean([row[f'{variant}_{key}'] for row in rows])) for key in METRICS
        }
    summary['beta'] = {
        key: float(np.nanmean([row[key] for row in rows]))
        for key in ('beta_mean', 'beta_std', 'beta_saturated_mean', 'beta_unsaturated_mean')
    }
    result = {
        'checkpoint': args.ckpt,
        'data_root': cfg['data']['root'],
        'split': split,
        'beta_scale': args.beta_scale,
        'condition': args.condition,
        'sample_count': len(rows),
        'protocol': 'Same fine-tuned backbone; clip and per-image min-max applied identically to all three variants',
        'summary': summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    fieldnames = ['name']
    for variant in ('none', 'fixed', 'adaptive'):
        fieldnames.extend(f'{variant}_{key}' for key in METRICS)
    fieldnames.extend(('beta_mean', 'beta_std', 'beta_saturated_mean', 'beta_unsaturated_mean'))
    with open(output.with_suffix('.csv'), 'w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f'Saved: {output}, {output.with_suffix(".csv")}')


if __name__ == '__main__':
    main()
