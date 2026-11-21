import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import FusionDataset
from models.modules.frequency import haar_forward, haar_inverse
from utils.config import create_model
from utils.metrics import compute_fusion_metrics


METRICS = ('entropy', 'sd', 'sf', 'vif', 'qabf')


def calibrate(image):
    image = image.clamp(0, 1)
    return (image - image.min()) / (image.max() - image.min()).clamp_min(1e-6)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--float_dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--beta_scale', type=float, default=1.3)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = create_model(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model'])
    head = model.adaptive_compensation.eval()
    if head is None:
        raise ValueError('The checkpoint does not enable adaptive compensation')
    head.output_scale = args.beta_scale
    dataset = FusionDataset(args.data_root, split='test', size=256)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    files = {path.stem: path for path in Path(args.float_dir).glob('*.npy')}

    rows = []
    for batch in tqdm(loader, desc='Public adaptive compensation'):
        name = Path(batch['name'][0]).stem
        if name not in files:
            raise FileNotFoundError(name)
        fused = torch.from_numpy(np.load(files[name], allow_pickle=False)).float()[None, None].to(device)
        ir = batch['ir'].to(device)
        vi = batch['vi'].to(device)
        adaptive, aux = head(fused, ir, vi)
        target = torch.maximum(ir, vi)
        fused_bands = haar_forward(fused)
        target_bands = haar_forward(target)
        fixed = haar_inverse(
            fused_bands[0],
            *(band + 0.5 * reference for band, reference in zip(fused_bands[1:], target_bands[1:])),
            output_size=fused.shape[-2:],
        )
        images = {
            variant: calibrate(image)[0, 0].cpu().numpy()
            for variant, image in (('none', fused), ('fixed', fixed), ('adaptive', adaptive))
        }
        row = {'name': batch['name'][0], 'beta_mean': float(aux['compensation_beta'].mean())}
        for variant, image in images.items():
            values = compute_fusion_metrics(image, batch['ir'][0, 0].numpy(), batch['vi'][0, 0].numpy())
            row.update({f'{variant}_{key}': values[key] for key in METRICS})
        rows.append(row)
    if not rows:
        raise ValueError('No samples evaluated')

    summary = {}
    for variant in ('none', 'fixed', 'adaptive'):
        summary[variant] = {
            key: float(np.mean([row[f'{variant}_{key}'] for row in rows])) for key in METRICS
        }
    summary['beta_mean'] = float(np.mean([row['beta_mean'] for row in rows]))
    result = {
        'checkpoint': args.ckpt,
        'data_root': args.data_root,
        'float_dir': args.float_dir,
        'beta_scale': args.beta_scale,
        'sample_count': len(rows),
        'summary': summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    fieldnames = ['name', 'beta_mean']
    fieldnames.extend(f'{variant}_{key}' for variant in ('none', 'fixed', 'adaptive') for key in METRICS)
    with open(output.with_suffix('.csv'), 'w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
