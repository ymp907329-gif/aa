import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from data.dataset import FusionDataset
from utils.metrics import compute_fusion_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--float_dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--output_norm', choices=('identity', 'minmax'), default='identity')
    parser.add_argument('--display_dir')
    args = parser.parse_args()
    dataset = FusionDataset(args.data_root, split='test', size=None if args.size == 0 else args.size)
    files = {path.stem: path for path in Path(args.float_dir).glob('*.npy')}
    missing = [name for name in dataset.names if name not in files]
    if missing:
        raise FileNotFoundError(f'Missing float images: {missing[:10]}')
    metric_keys = ('psnr', 'ssim', 'vif', 'entropy', 'sd', 'ag', 'sf', 'mi', 'qabf')
    rows = []
    for sample in dataset:
        name = Path(sample['name']).stem
        fused = np.load(files[name], allow_pickle=False).astype(np.float32)
        if fused.ndim == 3 and fused.shape[0] == 1:
            fused = fused[0]
        if fused.ndim == 3 and fused.shape[-1] == 3:
            fused = 0.299 * fused[..., 0] + 0.587 * fused[..., 1] + 0.114 * fused[..., 2]
        if fused.ndim != 2:
            raise ValueError(f'Expected single-channel output for {name}, got {fused.shape}')
        fused = np.clip(fused, 0, 1)
        if args.size != 0 and fused.shape != (args.size, args.size):
            image = Image.fromarray(fused, mode='F').resize((args.size, args.size), Image.BILINEAR)
            fused = np.array(image, dtype=np.float32)
        if args.output_norm == 'minmax':
            fused = (fused - fused.min()) / max(float(fused.max() - fused.min()), 1e-6)
        if args.display_dir:
            display_dir = Path(args.display_dir)
            display_dir.mkdir(parents=True, exist_ok=True)
            display = (np.clip(fused, 0, 1) * 255).round().astype(np.uint8)
            Image.fromarray(display).save(display_dir / f'{name}.png')
        values = compute_fusion_metrics(fused, sample['ir'].numpy(), sample['vi'].numpy())
        rows.append({'name': sample['name'], **values})
    if not rows:
        raise ValueError('No float images evaluated')
    results = {key: float(np.mean([row[key] for row in rows])) for key in metric_keys}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    with open(output.with_suffix('.csv'), 'w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=('name', *metric_keys))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f'Saved: {output}, {output.with_suffix(".csv")}')


if __name__ == '__main__':
    main()
