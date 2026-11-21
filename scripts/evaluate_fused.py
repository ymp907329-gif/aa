import argparse
import csv
import glob
import json
import os

import numpy as np
from PIL import Image

from data.dataset import FusionDataset
from utils.metrics import compute_fusion_metrics


def index_images(folder):
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    images = {}
    for path in sorted(glob.glob(os.path.join(folder, '*.*'))):
        if not path.lower().endswith(valid_ext):
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        if name in images:
            raise ValueError(f'Duplicate fused image stem: {name}')
        images[name] = path
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--fused_dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--output_norm', choices=('identity', 'minmax'), default='identity')
    args = parser.parse_args()
    size = None if args.size == 0 else args.size
    dataset = FusionDataset(args.data_root, split='test', size=size)
    fused_images = index_images(args.fused_dir)
    missing = [name for name in dataset.names if name not in fused_images]
    if missing:
        raise FileNotFoundError(f'Missing fused images: {missing}')
    metric_keys = ('psnr', 'ssim', 'vif', 'entropy', 'sd', 'ag', 'sf', 'mi', 'qabf')
    rows = []
    for sample in dataset:
        name = os.path.splitext(sample['name'])[0]
        with Image.open(fused_images[name]) as image:
            image = image.convert('L')
            if size is not None and image.size != (size, size):
                image = image.resize((size, size), Image.BILINEAR)
            fused = np.array(image, dtype=np.float32) / 255.0
        if args.output_norm == 'minmax':
            fused = (fused - fused.min()) / max(float(fused.max() - fused.min()), 1e-6)
        values = compute_fusion_metrics(fused, sample['ir'].numpy(), sample['vi'].numpy())
        rows.append({'name': sample['name'], **values})
    results = {key: float(np.mean([row[key] for row in rows])) for key in metric_keys}
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    csv_path = os.path.splitext(args.output)[0] + '.csv'
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=('name', *metric_keys))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f'Saved: {args.output}, {csv_path}')


if __name__ == '__main__':
    main()
