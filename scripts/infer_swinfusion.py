import importlib.util
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import FusionDataset
from utils.metrics import compute_fusion_metrics


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default='external/SwinFusion')
    parser.add_argument('--ckpt')
    parser.add_argument('--data_root', default='data/MSRS')
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_batches', type=int)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    network_path = repo / 'models' / 'network_swinfusion1.py'
    spec = importlib.util.spec_from_file_location('swinfusion_official', network_path)
    network = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(network)
    SwinFusion = network.SwinFusion

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    ckpt_path = args.ckpt or str(
        repo / 'Model' / 'Infrared_Visible_Fusion' / 'Infrared_Visible_Fusion' / 'models' / '10000_E.pth'
    )
    model = SwinFusion(
        upscale=1, in_chans=1, img_size=128, window_size=8, img_range=1.0,
        depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
        mlp_ratio=2, upsampler=None, resi_connection='1conv',
    )
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint.get('params', checkpoint), strict=True)
    model.eval().to(device)
    dataset = FusionDataset(args.data_root, split='test', size=256)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    os.makedirs(args.save_dir, exist_ok=True)
    metric_keys = ('psnr', 'ssim', 'vif', 'entropy', 'sd', 'ag', 'sf', 'mi', 'qabf')
    rows = []
    for index, batch in enumerate(tqdm(loader, desc='SwinFusion')):
        if args.max_batches is not None and index >= args.max_batches:
            break
        ir = batch['ir'].to(device)
        vi = batch['vi'].to(device)
        fused = model(ir, vi).clamp(0, 1)
        fused_image = fused[0].cpu().numpy()
        values = compute_fusion_metrics(fused_image, batch['ir'][0].numpy(), batch['vi'][0].numpy())
        rows.append({'name': batch['name'][0], **values})
        image = (fused_image.squeeze(0) * 255).round().astype(np.uint8)
        output_name = os.path.splitext(batch['name'][0])[0] + '.png'
        Image.fromarray(image).save(os.path.join(args.save_dir, output_name))
        float_dir = Path(args.save_dir) / 'float'
        float_dir.mkdir(exist_ok=True)
        np.save(float_dir / (Path(output_name).stem + '.npy'), np.squeeze(fused_image).astype(np.float32))
    if not rows:
        raise ValueError('No images were evaluated')
    results = {key: float(np.mean([row[key] for row in rows])) for key in metric_keys}
    with open(os.path.join(args.save_dir, 'metrics.json'), 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    with open(os.path.join(args.save_dir, 'metrics.csv'), 'w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=('name', *metric_keys))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
