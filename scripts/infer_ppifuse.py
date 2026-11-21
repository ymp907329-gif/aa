import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import FusionDataset
from utils.metrics import compute_fusion_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PPIFUSE_ROOT = Path(r'C:\Users\Administrator\Desktop\代码库\PPIFuse-master')


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default=str(PPIFUSE_ROOT))
    parser.add_argument('--ckpt', default=str(PROJECT_ROOT / 'checkpoints' / 'baselines' / 'ppifuse' / 'PPIFuse_10.pth'))
    parser.add_argument('--data_root', default='data/MSRS')
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_batches', type=int)
    args = parser.parse_args()
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = PROJECT_ROOT / save_dir
    ckpt = Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = PROJECT_ROOT / ckpt
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    import importlib
    import types
    package = types.ModuleType('ppifuse_models')
    package.__path__ = [str(repo / 'models')]
    sys.modules['ppifuse_models'] = package
    backbone = types.ModuleType('ppifuse_models.backbone')
    backbone.__path__ = [str(repo / 'models' / 'backbone')]
    sys.modules['ppifuse_models.backbone'] = backbone
    fusion_unet = importlib.import_module('ppifuse_models.backbone.FusionUNet')

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = fusion_unet.Net().to(device)
    checkpoint = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['net'], strict=True)
    model.eval()
    dataset = FusionDataset(str(data_root), split='test', size=256)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    save_dir.mkdir(parents=True, exist_ok=True)
    metric_keys = ('psnr', 'ssim', 'vif', 'entropy', 'sd', 'ag', 'sf', 'mi', 'qabf')
    rows = []
    for index, batch in enumerate(tqdm(loader, desc='PPIFuse')):
        if args.max_batches is not None and index >= args.max_batches:
            break
        ir = batch['ir'].to(device)
        vi = batch['vi'].to(device)
        height, width = ir.shape[-2:]
        height_pad = (64 - height % 64) % 64
        width_pad = (64 - width % 64) % 64
        if height_pad:
            ir = torch.cat([ir, torch.flip(ir, [2])], 2)[:, :, :height + height_pad, :]
            vi = torch.cat([vi, torch.flip(vi, [2])], 2)[:, :, :height + height_pad, :]
        if width_pad:
            ir = torch.cat([ir, torch.flip(ir, [3])], 3)[:, :, :, :width + width_pad]
            vi = torch.cat([vi, torch.flip(vi, [3])], 3)[:, :, :, :width + width_pad]
        fused = model(ir, vi)[..., :height, :width]
        raw_fused = ((fused + 1) / 2).clamp(0, 1)
        fused = (fused - fused.min()) / (fused.max() - fused.min()).clamp_min(1e-8)
        fused_image = fused[0, 0].cpu().numpy()
        image = (fused_image.clip(0, 1) * 255).round().astype(np.uint8)
        output_name = os.path.splitext(batch['name'][0])[0] + '.png'
        Image.fromarray(image).save(save_dir / output_name)
        float_dir = save_dir / 'float'
        float_dir.mkdir(exist_ok=True)
        np.save(float_dir / (Path(output_name).stem + '.npy'), raw_fused[0, 0].detach().cpu().numpy().astype(np.float32))
        quantized = image.astype(np.float32) / 255.0
        values = compute_fusion_metrics(quantized, batch['ir'][0].numpy(), batch['vi'][0].numpy())
        rows.append({'name': batch['name'][0], **values})
    if not rows:
        raise ValueError('No images were evaluated')
    results = {key: float(np.mean([row[key] for row in rows])) for key in metric_keys}
    with open(save_dir / 'metrics.json', 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    with open(save_dir / 'metrics.csv', 'w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=('name', *metric_keys))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
