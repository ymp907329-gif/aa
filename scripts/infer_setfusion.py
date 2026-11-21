import argparse
import csv
import importlib.util
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETFUSION_ROOT = Path(r'C:\Users\Administrator\Desktop\代码库\SETFusion-main')


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default=str(SETFUSION_ROOT))
    parser.add_argument('--ckpt', default=str(PROJECT_ROOT / 'checkpoints' / 'baselines' / 'setfusion' / 'model.pth'))
    parser.add_argument('--data_root', default='data/MSRS')
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_batches', type=int)
    parser.add_argument('--size', type=int, default=256)
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
    spec = importlib.util.spec_from_file_location('setfusion_public', repo / 'code' / 'Networks' / 'net.py')
    network = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(network)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = network.MODEL(in_channel=2).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    dataset = FusionDataset(str(data_root), split='test', size=None if args.size == 0 else args.size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    save_dir.mkdir(parents=True, exist_ok=True)
    metric_keys = ('psnr', 'ssim', 'vif', 'entropy', 'sd', 'ag', 'sf', 'mi', 'qabf')
    rows = []
    for index, batch in enumerate(tqdm(loader, desc='SETFusion')):
        if args.max_batches is not None and index >= args.max_batches:
            break
        ir = batch['ir'][0]
        vi = batch['vi'][0]
        height, width = ir.shape[-2:]
        height_pad = (height // 8 + 1) * 8 - height
        width_pad = (width // 8 + 1) * 8 - width
        ir = torch.cat([ir, torch.flip(ir, [1])], 1)[:, :height + height_pad, :]
        ir = torch.cat([ir, torch.flip(ir, [2])], 2)[:, :, :width + width_pad]
        vi = torch.cat([vi, torch.flip(vi, [1])], 1)[:, :height + height_pad, :]
        vi = torch.cat([vi, torch.flip(vi, [2])], 2)[:, :, :width + width_pad]
        inputs = torch.cat((ir, vi), 0).unsqueeze(0).to(device)
        fused = model(inputs)[..., :height, :width]
        fused_image = np.squeeze(fused.detach().cpu().numpy())
        image = (fused_image * 255).astype(np.uint8)
        output_name = os.path.splitext(batch['name'][0])[0] + '.png'
        Image.fromarray(image).save(save_dir / output_name)
        float_dir = save_dir / 'float'
        float_dir.mkdir(exist_ok=True)
        np.save(float_dir / (Path(output_name).stem + '.npy'), fused_image.astype(np.float32))
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
