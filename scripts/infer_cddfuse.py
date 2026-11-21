import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'external' / 'MMIF-CDDFuse'))

from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
from data.dataset import FusionDataset
from utils.metrics import compute_fusion_metrics


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='external/MMIF-CDDFuse/models/CDDFuse_IVF.pth')
    parser.add_argument('--data_root', default='data/MSRS')
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_batches', type=int)
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    decoder = nn.DataParallel(Restormer_Decoder()).to(device)
    base_fuse = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
    detail_fuse = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    encoder.load_state_dict(checkpoint.get('CDDF_Encoder', checkpoint['DIDF_Encoder']))
    decoder.load_state_dict(checkpoint.get('CDDF_Decoder', checkpoint['DIDF_Decoder']))
    base_fuse.load_state_dict(checkpoint['BaseFuseLayer'])
    detail_fuse.load_state_dict(checkpoint['DetailFuseLayer'])
    encoder.eval()
    decoder.eval()
    base_fuse.eval()
    detail_fuse.eval()

    dataset = FusionDataset(args.data_root, split='test', size=256)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    os.makedirs(args.save_dir, exist_ok=True)
    metric_keys = ('psnr', 'ssim', 'vif', 'entropy', 'sd', 'ag', 'sf', 'mi', 'qabf')
    rows = []
    for index, batch in enumerate(tqdm(loader, desc='CDDFuse')):
        if args.max_batches is not None and index >= args.max_batches:
            break
        ir, vi = batch['ir'].to(device), batch['vi'].to(device)
        feature_v_b, feature_v_d, _ = encoder(vi)
        feature_i_b, feature_i_d, _ = encoder(ir)
        feature_f_b = base_fuse(feature_v_b + feature_i_b)
        feature_f_d = detail_fuse(feature_v_d + feature_i_d)
        fused, _ = decoder(vi, feature_f_b, feature_f_d)
        raw_fused = fused
        data_min, data_max = torch.min(fused), torch.max(fused)
        fused = (fused - data_min) / (data_max - data_min).clamp_min(1e-8)
        fused_image = fused[0].cpu().numpy()
        values = compute_fusion_metrics(
            fused_image, batch['ir'][0].numpy(), batch['vi'][0].numpy()
        )
        rows.append({'name': batch['name'][0], **values})
        image = (fused_image.squeeze(0).clip(0, 1) * 255).round().astype(np.uint8)
        output_name = os.path.splitext(batch['name'][0])[0] + '.png'
        Image.fromarray(image).save(os.path.join(args.save_dir, output_name))
        float_dir = Path(args.save_dir) / 'float'
        float_dir.mkdir(exist_ok=True)
        np.save(float_dir / (Path(output_name).stem + '.npy'), raw_fused[0, 0].detach().cpu().numpy().astype(np.float32))

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
