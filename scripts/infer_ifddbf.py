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

from data.dataset import FusionDataset
from utils.metrics import compute_fusion_metrics

IFD_ROOT = Path(r'C:\Users\Administrator\Desktop\代码库\IFD-DBF-main')


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default=str(IFD_ROOT))
    parser.add_argument('--ckpt')
    parser.add_argument('--data_root', default='data/MSRS')
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_batches', type=int)
    args = parser.parse_args()
    repo = Path(args.repo)
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, FeatureUpdater
    from DBfusion import DBFusion

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    ckpt_path = args.ckpt or str(repo / 'models' / 'test.pth')
    checkpoint = torch.load(ckpt_path, map_location=device)
    encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    decoder = nn.DataParallel(Restormer_Decoder()).to(device)
    base_fuse = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
    updater = nn.DataParallel(FeatureUpdater()).to(device)
    detail_fuse = nn.DataParallel(DBFusion(64, 2, solver='anderson')).to(device)
    encoder.load_state_dict(checkpoint['DIDF_Encoder'])
    decoder.load_state_dict(checkpoint['DIDF_Decoder'])
    base_fuse.load_state_dict(checkpoint['BaseFuseLayer'])
    detail_fuse.load_state_dict(checkpoint['DetailFuseLayer'])
    updater.load_state_dict(checkpoint['F_Encoder'])
    for module in (encoder, decoder, base_fuse, updater, detail_fuse):
        module.eval()

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = Path(r'C:\Users\Administrator\PycharmProjects\IVF-DecompFuse') / data_root
    dataset = FusionDataset(str(data_root), split='test', size=256)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = Path(r'C:\Users\Administrator\PycharmProjects\IVF-DecompFuse') / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    metric_keys = ('psnr', 'ssim', 'vif', 'entropy', 'sd', 'ag', 'sf', 'mi', 'qabf')
    rows = []
    for index, batch in enumerate(tqdm(loader, desc='IFD-DBF')):
        if args.max_batches is not None and index >= args.max_batches:
            break
        ir, vi = batch['ir'].to(device), batch['vi'].to(device)
        feature_v_b, feature_v_d, _ = encoder(vi)
        feature_i_b, feature_i_d, _ = encoder(ir)
        feature_jv_b, feature_jv_d, feature_ji_b, feature_ji_d = updater(
            feature_v_b, feature_v_d, feature_i_b, feature_i_d
        )
        feature_f_b = base_fuse(feature_jv_b + feature_ji_b)
        feature_f_d, _, _ = detail_fuse([feature_ji_d, feature_jv_d])
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
        Image.fromarray(image).save(save_dir / output_name)
        float_dir = save_dir / 'float'
        float_dir.mkdir(exist_ok=True)
        np.save(float_dir / (Path(output_name).stem + '.npy'), raw_fused[0, 0].detach().cpu().numpy().astype(np.float32))

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
