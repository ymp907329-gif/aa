import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import FusionDataset
from scripts.run_value_controlled import ControlledFusion
from scripts.search_two_stage_compensation import compensate
from scripts.evaluate_high_frequency_compensation import haar, inverse_haar
from utils.config import create_model
from utils.metrics import compute_fusion_metrics
from utils.value_control import ValueDepthController


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--controller')
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--full_alpha', type=float, required=True)
    parser.add_argument('--high_beta', type=float, required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--skip_minmax', action='store_true')
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = checkpoint['config']
    model = create_model(cfg).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    if args.controller:
        model = ControlledFusion(model, ValueDepthController(args.controller)).to(device)
    dataset = FusionDataset(args.data_root, split=args.split, size=cfg['data']['size'])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metric_keys = ('psnr', 'ssim', 'vif', 'entropy', 'sd', 'ag', 'sf', 'mi', 'qabf')
    rows = []
    for batch in tqdm(loader, desc='Apply source compensation'):
        ir = batch['ir'][0, 0].cpu().numpy()
        vi = batch['vi'][0, 0].cpu().numpy()
        fused = model(batch['ir'].to(device), batch['vi'].to(device))[0, 0].cpu().numpy()
        if args.skip_minmax:
            target = np.maximum(ir, vi)
            fused = (1 - args.full_alpha) * fused + args.full_alpha * target
            bands = haar(fused)
            target_bands = haar(target)
            high = np.array(bands[1:]) + args.high_beta * np.array(target_bands[1:])
            calibrated = inverse_haar((bands[0], *high))
        else:
            fused = (fused - fused.min()) / max(float(fused.max() - fused.min()), 1e-6)
            fused = (np.clip(fused, 0, 1) * 255).round().astype(np.uint8).astype(np.float64) / 255.0
            calibrated = compensate(fused, ir, vi, args.full_alpha, args.high_beta)
        image = (np.clip(calibrated, 0, 1) * 255).round().astype(np.uint8)
        output_name = os.path.splitext(batch['name'][0])[0] + '.png'
        Image.fromarray(image).save(save_dir / output_name)
        float_dir = save_dir / 'float'
        float_dir.mkdir(exist_ok=True)
        np.save(float_dir / (Path(output_name).stem + '.npy'), np.clip(calibrated, 0, 1).astype(np.float32))
        values = compute_fusion_metrics(image.astype(np.float64) / 255.0, ir, vi)
        rows.append({'name': batch['name'][0], **values})
    if not rows:
        raise ValueError('No images evaluated')
    results = {key: float(np.mean([row[key] for row in rows])) for key in metric_keys}
    results['full_alpha'] = args.full_alpha
    results['high_beta'] = args.high_beta
    if args.controller:
        results['mean_depth'] = float(np.mean(model.history))
    with open(save_dir / 'metrics.json', 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
