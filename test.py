import argparse
import csv
import json
import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import FusionDataset
from utils.config import apply_ablation, create_model, load_config
from utils.efficiency import measure_efficiency
from utils.metrics import compute_fusion_metrics


@torch.no_grad()
def evaluate(model, loader, device, save_dir=None, max_batches=None, output_norm='identity'):
    model.eval()
    metric_keys = ('psnr', 'ssim', 'vif', 'entropy', 'sd', 'ag', 'sf', 'mi', 'qabf')
    results = {key: [] for key in metric_keys}
    per_image = []
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    for index, batch in enumerate(tqdm(loader, desc='Testing')):
        if max_batches is not None and index >= max_batches:
            break
        ir = batch['ir'].to(device)
        vi = batch['vi'].to(device)
        out = model(ir, vi)
        raw_out = out
        if output_norm == 'minmax':
            data_min, data_max = torch.min(out), torch.max(out)
            out = (out - data_min) / (data_max - data_min).clamp_min(1e-6)
        for i in range(out.shape[0]):
            fused = out[i].cpu().numpy()
            values = compute_fusion_metrics(fused, ir[i].cpu().numpy(), vi[i].cpu().numpy())
            for key, value in values.items():
                results[key].append(value)
            per_image.append({'name': batch['name'][i], **values})
            if save_dir:
                image = (fused.squeeze(0).clip(0, 1) * 255).round().astype(np.uint8)
                output_name = os.path.splitext(batch['name'][i])[0] + '.png'
                Image.fromarray(image).save(os.path.join(save_dir, output_name))
                float_dir = os.path.join(save_dir, 'float')
                os.makedirs(float_dir, exist_ok=True)
                np.save(os.path.join(float_dir, os.path.splitext(output_name)[0] + '.npy'),
                        raw_out[i].detach().cpu().numpy().astype(np.float32))
    if not results['psnr']:
        raise ValueError('No test samples were evaluated')
    if save_dir:
        with open(os.path.join(save_dir, 'metrics.csv'), 'w', encoding='utf-8-sig', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=('name', *metric_keys))
            writer.writeheader()
            writer.writerows(per_image)
    return {key: float(np.mean(values)) for key, values in results.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ablation', choices=('baseline', 'aid', 'dfd', 'freqcl', 'full'))
    parser.add_argument('--data_root')
    parser.add_argument('--band_mask', nargs=4, type=int, choices=(0, 1))
    parser.add_argument('--output_norm', choices=('identity', 'minmax'), default='identity')
    parser.add_argument('--zero_alignment', action='store_true')
    parser.add_argument('--incremental_detail', action='store_true')
    parser.add_argument('--high_fusion_mode', choices=('learned', 'energy', 'mix'))
    parser.add_argument('--energy_mix', type=float)
    parser.add_argument('--stop_quantile', type=float)
    parser.add_argument('--aid_output_scale', type=float)
    parser.add_argument('--save_dir')
    parser.add_argument('--max_batches', type=int)
    parser.add_argument('--efficiency', action='store_true')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get('config', load_config(args.config))
    if args.ablation:
        cfg = apply_ablation(cfg, args.ablation)
    if args.data_root:
        cfg['data']['root'] = args.data_root
    if args.incremental_detail:
        cfg['model'].setdefault('aid', {})['incremental_detail'] = True
    if args.stop_quantile is not None:
        cfg['model'].setdefault('aid', {})['stop_quantile'] = args.stop_quantile
    if args.aid_output_scale is not None:
        cfg['model'].setdefault('aid', {})['output_scale'] = args.aid_output_scale
    if args.high_fusion_mode:
        cfg['model'].setdefault('dfd', {})['high_fusion_mode'] = args.high_fusion_mode
    if args.energy_mix is not None:
        cfg['model'].setdefault('dfd', {})['high_energy_mix'] = args.energy_mix
    if args.band_mask:
        cfg['model'].setdefault('dfd', {})['band_mask'] = args.band_mask
    test_ds = FusionDataset(cfg['data']['root'], split=cfg['data']['test_split'], size=cfg['data']['size'])
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=cfg['data'].get('num_workers', 0)
    )
    model = create_model(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    if args.zero_alignment:
        if not hasattr(model.fusion, 'max_offset'):
            raise ValueError('Zero alignment requires DFD')
        model.fusion.max_offset = 0.0
    print(f'Loaded checkpoint: {args.ckpt} (epoch {ckpt.get("epoch", "?")})')

    results = evaluate(model, test_loader, device, args.save_dir, args.max_batches, args.output_norm)
    print('\n===== Test Results =====')
    for key, value in results.items():
        print(f'{key:>10}: {value:.6f}')
    if args.efficiency:
        sample = test_ds[0]
        ir = sample['ir'].unsqueeze(0).to(device)
        vi = sample['vi'].unsqueeze(0).to(device)
        efficiency = measure_efficiency(model, ir, vi, args.warmup, args.repeats)
        results['efficiency'] = efficiency
        print('\n===== Efficiency =====')
        print(json.dumps(efficiency, indent=2))
    if args.save_dir:
        with open(os.path.join(args.save_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
