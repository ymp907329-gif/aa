import argparse
import json
import os

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses.fusion_loss import sobel_gradient
from train import make_loaders
from utils.config import create_model
from utils.metrics import compute_fusion_metrics


def ssim_map(x, y, window_size=11, sigma=1.5):
    coords = torch.arange(window_size, device=x.device, dtype=x.dtype) - window_size // 2
    kernel_1d = torch.exp(-coords.square() / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = (kernel_1d[:, None] @ kernel_1d[None, :]).expand(1, 1, window_size, window_size)
    padding = window_size // 2
    mu_x = F.conv2d(x, kernel, padding=padding)
    mu_y = F.conv2d(y, kernel, padding=padding)
    sigma_x = F.conv2d(x * x, kernel, padding=padding) - mu_x.square()
    sigma_y = F.conv2d(y * y, kernel, padding=padding) - mu_y.square()
    sigma_xy = F.conv2d(x * y, kernel, padding=padding) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    )


def fusion_value(fused, ir, vi):
    structure = 0.5 * (ssim_map(fused, ir) + ssim_map(fused, vi))
    structure = ((structure + 1) * 0.5).clamp(0, 1)
    fused_gx, fused_gy = sobel_gradient(fused)
    ir_gx, ir_gy = sobel_gradient(ir)
    vi_gx, vi_gy = sobel_gradient(vi)
    fused_mag = torch.sqrt(fused_gx.square() + fused_gy.square() + 1e-12)
    ir_mag = torch.sqrt(ir_gx.square() + ir_gy.square() + 1e-12)
    vi_mag = torch.sqrt(vi_gx.square() + vi_gy.square() + 1e-12)
    target_mag = torch.maximum(ir_mag, vi_mag)
    gradient = (
        (torch.minimum(fused_mag, target_mag) + 1e-6)
        / (torch.maximum(fused_mag, target_mag) + 1e-6)
    ).clamp(0, 1)
    intensity = (1 - (fused - torch.maximum(ir, vi)).abs()).clamp(0, 1)
    value = 0.4 * structure + 0.4 * gradient + 0.2 * intensity
    return value, {
        'structure': structure.mean().item(),
        'gradient': gradient.mean().item(),
        'intensity': intensity.mean().item(),
    }


def save_example(output_dir, name, best, oracle, candidates):
    stem = os.path.splitext(name)[0]
    palette = np.array([
        [45, 110, 210], [40, 175, 105], [240, 190, 45], [215, 65, 55],
    ], dtype=np.uint8)
    best_np = best.squeeze().cpu().numpy().astype(np.int64)
    Image.fromarray(palette[best_np]).save(os.path.join(output_dir, f'{stem}_best_iterations.png'))
    oracle_np = (oracle.squeeze().cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
    Image.fromarray(oracle_np).save(os.path.join(output_dir, f'{stem}_oracle.png'))
    for index, candidate in enumerate(candidates, start=1):
        image = (candidate.squeeze().cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
        Image.fromarray(image).save(os.path.join(output_dir, f'{stem}_iteration_{index}.png'))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints/ablation/region_lifting_miqabf/best.pth')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output', default='results/research/iteration_value/e4')
    parser.add_argument('--max_batches', type=int)
    parser.add_argument('--num_examples', type=int, default=3)
    parser.add_argument('--value_pool', type=int, default=1)
    args = parser.parse_args()
    if args.value_pool < 1:
        raise ValueError('value_pool must be positive')
    os.makedirs(args.output, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.ckpt, map_location=device)
    model = create_model(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    if not model.use_aid or not model.use_dfd:
        raise ValueError('Iteration value analysis requires AID and DFD')

    _, val_loader = make_loaders(checkpoint['config'])
    candidate_metrics = []
    candidate_components = []
    oracle_metrics = []
    oracle_values = []
    iteration_counts = None
    image_best_counts = None
    example_count = 0
    for index, batch in enumerate(tqdm(val_loader, desc='Iteration value')):
        if args.max_batches is not None and index >= args.max_batches:
            break
        ir, vi = batch['ir'].to(device), batch['vi'].to(device)
        f0_ir, _, feature_ir = model.encoder_ir(ir)
        f0_vi, _, feature_vi = model.encoder_vi(vi)
        _, _, aid_aux = model.aid(feature_ir, feature_vi, return_steps=True)
        candidates, values, components = [], [], []
        for step_ir, step_vi in aid_aux['iteration_features']:
            fused_feature = model.fusion(step_ir, step_vi)
            fused = model.decoder(fused_feature, f0_ir, f0_vi)
            value, component = fusion_value(fused, ir, vi)
            candidates.append(fused)
            values.append(value)
            components.append(component)
        value_stack = torch.stack(values, dim=1)
        candidate_stack = torch.stack(candidates, dim=1)
        if args.value_pool > 1:
            pooled_value = F.avg_pool2d(
                value_stack.squeeze(2), kernel_size=args.value_pool, stride=args.value_pool
            )
            best = F.interpolate(
                pooled_value.argmax(dim=1, keepdim=True).float(),
                size=value_stack.shape[-2:], mode='nearest',
            ).long()
        else:
            best = value_stack.argmax(dim=1)
        best_expanded = best.unsqueeze(1).expand(-1, 1, 1, -1, -1)
        oracle = torch.gather(candidate_stack, 1, best_expanded).squeeze(1)
        oracle_value = torch.gather(value_stack, 1, best_expanded).squeeze(1)
        counts = torch.bincount(best.flatten(), minlength=len(candidates)).cpu()
        image_best = value_stack.mean(dim=(2, 3, 4)).argmax(dim=1).flatten().cpu()
        image_counts = torch.bincount(image_best, minlength=len(candidates))
        iteration_counts = counts if iteration_counts is None else iteration_counts + counts
        image_best_counts = image_counts if image_best_counts is None else image_best_counts + image_counts
        oracle_values.append(oracle_value.mean().item())

        for i, (fused, value, component) in enumerate(zip(candidates, values, components)):
            metrics = compute_fusion_metrics(
                fused[0].cpu().numpy(), ir[0].cpu().numpy(), vi[0].cpu().numpy()
            )
            metrics['value'] = value.mean().item()
            metrics['components'] = component
            if len(candidate_metrics) <= i:
                candidate_metrics.append([])
                candidate_components.append([])
            candidate_metrics[i].append(metrics)
            candidate_components[i].append(component)
        oracle_metrics.append(compute_fusion_metrics(
            oracle[0].cpu().numpy(), ir[0].cpu().numpy(), vi[0].cpu().numpy()
        ))
        if example_count < args.num_examples:
            save_example(args.output, batch['name'][0], best[0], oracle[0], [item[0] for item in candidates])
            example_count += 1

    if not oracle_metrics:
        raise ValueError('No validation samples were evaluated')
    metric_keys = ('entropy', 'sd', 'sf', 'mi', 'vif', 'qabf', 'ssim', 'psnr')
    iterations = []
    for index, rows in enumerate(candidate_metrics, start=1):
        iterations.append({
            'iteration': index,
            'metrics': {key: float(np.mean([row[key] for row in rows])) for key in metric_keys},
            'value': float(np.mean([row['value'] for row in rows])),
            'components': {
                key: float(np.mean([row['components'][key] for row in rows]))
                for key in ('structure', 'gradient', 'intensity')
            },
        })
    histogram = (iteration_counts.float() / iteration_counts.sum()).tolist()
    image_histogram = (image_best_counts.float() / image_best_counts.sum()).tolist()
    summary = {
        'checkpoint': args.ckpt,
        'samples': len(oracle_metrics),
        'value_pool': args.value_pool,
        'weights': {'structure': 0.4, 'gradient': 0.4, 'intensity': 0.2},
        'iterations': iterations,
        'region_best_iteration_distribution': histogram,
        'image_best_iteration_distribution': image_histogram,
        'oracle': {
            'metrics': {key: float(np.mean([row[key] for row in oracle_metrics])) for key in metric_keys},
            'value': float(np.mean(oracle_values)),
            'note': 'Pixel-wise oracle selected on validation data; not an inference result.',
        },
    }
    with open(os.path.join(args.output, 'analysis.json'), 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
