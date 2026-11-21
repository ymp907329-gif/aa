import argparse
import json
import os

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from scripts.analyze_iteration_value import fusion_value
from train import make_loaders
from utils.config import create_model


def contribution_map(contribution):
    value = contribution.squeeze().cpu().numpy()
    scale = max(float(np.percentile(np.abs(value), 99)), 1e-6)
    normalized = np.clip(value / scale, -1, 1)
    red = (127 + 128 * normalized).astype(np.uint8)
    blue = (127 - 128 * normalized).astype(np.uint8)
    green = (127 - 128 * np.abs(normalized)).astype(np.uint8)
    return np.stack((red, green, blue), axis=-1)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='checkpoints/ablation/region_lifting_miqabf/best.pth')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output', default='results/research/counterfactual_value/e4')
    parser.add_argument('--max_batches', type=int, default=8)
    parser.add_argument('--value_pool', type=int, default=16)
    parser.add_argument('--num_examples', type=int, default=2)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    device = torch.device(
        'cpu' if args.gpu < 0 else f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    )
    checkpoint = torch.load(args.ckpt, map_location=device)
    model = create_model(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    _, val_loader = make_loaders(checkpoint['config'])

    iteration_rows = []
    best_counts = None
    example_count = 0
    for index, batch in enumerate(tqdm(val_loader, desc='Counterfactual value')):
        if args.max_batches is not None and index >= args.max_batches:
            break
        ir, vi = batch['ir'].to(device), batch['vi'].to(device)
        f0_ir, _, feature_ir = model.encoder_ir(ir)
        f0_vi, _, feature_vi = model.encoder_vi(vi)
        _, _, aid_aux = model.aid(feature_ir, feature_vi, return_steps=True)
        cross_iterations = []
        for step_ir, step_vi in aid_aux['iteration_features']:
            fused = model.decoder(model.fusion(step_ir, step_vi), f0_ir, f0_vi)
            no_ir = model.decoder(
                model.fusion(torch.zeros_like(step_ir), step_vi), torch.zeros_like(f0_ir), f0_vi
            )
            no_vi = model.decoder(
                model.fusion(step_ir, torch.zeros_like(step_vi)), f0_ir, torch.zeros_like(f0_vi)
            )
            full_value, _ = fusion_value(fused, ir, vi)
            no_ir_value, _ = fusion_value(no_ir, ir, vi)
            no_vi_value, _ = fusion_value(no_vi, ir, vi)
            c_ir = full_value - no_ir_value
            c_vi = full_value - no_vi_value
            cross = torch.minimum(c_ir, c_vi)
            if args.value_pool > 1:
                pool_args = {'kernel_size': args.value_pool, 'stride': args.value_pool}
                c_ir = torch.nn.functional.avg_pool2d(c_ir, **pool_args)
                c_vi = torch.nn.functional.avg_pool2d(c_vi, **pool_args)
                cross = torch.nn.functional.avg_pool2d(cross, **pool_args)
            cross_iterations.append((c_ir, c_vi, cross))

        best = torch.stack([item[2] for item in cross_iterations], dim=1).argmax(dim=1)
        counts = torch.bincount(best.flatten(), minlength=len(cross_iterations)).cpu()
        best_counts = counts if best_counts is None else best_counts + counts
        for iteration, (c_ir, c_vi, cross) in enumerate(cross_iterations, start=1):
            row = {
                'iteration': iteration,
                'c_ir_mean': c_ir.mean().item(),
                'c_vi_mean': c_vi.mean().item(),
                'cross_mean': cross.mean().item(),
                'c_ir_negative_fraction': (c_ir < 0).float().mean().item(),
                'c_vi_negative_fraction': (c_vi < 0).float().mean().item(),
                'cross_negative_fraction': (cross < 0).float().mean().item(),
            }
            if len(iteration_rows) < iteration:
                iteration_rows.append([])
            iteration_rows[iteration - 1].append(row)
        if example_count < args.num_examples:
            stem = os.path.splitext(batch['name'][0])[0]
            for iteration, (_, _, cross) in enumerate(cross_iterations, start=1):
                image = contribution_map(cross[0])
                Image.fromarray(image).save(os.path.join(
                    args.output, f'{stem}_cross_iteration_{iteration}.png'
                ))
            example_count += 1

    if not iteration_rows:
        raise ValueError('No validation samples were evaluated')
    iterations = []
    for iteration, rows in enumerate(iteration_rows, start=1):
        iterations.append({
            'iteration': iteration,
            **{key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != 'iteration'},
        })
    histogram = (best_counts.float() / best_counts.sum()).tolist()
    summary = {
        'checkpoint': args.ckpt,
        'samples': len(iteration_rows[0]),
        'value_pool': args.value_pool,
        'iterations': iterations,
        'best_cross_iteration_distribution': histogram,
        'interpretation': 'Positive contributions improve the fusion proxy; min(C_ir, C_vi) requires both modalities to contribute.',
    }
    with open(os.path.join(args.output, 'analysis.json'), 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
