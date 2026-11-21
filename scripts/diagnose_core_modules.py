import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import FusionDataset
from utils.config import create_model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_batches', type=int)
    parser.add_argument('--trace_iterations', action='store_true')
    args = parser.parse_args()
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')
    checkpoint = torch.load(args.ckpt, map_location=device)
    cfg = checkpoint['config']
    cfg['data']['root'] = args.data_root
    model = create_model(cfg).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    dataset = FusionDataset(
        args.data_root, split=cfg['data']['test_split'], size=cfg['data']['size']
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    records = []
    step_counts = {}
    reconstruction_errors = []
    iteration_traces = []
    iteration_l2_traces = []
    for index, batch in enumerate(tqdm(loader, desc='Core diagnostics')):
        if args.max_batches is not None and index >= args.max_batches:
            break
        ir = batch['ir'].to(device)
        vi = batch['vi'].to(device)
        out, aux = model(ir, vi, return_aux=True)
        aid = aux['aid']
        dfd = aux['dfd']
        steps_map = aid['steps_map'].float().view(-1).cpu()
        values, counts = torch.unique(steps_map, return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist()):
            key = str(int(value))
            step_counts[key] = step_counts.get(key, 0) + count
        record = {
            'name': batch['name'][0],
            'mean_steps': float(steps_map.mean()),
            'min_steps': float(steps_map.min()),
            'max_steps': float(steps_map.max()),
            'consistency': float(aid['consistency'].cpu()),
            'stop_score': float(aid['stop_score'].cpu()),
            'mean_offset': float(dfd['offset'].abs().mean().cpu()),
            'max_offset': float(dfd['offset'].abs().max().cpu()),
            'output_mean': float(out.mean().cpu()),
            'output_std': float(out.std().cpu()),
            'output_min': float(out.min().cpu()),
            'output_max': float(out.max().cpu()),
        }
        records.append(record)
        if args.trace_iterations and model.aid is not None:
            _, _, feature_ir = model.encoder_ir(ir)
            _, _, feature_vi = model.encoder_vi(vi)
            _, _, trace_aux = model.aid(feature_ir, feature_vi, return_steps=True)
            previous_ir, previous_vi = feature_ir, feature_vi
            trace = []
            l2_trace = []
            for step_ir, step_vi in trace_aux['iteration_features']:
                change = 0.5 * (
                    model.aid._change(step_ir, previous_ir)
                    + model.aid._change(step_vi, previous_vi)
                )
                change_l2 = torch.sqrt(
                    (step_ir - previous_ir).square().sum(dim=(1, 2, 3))
                    + (step_vi - previous_vi).square().sum(dim=(1, 2, 3))
                )
                trace.append(float(change.mean().cpu()))
                l2_trace.append(float(change_l2.mean().cpu()))
                previous_ir, previous_vi = step_ir, step_vi
            iteration_traces.append(trace)
            iteration_l2_traces.append(l2_trace)
        if model.fusion.lifting is not None:
            _, _, features = model.encoder_ir(ir)
            bands = model.fusion.lifting(features)
            reconstructed = model.fusion.lifting.inverse(*bands, output_size=features.shape[-2:])
            reconstruction_errors.append(float((reconstructed - features).abs().max().cpu()))
    if not records:
        raise ValueError('No samples were evaluated')
    numeric_keys = tuple(key for key in records[0] if key != 'name')
    summary = {
        'checkpoint': args.ckpt,
        'data_root': args.data_root,
        'sample_count': len(records),
        'wavelet': model.fusion.wavelet,
        'region_adaptive': model.aid.region_adaptive,
        'nonexpansive': model.aid.nonexpansive if model.aid is not None else None,
        'step_counts_pixels': step_counts,
        'step_fraction_pixels': {
            key: count / sum(step_counts.values()) for key, count in sorted(step_counts.items())
        },
        'mean': {key: float(np.mean([row[key] for row in records])) for key in numeric_keys},
        'std': {key: float(np.std([row[key] for row in records])) for key in numeric_keys},
        'lifting_reconstruction_max_abs': {
            'mean': float(np.mean(reconstruction_errors)) if reconstruction_errors else None,
            'max': float(np.max(reconstruction_errors)) if reconstruction_errors else None,
        },
    }
    if iteration_traces:
        depth = max(len(trace) for trace in iteration_traces)
        mean_trace = [
            float(np.mean([trace[index] for trace in iteration_traces if len(trace) > index]))
            for index in range(depth)
        ]
        mean_l2_trace = [
            float(np.mean([trace[index] for trace in iteration_l2_traces if len(trace) > index]))
            for index in range(depth)
        ]
        summary['iteration_convergence'] = {
            'relative_l1_mean': mean_trace,
            'relative_l1_monotonic_sample_fraction': float(np.mean([
                all(later <= earlier + 1e-8 for earlier, later in zip(trace, trace[1:]))
                for trace in iteration_traces
            ])),
            'relative_l1_initial_to_final_ratio': mean_trace[-1] / (mean_trace[0] + 1e-12),
            'absolute_l2_mean': mean_l2_trace,
            'absolute_l2_monotonic_sample_fraction': float(np.mean([
                all(later <= earlier * (1 + 1e-6) + 1e-8 for earlier, later in zip(trace, trace[1:]))
                for trace in iteration_l2_traces
            ])),
            'absolute_l2_initial_to_final_ratio': mean_l2_trace[-1] / (mean_l2_trace[0] + 1e-12),
        }
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
