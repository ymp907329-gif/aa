import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import FusionDataset
from utils.config import create_model
from utils.metrics import compute_fusion_metrics


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--split', choices=('train', 'val', 'test'))
    parser.add_argument('--exclude_test_overlap', action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_batches', type=int)
    parser.add_argument('--modes', nargs='+', choices=('identity', 'minmax'), default=('identity', 'minmax'))
    args = parser.parse_args()
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')
    checkpoint = torch.load(args.ckpt, map_location=device)
    cfg = checkpoint['config']
    model = create_model(cfg).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    split = args.split or cfg['data']['test_split']
    exclude_names = None
    if args.exclude_test_overlap:
        if split == 'test':
            raise ValueError('Test overlap exclusion is only meaningful for non-test splits')
        exclude_names = set(FusionDataset(args.data_root, split='test', size=None).names)
    dataset = FusionDataset(
        args.data_root, split=split, size=cfg['data']['size'], exclude_names=exclude_names
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    records = []
    for index, batch in enumerate(tqdm(loader, desc='Iteration quality')):
        if args.max_batches is not None and index >= args.max_batches:
            break
        ir = batch['ir'].to(device)
        vi = batch['vi'].to(device)
        candidates = model.forward_iterations(ir, vi)
        for depth, (out, _) in enumerate(candidates, start=1):
            for mode in args.modes:
                fused = out[0]
                if mode == 'minmax':
                    fused = (fused - fused.min()) / (fused.max() - fused.min()).clamp_min(1e-6)
                values = compute_fusion_metrics(
                    fused.cpu().numpy(), ir[0].cpu().numpy(), vi[0].cpu().numpy()
                )
                records.append({'name': batch['name'][0], 'depth': depth, 'output': mode, **values})
    if not records:
        raise ValueError('No samples were evaluated')
    metric_keys = ('entropy', 'sd', 'sf', 'mi', 'vif', 'qabf')
    depths = sorted({record['depth'] for record in records})
    summary = {'checkpoint': args.ckpt, 'data_root': args.data_root, 'sample_count': len(loader), 'results': {}}
    for mode in args.modes:
        summary['results'][mode] = {}
        for depth in depths:
            selected = [record for record in records if record['output'] == mode and record['depth'] == depth]
            summary['results'][mode][str(depth)] = {
                metric: float(np.mean([record[metric] for record in selected])) for metric in metric_keys
            }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output) + '.json', 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    with open(str(output) + '.csv', 'w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary['results'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
