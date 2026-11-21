import json
from pathlib import Path

import numpy as np

from data.dataset import FusionDataset
from utils.metrics import compute_fusion_metrics


def main():
    output = {}
    for dataset in ('MSRS', 'RoadScene', 'TNO', 'VIFB'):
        name = dataset.lower()
        samples = FusionDataset(f'data/{dataset}', split='test', size=256)
        output[name] = {}
        for mode in ('average', 'maximum'):
            rows = []
            for sample in samples:
                ir = sample['ir'].numpy()
                vi = sample['vi'].numpy()
                fused = 0.5 * (ir + vi) if mode == 'average' else np.maximum(ir, vi)
                rows.append(compute_fusion_metrics(fused, ir, vi))
            output[name][mode] = {
                key: float(np.mean([row[key] for row in rows]))
                for key in ('psnr', 'ssim', 'vif', 'entropy', 'sd', 'ag', 'sf', 'mi', 'qabf')
            }
            print(name, mode, output[name][mode])
    path = Path('results/comparison/learning_free_float.json')
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved: {path}')


if __name__ == '__main__':
    main()
