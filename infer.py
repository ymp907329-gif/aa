import argparse
import os

import numpy as np
from PIL import Image
import torch

from utils.config import apply_ablation, create_model, load_config


def load_image(path):
    with Image.open(path) as image:
        array = np.asarray(image.convert('L'), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ir', required=True)
    parser.add_argument('--vi', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--output', default='fused.png')
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--ablation', choices=('baseline', 'aid', 'dfd', 'freqcl', 'full'))
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get('config', load_config(args.config))
    if args.ablation:
        cfg = apply_ablation(cfg, args.ablation)
    model = create_model(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    ir = load_image(args.ir).to(device)
    vi = load_image(args.vi).to(device)
    if ir.shape != vi.shape:
        raise ValueError(f'Input size mismatch: IR={tuple(ir.shape)}, VI={tuple(vi.shape)}')
    fused = model(ir, vi)[0, 0].cpu().numpy()
    image = Image.fromarray((fused.clip(0, 1) * 255).round().astype(np.uint8))
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    image.save(args.output)
    print(f'Saved fused image: {args.output}')


if __name__ == '__main__':
    main()
