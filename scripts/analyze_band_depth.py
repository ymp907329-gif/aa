import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import FusionDataset
from models.modules.frequency import haar_forward, haar_inverse
from test import evaluate
from utils.config import create_model


class BandDepthModel(nn.Module):
    def __init__(self, model, low_depth, high_depth):
        super().__init__()
        self.model = model
        self.low_depth = low_depth
        self.high_depth = high_depth

    def forward(self, ir, vi):
        model = self.model
        f0_ir, _, feature_ir = model.encoder_ir(ir)
        f0_vi, _, feature_vi = model.encoder_vi(vi)
        _, _, aux = model.aid(feature_ir, feature_vi, return_steps=True)
        iterations = aux['iteration_features']
        low_ir, low_vi = iterations[self.low_depth - 1]
        high_ir, high_vi = iterations[self.high_depth - 1]
        low_bands_ir = haar_forward(low_ir)
        low_bands_vi = haar_forward(low_vi)
        high_bands_ir = haar_forward(high_ir)
        high_bands_vi = haar_forward(high_vi)
        mixed_ir = haar_inverse(
            low_bands_ir[0], high_bands_ir[1], high_bands_ir[2], high_bands_ir[3],
            output_size=low_ir.shape[-2:],
        )
        mixed_vi = haar_inverse(
            low_bands_vi[0], high_bands_vi[1], high_bands_vi[2], high_bands_vi[3],
            output_size=low_vi.shape[-2:],
        )
        fused, _ = model.fusion(mixed_ir, mixed_vi, return_aux=True)
        if model.use_skip:
            return model.decoder(fused, f0_ir, f0_vi)
        return model.decoder(fused)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--low_depth', type=int, default=4)
    parser.add_argument('--high_depth', type=int, default=1)
    parser.add_argument('--output_norm', choices=('identity', 'minmax'), default='identity')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_batches', type=int)
    args = parser.parse_args()
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')
    checkpoint = torch.load(args.ckpt, map_location=device)
    cfg = checkpoint['config']
    model = create_model(cfg).to(device)
    model.load_state_dict(checkpoint['model'])
    if model.aid is None or not model.use_dfd:
        raise ValueError('Band-depth analysis requires AID and DFD')
    depth = model.aid.max_steps
    if not 1 <= args.low_depth <= depth or not 1 <= args.high_depth <= depth:
        raise ValueError(f'Depths must be in [1, {depth}]')
    wrapper = BandDepthModel(model, args.low_depth, args.high_depth).eval()
    dataset = FusionDataset(args.data_root, split=cfg['data']['test_split'], size=cfg['data']['size'])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    results = evaluate(wrapper, loader, device, args.save_dir, args.max_batches, args.output_norm)
    print(results)


if __name__ == '__main__':
    main()
