import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import FusionDataset
from models.modules.frequency import haar_forward, haar_inverse
from test import evaluate
from utils.config import create_model


class ClosedFormLowFrequencyModel(nn.Module):
    def __init__(self, model, coupling):
        super().__init__()
        if coupling < 0:
            raise ValueError('coupling must be nonnegative')
        self.model = model
        self.coupling = coupling

    def forward(self, ir, vi):
        model = self.model
        f0_ir, _, feature_ir = model.encoder_ir(ir)
        f0_vi, _, feature_vi = model.encoder_vi(vi)
        bands_ir = haar_forward(feature_ir)
        bands_vi = haar_forward(feature_vi)
        denominator = 1 + 2 * self.coupling
        low_ir = ((1 + self.coupling) * bands_ir[0] + self.coupling * bands_vi[0]) / denominator
        low_vi = (self.coupling * bands_ir[0] + (1 + self.coupling) * bands_vi[0]) / denominator
        mixed_ir = haar_inverse(low_ir, *bands_ir[1:], output_size=feature_ir.shape[-2:])
        mixed_vi = haar_inverse(low_vi, *bands_vi[1:], output_size=feature_vi.shape[-2:])
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
    parser.add_argument('--coupling', type=float, default=1.0)
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
    if not model.use_dfd:
        raise ValueError('Closed-form analysis requires the DFD model')
    wrapper = ClosedFormLowFrequencyModel(model, args.coupling).eval()
    dataset = FusionDataset(args.data_root, split=cfg['data']['test_split'], size=cfg['data']['size'])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    results = evaluate(wrapper, loader, device, args.save_dir, args.max_batches, args.output_norm)
    print(results)


if __name__ == '__main__':
    main()
