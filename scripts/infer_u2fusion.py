import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import FusionDataset
from utils.metrics import compute_fusion_metrics


class U2FusionGenerator(nn.Module):
    def __init__(self, weights_path):
        super().__init__()
        encoder_specs = (
            ('conv1_1', 2), ('dense_block_conv1', 44), ('dense_block_conv2', 88),
            ('dense_block_conv3', 132), ('dense_block_conv4', 176), ('dense_block_conv5', 220),
        )
        self.encoder_names = []
        for name, channels in encoder_specs:
            layer = nn.Conv2d(channels, 44, 3)
            setattr(self, name, layer)
            self.encoder_names.append(name)
        decoder_specs = (('conv2_1', 264, 128), ('conv2_2', 128, 64), ('conv2_3', 64, 32), ('conv2_4', 32, 1))
        self.decoder_names = []
        for name, in_channels, out_channels in decoder_specs:
            setattr(self, name, nn.Conv2d(in_channels, out_channels, 3))
            self.decoder_names.append(name)
        weights = np.load(weights_path)
        for name in self.encoder_names:
            self._load_layer('Generator/encoder', name, weights)
        for name in self.decoder_names:
            self._load_layer('Generator_1/decoder', name, weights)

    def _load_layer(self, scope, name, weights):
        layer = getattr(self, name)
        kernel = weights[f'{scope}/{name}/kernel'].transpose(3, 2, 0, 1)
        bias = weights[f'{scope}/{name}/bias']
        with torch.no_grad():
            layer.weight.copy_(torch.from_numpy(kernel.copy()))
            layer.bias.copy_(torch.from_numpy(bias.copy()))

    @staticmethod
    def _conv(layer, x):
        return layer(F.pad(x, (1, 1, 1, 1), mode='reflect'))

    def forward(self, vis, ir):
        x = torch.cat((vis, ir), dim=1)
        for index, name in enumerate(self.encoder_names):
            out = F.leaky_relu(self._conv(getattr(self, name), x), negative_slope=0.2)
            x = torch.cat((out, x), dim=1) if index else out
        for name in self.decoder_names[:-1]:
            x = F.leaky_relu(self._conv(getattr(self, name), x), negative_slope=0.2)
        x = self._conv(getattr(self, self.decoder_names[-1]), x)
        return torch.tanh(x) / 2 + 0.5


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='external/U2Fusion/model/u2fusion_weights.npz')
    parser.add_argument('--data_root', default='data/MSRS')
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_batches', type=int)
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model = U2FusionGenerator(args.weights).eval().to(device)
    dataset = FusionDataset(args.data_root, split='test', size=256)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    os.makedirs(args.save_dir, exist_ok=True)
    metric_keys = ('psnr', 'ssim', 'vif', 'entropy', 'sd', 'ag', 'sf', 'mi', 'qabf')
    rows = []
    for index, batch in enumerate(tqdm(loader, desc='U2Fusion')):
        if args.max_batches is not None and index >= args.max_batches:
            break
        vis = batch['vi'].to(device)
        ir = batch['ir'].to(device)
        fused = model(vis, ir).clamp(0, 1)
        fused_image = fused[0].cpu().numpy()
        values = compute_fusion_metrics(fused_image, batch['ir'][0].numpy(), batch['vi'][0].numpy())
        rows.append({'name': batch['name'][0], **values})
        image = (fused_image.squeeze(0) * 255).round().astype(np.uint8)
        output_name = os.path.splitext(batch['name'][0])[0] + '.png'
        Image.fromarray(image).save(os.path.join(args.save_dir, output_name))
        float_dir = Path(args.save_dir) / 'float'
        float_dir.mkdir(exist_ok=True)
        np.save(float_dir / (Path(output_name).stem + '.npy'), np.squeeze(fused_image).astype(np.float32))
    if not rows:
        raise ValueError('No images were evaluated')
    results = {key: float(np.mean([row[key] for row in rows])) for key in metric_keys}
    with open(os.path.join(args.save_dir, 'metrics.json'), 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    with open(os.path.join(args.save_dir, 'metrics.csv'), 'w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=('name', *metric_keys))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
