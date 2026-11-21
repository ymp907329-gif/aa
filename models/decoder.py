import torch
import torch.nn as nn

from .modules.conv import ConvBnLeakyRelu2d


class Decoder(nn.Module):
    def __init__(self, base_ch=64, use_skip=False, output_anchor=False, residual_scale=0.1):
        super().__init__()
        if residual_scale <= 0:
            raise ValueError('residual_scale must be positive')
        self.use_skip = use_skip
        self.output_anchor = output_anchor
        self.residual_scale = residual_scale
        self.conv1 = ConvBnLeakyRelu2d(base_ch, base_ch)
        self.conv2 = ConvBnLeakyRelu2d(base_ch, base_ch // 2)
        if use_skip:
            self.skip_ir = ConvBnLeakyRelu2d(base_ch // 2, base_ch // 2)
            self.skip_vi = ConvBnLeakyRelu2d(base_ch // 2, base_ch // 2)
            self.skip_weight = nn.Conv2d(base_ch, 2, kernel_size=1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_ch // 2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )
        if output_anchor:
            nn.init.zeros_(self.conv3[0].weight)

    def forward(self, x, skip_ir=None, skip_vi=None, anchor=None):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_skip:
            if skip_ir is None or skip_vi is None:
                raise ValueError('Decoder skip features are required when use_skip=True')
            weights = torch.softmax(self.skip_weight(torch.cat((skip_ir, skip_vi), dim=1)), dim=1)
            skip = weights[:, 0:1] * self.skip_ir(skip_ir) + weights[:, 1:2] * self.skip_vi(skip_vi)
            x = x + skip
        out = self.conv3(x)
        if not self.output_anchor:
            return out
        if anchor is None:
            raise ValueError('An output anchor is required when output_anchor=True')
        return (anchor + self.residual_scale * (out - 0.5)).clamp(0, 1)
