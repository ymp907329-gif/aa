import torch.nn as nn


class ConvBnLeakyRelu2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, negative_slope=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvBnRelu2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)