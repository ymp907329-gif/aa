import torch
import torch.nn as nn


class Inception(nn.Module):
    """多分支 Inception：1x1、3x3、5x5 与恒等分支，通道拼接后压缩"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.compress = nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        out = torch.cat([x, b1, b2, b3], dim=1)
        return self.compress(out)


class Inception_ResidualBlock(nn.Module):
    """两个 Inception 串联 + 残差连接"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.inception1 = Inception(in_ch, out_ch)
        self.inception2 = Inception(out_ch, out_ch)
        self.relu = nn.ReLU(inplace=True)

        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.inception1(x)
        out = self.inception2(out)
        out = out + identity
        return self.relu(out)