import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """CBAM 风格的通道注意力"""

    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """CBAM 风格的空间注意力"""

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class AFF(nn.Module):
    """Attentional Feature Fusion：自适应加权融合两个特征图"""

    def __init__(self, channels, r=4):
        super().__init__()
        inter = channels // r
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter, 1, bias=False),
            nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter, 1, bias=False),
            nn.BatchNorm2d(inter),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        xa = x + y
        wei = self.sigmoid(self.local_att(xa) + self.global_att(xa))
        return x * wei + y * (1 - wei)


class CrossAttention(nn.Module):
    """双模态交叉注意力：为每个模态生成权重图"""

    def __init__(self, channel):
        super().__init__()
        self.conv_x = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1, bias=False),
            nn.Sigmoid(),
        )
        self.conv_y = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)
        return self.conv_x(out), self.conv_y(out)