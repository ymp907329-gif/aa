import torch
import torch.nn as nn
import torch.nn.functional as F


class PooledGlobalContext(nn.Module):
    def __init__(self, channels, pool_size=8, heads=4):
        super().__init__()
        if channels % heads:
            raise ValueError('channels must be divisible by heads')
        if pool_size < 1:
            raise ValueError('pool_size must be positive')
        self.pool_size = pool_size
        self.heads = heads
        self.norm = nn.GroupNorm(1, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.qkv_dw = nn.Conv2d(
            channels * 3, channels * 3, 3, padding=1, groups=channels * 3, bias=False
        )
        self.project = nn.Conv2d(channels, channels, 1, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        pooled = F.avg_pool2d(self.norm(x), kernel_size=self.pool_size)
        batch, channels, height, width = pooled.shape
        qkv = self.qkv_dw(self.qkv(pooled))
        q, k, v = qkv.chunk(3, dim=1)
        head_dim = channels // self.heads
        q = q.reshape(batch, self.heads, head_dim, height * width)
        k = k.reshape(batch, self.heads, head_dim, height * width)
        v = v.reshape(batch, self.heads, head_dim, height * width)
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        attention = (q.transpose(-2, -1) @ k).softmax(dim=-1)
        context = (attention @ v.transpose(-2, -1)).transpose(-2, -1)
        context = context.reshape(batch, channels, height, width)
        context = F.interpolate(self.project(context), size=x.shape[-2:], mode='bilinear', align_corners=False)
        return x + self.gate * context
