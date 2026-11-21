import torch.nn as nn
from .modules.global_context import PooledGlobalContext
from .modules.inception import Inception_ResidualBlock


class Encoder(nn.Module):
    """双流编码器（单流形式，对应一路输入）"""

    def __init__(self, in_ch=1, base_ch=64, use_global=False, global_pool=8, global_heads=4):
        super().__init__()
        self.stage0 = Inception_ResidualBlock(in_ch, base_ch // 2)
        self.stage1 = Inception_ResidualBlock(base_ch // 2, base_ch)
        self.stage2 = Inception_ResidualBlock(base_ch, base_ch)
        self.global_context = (
            PooledGlobalContext(base_ch, global_pool, global_heads) if use_global else None
        )

    def forward(self, x):
        f0 = self.stage0(x)   # (B, base_ch//2, H, W)
        f1 = self.stage1(f0)  # (B, base_ch, H, W)
        f2 = self.stage2(f1)  # (B, base_ch, H, W)
        if self.global_context is not None:
            f2 = self.global_context(f2)
        return f0, f1, f2