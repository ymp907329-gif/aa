import torch
import torch.nn as nn
from .attention import ChannelAttention, AFF, CrossAttention


class FusionLayer(nn.Module):
    """
    三分支融合层：
      1) 公共分支：通道注意力加权后相加
      2) 特有分支：AFF 自适应融合
      3) 交叉分支：交叉注意力加权
    最后三路拼接，1x1 卷积压缩回 base_ch。
    """

    def __init__(self, ch):
        super().__init__()
        # 公共分支
        self.att_B = ChannelAttention(ch)
        self.conv_B = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 特有分支
        self.aff_D = AFF(ch)
        self.conv_D = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 交叉分支
        self.cross = CrossAttention(ch)
        self.conv_cross = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 输出融合
        self.fuse = nn.Sequential(
            nn.Conv2d(ch * 3, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, f_ir, f_vi):
        # 1) 公共特征：两模态的共享信息，用通道注意力后相加
        f_B = f_ir * self.att_B(f_ir) + f_vi * self.att_B(f_vi)
        f_B = self.conv_B(f_B)

        # 2) 特有特征：用 AFF 融合两模态的特有信息
        f_D = self.aff_D(f_ir, f_vi)
        f_D = self.conv_D(f_D)

        # 3) 交叉特征：互相加权
        att_ir, att_vi = self.cross(f_ir, f_vi)
        f_cross = f_ir * att_ir + f_vi * att_vi
        f_cross = self.conv_cross(f_cross)

        # 4) 三路拼接融合
        out = self.fuse(torch.cat([f_B, f_D, f_cross], dim=1))
        return out