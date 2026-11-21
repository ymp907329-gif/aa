import torch
import torch.nn as nn
import torch.nn.functional as F


class IlluminationSaliencyWeight(nn.Module):
    def __init__(self, hidden_ch=16):
        super().__init__()
        self.estimator = nn.Sequential(
            nn.Conv2d(4, hidden_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_ch, 1, 1),
        )

    def forward(self, ir, vi):
        illumination = F.avg_pool2d(vi, kernel_size=15, stride=1, padding=7)
        local_ir = F.avg_pool2d(ir, kernel_size=15, stride=1, padding=7)
        saliency = F.relu(ir - local_ir)
        prior = torch.sigmoid(4 * saliency + 2 * (0.35 - illumination))
        learned = torch.sigmoid(self.estimator(torch.cat((ir, vi, illumination, saliency), dim=1)))
        weight = 0.5 * prior + 0.5 * learned
        return weight, prior
