import torch
import torch.nn as nn
import torch.nn.functional as F

from .frequency import haar_forward, haar_inverse


def sobel(x):
    kernel_x = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8
    kernel_y = kernel_x.transpose(2, 3)
    channels = x.shape[1]
    kernel_x = kernel_x.expand(channels, 1, 3, 3)
    kernel_y = kernel_y.expand(channels, 1, 3, 3)
    gx = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), kernel_x, groups=channels)
    gy = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), kernel_y, groups=channels)
    return gx, gy


class AdaptiveSourceDetailCompensation(nn.Module):
    def __init__(self, hidden_channels=16, beta_max=1.0, noise_kernel=5, output_scale=1.0,
                 directional=False):
        super().__init__()
        if beta_max <= 0:
            raise ValueError('beta_max must be positive')
        if output_scale <= 0:
            raise ValueError('output_scale must be positive')
        if noise_kernel < 1 or noise_kernel % 2 == 0:
            raise ValueError('noise_kernel must be a positive odd integer')
        self.beta_max = beta_max
        self.output_scale = output_scale
        self.noise_kernel = noise_kernel
        self.directional = directional
        self.head = nn.Sequential(
            nn.Conv2d(5, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 3 if directional else 1, kernel_size=1),
        )

    def noise_estimate(self, x):
        local_mean = F.avg_pool2d(x, self.noise_kernel, stride=1, padding=self.noise_kernel // 2)
        residual = x - local_mean
        local_energy = F.avg_pool2d(residual.square(), self.noise_kernel, stride=1,
                                    padding=self.noise_kernel // 2)
        return torch.sqrt(local_energy + 1e-8)

    def forward(self, fused, ir, vi):
        target = torch.maximum(ir, vi)
        descriptor_input = fused.detach()
        noise_ir = self.noise_estimate(ir)
        noise_vi = self.noise_estimate(vi)
        fused_gx, fused_gy = sobel(descriptor_input)
        target_gx, target_gy = sobel(target)
        numerator = fused_gx * target_gx + fused_gy * target_gy
        denominator = torch.sqrt((fused_gx.square() + fused_gy.square()) *
                                 (target_gx.square() + target_gy.square()) + 1e-8)
        consistency = numerator / (denominator + 1e-6)
        saturation = (torch.maximum(torch.maximum(ir, vi), descriptor_input) >= 0.98).to(fused.dtype)
        features = torch.cat((noise_ir, noise_vi, consistency, saturation,
                              descriptor_input), dim=1)
        beta = self.beta_max * self.output_scale * torch.sigmoid(self.head(features))
        fused_bands = haar_forward(fused)
        target_bands = haar_forward(target)
        beta_band = F.interpolate(beta, size=fused_bands[1].shape[-2:], mode='area')
        if self.directional:
            high = tuple(band + beta_band[:, index:index + 1] * reference
                         for index, (band, reference) in enumerate(zip(fused_bands[1:], target_bands[1:])))
        else:
            high = tuple(band + beta_band * reference
                         for band, reference in zip(fused_bands[1:], target_bands[1:]))
        output = haar_inverse(fused_bands[0], *high, output_size=fused.shape[-2:])
        aux = {
            'compensation_beta': beta,
            'compensation_noise_ir': noise_ir,
            'compensation_noise_vi': noise_vi,
            'compensation_consistency': consistency,
            'compensation_saturation': saturation,
            'uncompensated_output': fused,
        }
        return output, aux
