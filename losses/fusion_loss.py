import torch
import torch.nn as nn
import torch.nn.functional as F

from .freq_contrast import FrequencyContrastiveLoss


def sobel_gradient(x):
    kernel_x = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8
    kernel_y = kernel_x.transpose(2, 3)
    channels = x.shape[1]
    kernel_x = kernel_x.expand(channels, 1, 3, 3)
    kernel_y = kernel_y.expand(channels, 1, 3, 3)
    gx = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), kernel_x, groups=channels)
    gy = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), kernel_y, groups=channels)
    return gx, gy


def teacher_gradient_distillation(fused, teacher, ir, vi, max_ratio=1.5, min_cosine=0.0):
    if max_ratio <= 0 or not -1 <= min_cosine <= 1:
        raise ValueError('Invalid teacher gradient reliability bounds')
    fused_gx, fused_gy = sobel_gradient(fused)
    teacher_gx, teacher_gy = sobel_gradient(teacher)
    ir_gx, ir_gy = sobel_gradient(ir)
    vi_gx, vi_gy = sobel_gradient(vi)
    teacher_magnitude = torch.sqrt(teacher_gx.square() + teacher_gy.square() + 1e-12)
    ir_magnitude = torch.sqrt(ir_gx.square() + ir_gy.square() + 1e-12)
    vi_magnitude = torch.sqrt(vi_gx.square() + vi_gy.square() + 1e-12)
    source_magnitude = torch.maximum(ir_magnitude, vi_magnitude)
    ir_cosine = (teacher_gx * ir_gx + teacher_gy * ir_gy) / (
        teacher_magnitude * ir_magnitude + 1e-6
    )
    vi_cosine = (teacher_gx * vi_gx + teacher_gy * vi_gy) / (
        teacher_magnitude * vi_magnitude + 1e-6
    )
    reliable = (
        (teacher_magnitude > 1e-5)
        & (torch.maximum(ir_cosine, vi_cosine) >= min_cosine)
        & (teacher_magnitude <= max_ratio * source_magnitude + 1e-6)
    ).detach()
    weights = (teacher_magnitude * reliable).detach()
    error = 0.5 * (
        (fused_gx - teacher_gx).abs() + (fused_gy - teacher_gy).abs()
    )
    return (error * weights).sum() / (weights.sum() + 1e-6)


def ssim(x, y, window_size=11, sigma=1.5):
    channels = x.shape[1]
    coords = torch.arange(window_size, device=x.device, dtype=x.dtype) - window_size // 2
    kernel_1d = torch.exp(-coords.square() / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = kernel_1d[:, None] @ kernel_1d[None, :]
    kernel = kernel.expand(channels, 1, window_size, window_size)
    padding = window_size // 2
    mu_x = F.conv2d(x, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=padding, groups=channels)
    sigma_x = F.conv2d(x * x, kernel, padding=padding, groups=channels) - mu_x.square()
    sigma_y = F.conv2d(y * y, kernel, padding=padding, groups=channels) - mu_y.square()
    sigma_xy = F.conv2d(x * y, kernel, padding=padding, groups=channels) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    value = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    )
    return value.mean()


def edge_preservation_loss(fused, ir, vi, overshoot_penalty=1.0):
    if not 0 <= overshoot_penalty <= 1:
        raise ValueError('overshoot_penalty must be in [0, 1]')
    fused_gx, fused_gy = sobel_gradient(fused)
    fused_strength = torch.sqrt(fused_gx.square() + fused_gy.square() + 1e-8)
    fused_angle = torch.atan2(fused_gy, fused_gx)
    losses = []
    for source in (ir, vi):
        source_gx, source_gy = sobel_gradient(source)
        source_strength = torch.sqrt(source_gx.square() + source_gy.square() + 1e-8)
        source_angle = torch.atan2(source_gy, source_gx)
        symmetric_ratio = torch.minimum(source_strength, fused_strength) / (
            torch.maximum(source_strength, fused_strength) + 1e-8
        )
        preservation_ratio = torch.clamp(fused_strength / (source_strength + 1e-8), max=1.0)
        strength_ratio = (
            overshoot_penalty * symmetric_ratio
            + (1 - overshoot_penalty) * preservation_ratio
        )
        angle_delta = (source_angle - fused_angle).abs() % torch.pi
        angle_delta = torch.minimum(angle_delta, torch.pi - angle_delta)
        angle_ratio = 1 - angle_delta / (torch.pi / 2)
        strength_score = 0.9994 * torch.sigmoid(15 * (strength_ratio - 0.5))
        angle_score = 0.9879 * torch.sigmoid(22 * (angle_ratio - 0.8))
        losses.append((strength_score * angle_score * source_strength).sum() / (source_strength.sum() + 1e-8))
    return 1 - 0.5 * (losses[0] + losses[1])


def spatial_smoothness(x):
    horizontal = (x[..., 1:] - x[..., :-1]).abs().mean()
    vertical = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    return horizontal + vertical


class FusionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pixel_weight = config.get('lambda_pixel', 1.0)
        self.gradient_weight = config.get('lambda_gradient', 1.0)
        self.ssim_weight = config.get('lambda_ssim', 1.0)
        self.freqcl_weight = config.get('lambda_freqcl', 0.1)
        self.offset_weight = config.get('lambda_offset', 0.0)
        self.weight_prior_weight = config.get('lambda_weight_prior', 0.0)
        self.weight_smooth_weight = config.get('lambda_weight_smooth', 0.0)
        self.orientation_weight = config.get('lambda_orientation', 0.0)
        self.edge_preservation_weight = config.get('lambda_edge_preservation', 0.0)
        self.edge_overshoot_penalty = config.get('edge_overshoot_penalty', 1.0)
        if not 0 <= self.edge_overshoot_penalty <= 1:
            raise ValueError('edge_overshoot_penalty must be in [0, 1]')
        self.source_reconstruction_weight = config.get('lambda_source_reconstruction', 0.0)
        self.intensity_mode = config.get('intensity_mode', 'average')
        self.gradient_mode = config.get('gradient_mode', 'componentwise')
        if self.intensity_mode not in ('average', 'max', 'adaptive'):
            raise ValueError("intensity_mode must be 'average', 'max', or 'adaptive'")
        if self.gradient_mode not in ('componentwise', 'vector'):
            raise ValueError("gradient_mode must be 'componentwise' or 'vector'")
        if self.orientation_weight > 0 and self.gradient_mode != 'vector':
            raise ValueError('Orientation loss requires vector gradient mode')
        self.freqcl = FrequencyContrastiveLoss(
            temperature=config.get('temperature', 0.2),
            levels=config.get('wavelet_levels', 2),
        )

    def forward(self, fused, ir, vi, aux=None, return_task_losses=False):
        weight_prior = fused.new_zeros(())
        weight_smooth = fused.new_zeros(())
        ir_weight = fused.new_zeros(())
        if self.intensity_mode == 'max':
            target = torch.maximum(ir, vi)
        elif self.intensity_mode == 'adaptive':
            if not aux or 'intensity_weight' not in aux:
                raise ValueError('Adaptive intensity loss requires model intensity weights')
            weight = aux['intensity_weight']
            prior = aux['intensity_prior']
            target = (weight * ir + (1 - weight) * vi).detach()
            weight_prior = F.mse_loss(weight, prior)
            weight_smooth = spatial_smoothness(weight)
            ir_weight = weight.mean().detach()
        else:
            target = None
        pixel = F.l1_loss(fused, target) if target is not None else 0.5 * (
            F.l1_loss(fused, ir) + F.l1_loss(fused, vi)
        )
        fused_gx, fused_gy = sobel_gradient(fused)
        ir_gx, ir_gy = sobel_gradient(ir)
        vi_gx, vi_gy = sobel_gradient(vi)
        orientation = fused.new_zeros(())
        if self.gradient_mode == 'vector':
            ir_magnitude = torch.sqrt(ir_gx.square() + ir_gy.square() + 1e-12)
            vi_magnitude = torch.sqrt(vi_gx.square() + vi_gy.square() + 1e-12)
            select_ir = ir_magnitude >= vi_magnitude
            target_gx = torch.where(select_ir, ir_gx, vi_gx)
            target_gy = torch.where(select_ir, ir_gy, vi_gy)
            if self.orientation_weight > 0:
                target_magnitude = torch.sqrt(target_gx.square() + target_gy.square() + 1e-12)
                fused_magnitude = torch.sqrt(fused_gx.square() + fused_gy.square() + 1e-12)
                cosine = (fused_gx * target_gx + fused_gy * target_gy) / (
                    fused_magnitude * target_magnitude + 1e-6
                )
                weights = target_magnitude.detach()
                orientation = ((1 - cosine) * weights).sum() / (weights.sum() + 1e-6)
        else:
            target_gx = torch.where(ir_gx.abs() >= vi_gx.abs(), ir_gx, vi_gx)
            target_gy = torch.where(ir_gy.abs() >= vi_gy.abs(), ir_gy, vi_gy)
        gradient = 0.5 * (
            F.l1_loss(fused_gx, target_gx) + F.l1_loss(fused_gy, target_gy)
        )
        structural = 1 - 0.5 * (ssim(fused, ir) + ssim(fused, vi))
        frequency = self.freqcl(fused, ir, vi) if self.freqcl_weight > 0 else fused.new_zeros(())
        offset_loss = fused.new_zeros(())
        if self.offset_weight > 0 and aux and 'dfd' in aux:
            offset = aux['dfd']['offset']
            offset_loss = offset.abs().mean() + spatial_smoothness(offset)
        source_reconstruction = fused.new_zeros(())
        if self.source_reconstruction_weight > 0:
            if not aux or 'source_ir' not in aux or 'source_vi' not in aux:
                raise ValueError('Source reconstruction loss requires source_ir and source_vi')
            source_ir, source_vi = aux['source_ir'], aux['source_vi']
            source_pixel = 0.5 * (
                F.l1_loss(source_ir, ir) + F.l1_loss(source_vi, vi)
            )
            source_structural = 1 - 0.5 * (
                ssim(source_ir, ir) + ssim(source_vi, vi)
            )
            source_reconstruction = source_pixel + source_structural
        edge_preservation = edge_preservation_loss(
            fused, ir, vi, self.edge_overshoot_penalty
        ) if self.edge_preservation_weight > 0 else fused.new_zeros(())
        total = (
            self.pixel_weight * pixel
            + self.gradient_weight * gradient
            + self.ssim_weight * structural
            + self.freqcl_weight * frequency
            + self.offset_weight * offset_loss
            + self.weight_prior_weight * weight_prior
            + self.weight_smooth_weight * weight_smooth
            + self.orientation_weight * orientation
            + self.edge_preservation_weight * edge_preservation
            + self.source_reconstruction_weight * source_reconstruction
        )
        components = {
            'pixel': pixel.detach(),
            'gradient': gradient.detach(),
            'orientation': orientation.detach(),
            'edge_preservation': edge_preservation.detach(),
            'ssim': structural.detach(),
            'freqcl': frequency.detach(),
            'offset': offset_loss.detach(),
            'weight_prior': weight_prior.detach(),
            'weight_smooth': weight_smooth.detach(),
            'source_reconstruction': source_reconstruction.detach(),
            'ir_weight': ir_weight,
        }
        if return_task_losses:
            task_losses = {
                'pixel': self.pixel_weight * pixel,
                'gradient': self.gradient_weight * gradient,
                'ssim': self.ssim_weight * structural,
                'freqcl': self.freqcl_weight * frequency,
                'offset': self.offset_weight * offset_loss,
                'weight_prior': self.weight_prior_weight * weight_prior,
                'weight_smooth': self.weight_smooth_weight * weight_smooth,
                'orientation': self.orientation_weight * orientation,
                'edge_preservation': self.edge_preservation_weight * edge_preservation,
                'source_reconstruction': self.source_reconstruction_weight * source_reconstruction,
            }
            return total, components, task_losses
        return total, components
