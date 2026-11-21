import torch
import torch.nn as nn
import torch.nn.functional as F


class TransportAligner(nn.Module):
    def __init__(self, grid_size=32, sinkhorn_iters=30, epsilon=0.1,
                 spatial_weight=1.0, max_offset=4.0):
        super().__init__()
        if grid_size < 2 or sinkhorn_iters < 1 or epsilon <= 0 or spatial_weight < 0:
            raise ValueError('Invalid transport alignment configuration')
        self.grid_size = grid_size
        self.sinkhorn_iters = sinkhorn_iters
        self.epsilon = epsilon
        self.spatial_weight = spatial_weight
        self.max_offset = max_offset

    def _coordinates(self, height, width, device, dtype):
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype), indexing='ij'
        )
        return torch.stack((xx, yy), dim=-1).reshape(-1, 2)

    def _transport_offset(self, source, target, original_width):
        batch, _, height, width = source.shape
        source_flat = F.normalize(source.flatten(2), dim=1, eps=1e-6).transpose(1, 2)
        target_flat = F.normalize(target.flatten(2), dim=1, eps=1e-6).transpose(1, 2)
        feature_cost = 1 - torch.bmm(target_flat, source_flat.transpose(1, 2))
        coords = self._coordinates(height, width, source.device, source.dtype)
        delta = coords[:, None, :] - coords[None, :, :]
        spatial_cost = delta.square().sum(dim=-1)
        cost = feature_cost + self.spatial_weight * spatial_cost
        max_grid_offset = max(1.0, self.max_offset * width / original_width)
        allowed = delta.abs().amax(dim=-1) <= max_grid_offset + 1e-6
        cost = cost.masked_fill(~allowed, 1e4)
        log_kernels = -cost / self.epsilon
        f = torch.zeros(batch, target_flat.shape[1], device=source.device, dtype=source.dtype)
        g = torch.zeros(batch, source_flat.shape[1], device=source.device, dtype=source.dtype)
        log_mass = torch.log(torch.tensor(float(target_flat.shape[1]), device=source.device, dtype=source.dtype))
        for _ in range(self.sinkhorn_iters):
            f = -log_mass - torch.logsumexp(log_kernels + g[:, None, :], dim=2)
            g = -log_mass - torch.logsumexp(log_kernels + f[:, :, None], dim=1)
        log_plan = log_kernels + f[:, :, None] + g[:, None, :]
        plan = torch.exp(log_plan)
        expected_source = torch.bmm(plan, coords[None].expand(batch, -1, -1))
        expected_source = expected_source / plan.sum(dim=2, keepdim=True).clamp_min(1e-8)
        offset = expected_source - coords[None]
        offset = offset.transpose(1, 2).reshape(batch, 2, height, width)
        mass_error = (plan.sum(dim=2) - 1 / target_flat.shape[1]).abs().max()
        return offset, mass_error

    def forward(self, f_ir, f_vi):
        original_size = f_ir.shape[-2:]
        source = F.adaptive_avg_pool2d(f_ir, (self.grid_size, self.grid_size))
        target = F.adaptive_avg_pool2d(f_vi, (self.grid_size, self.grid_size))
        offset, mass_error = self._transport_offset(source, target, original_size[1])
        offset = F.interpolate(offset, size=original_size, mode='bilinear', align_corners=True)
        height, width = original_size
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, height, device=f_ir.device, dtype=f_ir.dtype),
            torch.linspace(-1, 1, width, device=f_ir.device, dtype=f_ir.dtype), indexing='ij'
        )
        grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(f_ir.shape[0], -1, -1, -1)
        scale = offset.new_tensor([2 / max(width - 1, 1), 2 / max(height - 1, 1)]).view(1, 2, 1, 1)
        normalized = offset.permute(0, 2, 3, 1) * scale.permute(0, 2, 3, 1)
        aligned_ir = F.grid_sample(
            f_ir, grid + normalized, mode='bilinear', padding_mode='border', align_corners=True
        )
        return aligned_ir, f_vi, offset, {'transport_mass_error': mass_error.detach()}
