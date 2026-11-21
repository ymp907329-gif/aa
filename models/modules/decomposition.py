import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualRefine(nn.Module):
    def __init__(self, channels, nonexpansive=False):
        super().__init__()
        first = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        second = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        if nonexpansive:
            first = nn.utils.spectral_norm(first)
            second = nn.utils.spectral_norm(second)
        layers = [first]
        if not nonexpansive:
            layers.append(nn.BatchNorm2d(channels))
        layers.extend((nn.LeakyReLU(0.2, inplace=True), second))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class AdaptiveIterativeDecomposition(nn.Module):
    def __init__(self, channels, max_steps=4, min_steps=2, residual_tol=0.05,
                 consistency_tol=0.5, diffusion_step=0.12, kappa=0.25,
                 region_adaptive=False, stop_kernel=5, soft_region=False,
                 soft_temperature=0.05, incremental_detail=False, stop_quantile=None,
                 output_scale=1.0, nonexpansive=False):
        super().__init__()
        if not 0 <= output_scale <= 1:
            raise ValueError('AID output_scale must be in [0, 1]')
        if not 1 <= min_steps <= max_steps:
            raise ValueError('AID requires 1 <= min_steps <= max_steps')
        if stop_kernel < 1 or stop_kernel % 2 == 0:
            raise ValueError('AID stop_kernel must be a positive odd integer')
        if soft_temperature <= 0:
            raise ValueError('AID soft_temperature must be positive')
        if stop_quantile is not None and not 0 < stop_quantile < 1:
            raise ValueError('stop_quantile must be in (0, 1)')
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.residual_tol = residual_tol
        self.consistency_tol = consistency_tol
        self.diffusion_step = diffusion_step
        self.kappa = kappa
        self.region_adaptive = region_adaptive
        self.stop_kernel = stop_kernel
        self.soft_region = soft_region
        self.soft_temperature = soft_temperature
        self.incremental_detail = incremental_detail
        self.stop_quantile = stop_quantile
        self.output_scale = output_scale
        self.nonexpansive = nonexpansive
        self.low_ir = ResidualRefine(channels, nonexpansive)
        self.low_vi = ResidualRefine(channels, nonexpansive)
        self.high_ir = ResidualRefine(channels, nonexpansive)
        self.high_vi = ResidualRefine(channels, nonexpansive)

    def _anisotropic_diffusion(self, x):
        update = torch.zeros_like(x)
        for dim in (-2, -1):
            forward = x.diff(dim=dim)
            forward = F.pad(forward, (0, 1, 0, 0) if dim == -1 else (0, 0, 0, 1))
            backward = -F.pad(
                forward[..., :-1] if dim == -1 else forward[..., :-1, :],
                (1, 0, 0, 0) if dim == -1 else (0, 0, 1, 0),
            )
            for diff in (forward, backward):
                conduction = torch.exp(-(diff / self.kappa).square())
                update = update + conduction * diff
        return update

    @staticmethod
    def _change(new, old):
        dims = (1, 2, 3)
        return (new - old).abs().mean(dim=dims) / (old.abs().mean(dim=dims) + 1e-6)

    @staticmethod
    def _local_change(new, old):
        return (new - old).abs().mean(dim=1, keepdim=True) / (
            old.abs().mean(dim=1, keepdim=True) + 1e-6
        )

    @staticmethod
    def _gradient_energy(x):
        gx = x[..., 1:] - x[..., :-1]
        gy = x[..., 1:, :] - x[..., :-1, :]
        return gx.abs().mean(dim=(1, 2, 3)) + gy.abs().mean(dim=(1, 2, 3))

    @staticmethod
    def _local_gradient_energy(x):
        gx = (x[..., 1:] - x[..., :-1]).abs().mean(dim=1, keepdim=True)
        gy = (x[..., 1:, :] - x[..., :-1, :]).abs().mean(dim=1, keepdim=True)
        return F.pad(gx, (0, 1, 0, 0)) + F.pad(gy, (0, 0, 0, 1))

    def _pool_stop_score(self, x):
        if not self.region_adaptive or self.stop_kernel == 1:
            return x
        return F.avg_pool2d(x, self.stop_kernel, stride=1, padding=self.stop_kernel // 2)

    def forward(self, f_ir, f_vi, return_steps=False):
        iteration_features = []
        low_ir, low_vi = f_ir, f_vi
        high_ir = torch.zeros_like(f_ir)
        high_vi = torch.zeros_like(f_vi)
        batch = f_ir.shape[0]
        state_shape = (batch, 1, *f_ir.shape[-2:]) if self.region_adaptive else (batch,)
        active = torch.ones(state_shape, dtype=torch.bool, device=f_ir.device)
        steps = torch.full(state_shape, self.max_steps, dtype=torch.long, device=f_ir.device)

        update_gate = None
        for iteration in range(1, self.max_steps + 1):
            previous_low_ir, previous_low_vi = low_ir, low_vi
            previous_high_ir, previous_high_vi = high_ir, high_vi
            state_ir = low_ir + high_ir
            state_vi = low_vi + high_vi
            if self.nonexpansive:
                operator_ir = 0.5 * (self.low_ir(state_ir) + self.high_ir(state_ir))
                operator_vi = 0.5 * (self.low_vi(state_vi) + self.high_vi(state_vi))
                next_state_ir = 0.5 * (state_ir + operator_ir)
                next_state_vi = 0.5 * (state_vi + operator_vi)
                next_low_ir, next_low_vi = next_state_ir, next_state_vi
                next_high_ir = torch.zeros_like(next_state_ir)
                next_high_vi = torch.zeros_like(next_state_vi)
            else:
                next_low_ir = low_ir + self.diffusion_step * self._anisotropic_diffusion(low_ir)
                next_low_vi = low_vi + self.diffusion_step * self._anisotropic_diffusion(low_vi)
                next_low_ir = next_low_ir + 0.5 * self.low_ir(next_low_ir)
                next_low_vi = next_low_vi + 0.5 * self.low_vi(next_low_vi)
                base_ir = low_ir if self.incremental_detail else state_ir
                base_vi = low_vi if self.incremental_detail else state_vi
                residual_ir = base_ir - next_low_ir
                residual_vi = base_vi - next_low_vi
                next_high_ir = residual_ir + 0.5 * self.high_ir(residual_ir)
                next_high_vi = residual_vi + 0.5 * self.high_vi(residual_vi)
                if self.incremental_detail:
                    next_high_ir = high_ir + next_high_ir
                    next_high_vi = high_vi + next_high_vi

            mask = active if self.region_adaptive else active.view(-1, 1, 1, 1)
            low_ir = torch.where(mask, next_low_ir, low_ir)
            low_vi = torch.where(mask, next_low_vi, low_vi)
            high_ir = torch.where(mask, next_high_ir, high_ir)
            high_vi = torch.where(mask, next_high_vi, high_vi)

            if self.region_adaptive:
                change_ir = self._local_change(low_ir + high_ir, state_ir)
                change_vi = self._local_change(low_vi + high_vi, state_vi)
                consistency = 0.5 * (
                    F.cosine_similarity(low_ir + high_ir, low_vi + high_vi, dim=1).unsqueeze(1) + 1
                )
                complexity = 0.5 * (
                    self._local_gradient_energy(low_ir + high_ir)
                    + self._local_gradient_energy(low_vi + high_vi)
                )
            else:
                change_ir = self._change(low_ir + high_ir, state_ir)
                change_vi = self._change(low_vi + high_vi, state_vi)
                consistency = 0.5 * (
                    F.cosine_similarity(low_ir + high_ir, low_vi + high_vi, dim=1) + 1
                ).mean(dim=(1, 2))
                complexity = 0.5 * (
                    self._gradient_energy(low_ir + high_ir)
                    + self._gradient_energy(low_vi + high_vi)
                )
            if self.nonexpansive:
                score_ir, score_vi = change_ir, change_vi
            else:
                score_ir = change_ir * (1 + complexity) / (consistency + 0.1)
                score_vi = change_vi * (1 + complexity) / (consistency + 0.1)
            if self.soft_region:
                score_map = 0.5 * (
                    self._pool_stop_score(score_ir) + self._pool_stop_score(score_vi)
                )
                update_gate = torch.sigmoid(
                    (score_map - self.residual_tol) / self.soft_temperature
                ).detach()
                low_ir = previous_low_ir + update_gate * (low_ir - previous_low_ir)
                low_vi = previous_low_vi + update_gate * (low_vi - previous_low_vi)
                high_ir = previous_high_ir + update_gate * (high_ir - previous_high_ir)
                high_vi = previous_high_vi + update_gate * (high_vi - previous_high_vi)
            pooled_consistency = self._pool_stop_score(consistency)
            if self.stop_quantile is not None:
                if not self.region_adaptive:
                    raise ValueError('stop_quantile requires region_adaptive=True')
                joint_score = 0.5 * (
                    self._pool_stop_score(score_ir) + self._pool_stop_score(score_vi)
                )
                threshold = torch.quantile(
                    joint_score.flatten(1), self.stop_quantile, dim=1
                ).view(-1, 1, 1, 1)
                converged = (
                    (joint_score <= threshold)
                    & (pooled_consistency >= self.consistency_tol)
                )
            else:
                converged = (
                    (self._pool_stop_score(score_ir) < self.residual_tol)
                    & (self._pool_stop_score(score_vi) < self.residual_tol)
                    & (pooled_consistency >= self.consistency_tol)
                )
            stopped = active & converged & (iteration >= self.min_steps) & (not self.soft_region)
            steps = torch.where(stopped, torch.full_like(steps, iteration), steps)
            active = active & ~stopped
            if return_steps:
                iteration_features.append((
                    f_ir + self.output_scale * (low_ir + high_ir - f_ir),
                    f_vi + self.output_scale * (low_vi + high_vi - f_vi),
                ))
            if not active.any():
                break

        aux = {
            'steps': steps.float().mean(dim=(1, 2, 3)).detach()
            if self.region_adaptive else steps.detach(),
            'steps_map': steps.detach(),
            'mean_steps': steps.float().mean().detach(),
            'consistency': consistency.mean().detach(),
            'stop_score': (0.5 * (score_ir + score_vi)).mean().detach(),
            'low_ir': low_ir,
            'low_vi': low_vi,
            'high_ir': high_ir,
            'high_vi': high_vi,
        }
        if update_gate is not None:
            aux['update_gate_mean'] = update_gate.mean().detach()
        if return_steps:
            aux['iteration_features'] = iteration_features
        out_ir = f_ir + self.output_scale * (low_ir + high_ir - f_ir)
        out_vi = f_vi + self.output_scale * (low_vi + high_vi - f_vi)
        return out_ir, out_vi, aux
