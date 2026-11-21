import torch
import torch.nn as nn
import torch.nn.functional as F

from .transport import TransportAligner


def haar_forward(x):
    pad_h = x.shape[-2] % 2
    pad_w = x.shape[-1] % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]
    return (
        (a + b + c + d) * 0.5,
        (a - b + c - d) * 0.5,
        (a + b - c - d) * 0.5,
        (a - b - c + d) * 0.5,
    )


def haar_inverse(ll, lh, hl, hh, output_size=None):
    a = (ll + lh + hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    out = ll.new_zeros(ll.shape[0], ll.shape[1], ll.shape[2] * 2, ll.shape[3] * 2)
    out[..., 0::2, 0::2] = a
    out[..., 0::2, 1::2] = b
    out[..., 1::2, 0::2] = c
    out[..., 1::2, 1::2] = d
    if output_size is not None:
        out = out[..., :output_size[0], :output_size[1]]
    return out


def multilevel_haar(x, levels=2):
    bands = {}
    low = x
    for level in range(1, levels + 1):
        ll, lh, hl, hh = haar_forward(low)
        bands[f'L{level}_LL'] = ll
        bands[f'L{level}_LH'] = lh
        bands[f'L{level}_HL'] = hl
        bands[f'L{level}_HH'] = hh
        low = ll
    return bands


class LearnableWavelet2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.predict_h = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.update_h = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.predict_v = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.update_v = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self._initialize(self.predict_h, 1.0)
        self._initialize(self.predict_v, 1.0)
        self._initialize(self.update_h, 0.5)
        self._initialize(self.update_v, 0.5)

    @staticmethod
    def _initialize(conv, center_value):
        with torch.no_grad():
            conv.weight.zero_()
            channels = torch.arange(conv.in_channels, device=conv.weight.device)
            conv.weight[channels, channels, 1, 1] = center_value

    def forward(self, x):
        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        a = x[..., 0::2, 0::2]
        b = x[..., 0::2, 1::2]
        c = x[..., 1::2, 0::2]
        d = x[..., 1::2, 1::2]
        b = b - self.predict_h(a)
        d = d - self.predict_h(c)
        a = a + self.update_h(b)
        c = c + self.update_h(d)
        c = c - self.predict_v(a)
        d = d - self.predict_v(b)
        a = a + self.update_v(c)
        b = b + self.update_v(d)
        return a, b, c, d

    def inverse(self, ll, lh, hl, hh, output_size=None):
        a, b, c, d = ll, lh, hl, hh
        b = b - self.update_v(d)
        a = a - self.update_v(c)
        d = d + self.predict_v(b)
        c = c + self.predict_v(a)
        c = c - self.update_h(d)
        a = a - self.update_h(b)
        d = d + self.predict_h(c)
        b = b + self.predict_h(a)
        out = ll.new_zeros(ll.shape[0], ll.shape[1], ll.shape[2] * 2, ll.shape[3] * 2)
        out[..., 0::2, 0::2] = a
        out[..., 0::2, 1::2] = b
        out[..., 1::2, 0::2] = c
        out[..., 1::2, 1::2] = d
        if output_size is not None:
            out = out[..., :output_size[0], :output_size[1]]
        return out


class DeformableFrequencyDecoupling(nn.Module):
    def __init__(self, channels, max_offset=4.0, wavelet='haar',
                 high_fusion_mode='learned', energy_kernel=3, band_mask=None,
                 high_energy_mix=0.5, alignment_mode='deformable', transport_grid=32,
                 sinkhorn_iters=30, transport_epsilon=0.1, transport_spatial_weight=0.1):
        super().__init__()
        if alignment_mode not in ('deformable', 'transport'):
            raise ValueError("alignment_mode must be 'deformable' or 'transport'")
        if wavelet not in ('haar', 'lifting'):
            raise ValueError("DFD wavelet must be 'haar' or 'lifting'")
        if high_fusion_mode not in ('learned', 'energy', 'mix'):
            raise ValueError("DFD high_fusion_mode must be 'learned', 'energy', or 'mix'")
        if energy_kernel < 1 or energy_kernel % 2 == 0:
            raise ValueError('DFD energy_kernel must be a positive odd integer')
        if not 0 <= high_energy_mix <= 1:
            raise ValueError('high_energy_mix must be in [0, 1]')
        if band_mask is not None and (len(band_mask) != 4 or any(bit not in (0, 1) for bit in band_mask)):
            raise ValueError('band_mask must contain four 0/1 values')
        self.channels = channels
        self.max_offset = max_offset
        self.wavelet = wavelet
        self.high_fusion_mode = high_fusion_mode
        self.energy_kernel = energy_kernel
        self.high_energy_mix = high_energy_mix
        self.band_mask = tuple(band_mask) if band_mask is not None else None
        self.lifting = LearnableWavelet2d(channels) if wavelet == 'lifting' else None
        self.alignment_mode = alignment_mode
        self.transport = TransportAligner(
            grid_size=transport_grid, sinkhorn_iters=sinkhorn_iters,
            epsilon=transport_epsilon, spatial_weight=transport_spatial_weight,
            max_offset=max_offset,
        ) if alignment_mode == 'transport' else None
        self.offset = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, 2, 3, padding=1),
        )
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)
        self.low_fusion = self._band_refiner(channels)
        self.high_fusion = self._band_refiner(channels)
        self.low_weight = nn.Conv2d(channels * 2, 2, 1)
        self.high_weight = nn.Conv2d(channels * 2, 2, 1)
        self.output = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    @staticmethod
    def _band_refiner(channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def _base_grid(self, x):
        h, w = x.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype),
            indexing='ij',
        )
        grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        return grid

    def _warp(self, x, offset, direction):
        h, w = x.shape[-2:]
        scale = offset.new_tensor([2.0 / max(w - 1, 1), 2.0 / max(h - 1, 1)]).view(1, 2, 1, 1)
        normalized = offset.permute(0, 2, 3, 1) * scale.permute(0, 2, 3, 1)
        return F.grid_sample(
            x, self._base_grid(x) + direction * 0.5 * normalized,
            mode='bilinear', padding_mode='border', align_corners=True,
        )

    def _align(self, f_ir, f_vi):
        if self.alignment_mode == 'transport':
            aligned_ir, aligned_vi, offset, aux = self.transport(f_ir, f_vi)
            self.transport_aux = aux
            return aligned_ir, aligned_vi, offset
        offset = self.max_offset * torch.tanh(self.offset(torch.cat((f_ir, f_vi), dim=1)))
        aligned_ir = self._warp(f_ir, offset, -1)
        aligned_vi = self._warp(f_vi, offset, 1)
        return aligned_ir, aligned_vi, offset

    def _fuse_band(self, band_ir, band_vi, weight, refiner, mode='learned'):
        if mode != 'learned':
            energy_ir = F.avg_pool2d(
                band_ir.square(), self.energy_kernel, stride=1, padding=self.energy_kernel // 2
            )
            energy_vi = F.avg_pool2d(
                band_vi.square(), self.energy_kernel, stride=1, padding=self.energy_kernel // 2
            )
            weight_ir = energy_ir / (energy_ir + energy_vi + 1e-6)
            energy_fused = weight_ir * band_ir + (1 - weight_ir) * band_vi
        if mode != 'energy':
            weights = torch.softmax(weight(torch.cat((band_ir, band_vi), dim=1)), dim=1)
            learned_fused = weights[:, 0:1] * band_ir + weights[:, 1:2] * band_vi
        if mode == 'energy':
            fused = energy_fused
        elif mode == 'mix':
            fused = (1 - self.high_energy_mix) * learned_fused + self.high_energy_mix * energy_fused
        else:
            fused = learned_fused
        return fused + refiner(fused)

    def _forward_wavelet(self, x):
        return self.lifting(x) if self.wavelet == 'lifting' else haar_forward(x)

    def _inverse_wavelet(self, bands, output_size):
        if self.wavelet == 'lifting':
            return self.lifting.inverse(*bands, output_size=output_size)
        return haar_inverse(*bands, output_size=output_size)

    def forward(self, f_ir, f_vi, return_aux=False, low_frequency=None):
        aligned_ir, aligned_vi, offset = self._align(f_ir, f_vi)
        bands_ir = self._forward_wavelet(aligned_ir)
        bands_vi = self._forward_wavelet(aligned_vi)
        low_ir, low_vi = bands_ir[0], bands_vi[0]
        if low_frequency is not None:
            raw_ir, raw_vi = low_frequency
            aligned_low_ir = self._warp(raw_ir, offset, -1)
            aligned_low_vi = self._warp(raw_vi, offset, 1)
            low_ir = self._forward_wavelet(aligned_low_ir)[0]
            low_vi = self._forward_wavelet(aligned_low_vi)[0]
        ll = self._fuse_band(low_ir, low_vi, self.low_weight, self.low_fusion)
        high = [
            self._fuse_band(
                bands_ir[i], bands_vi[i], self.high_weight, self.high_fusion,
                mode=self.high_fusion_mode,
            )
            for i in range(1, 4)
        ]
        if self.band_mask is not None:
            if not self.band_mask[0]:
                ll = torch.zeros_like(ll)
            high = [
                band if self.band_mask[index + 1] else torch.zeros_like(band)
                for index, band in enumerate(high)
            ]
        fused = self._inverse_wavelet((ll, *high), output_size=f_ir.shape[-2:])
        fused = self.output(fused)
        if return_aux:
            aux = {
                'offset': offset,
                'mean_offset': offset.abs().mean().detach(),
                'aligned_ir': aligned_ir,
                'aligned_vi': aligned_vi,
                'bands': (ll, *high),
            }
            if self.alignment_mode == 'transport':
                aux.update(self.transport_aux)
            return fused, aux
        return fused
