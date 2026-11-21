import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.frequency import multilevel_haar


class FrequencyContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.2, levels=2, pool_size=4):
        super().__init__()
        self.temperature = temperature
        self.levels = levels
        self.pool_size = pool_size

    def _embedding(self, band):
        pooled = F.adaptive_avg_pool2d(band, self.pool_size).flatten(1)
        magnitude = F.adaptive_avg_pool2d(band.abs(), self.pool_size).flatten(1)
        embedding = torch.cat((pooled, magnitude), dim=1)
        return F.normalize(embedding, dim=1, eps=1e-6)

    def forward(self, fused, ir, vi):
        fused_bands = multilevel_haar(fused, self.levels)
        ir_bands = multilevel_haar(ir, self.levels)
        vi_bands = multilevel_haar(vi, self.levels)
        names = list(fused_bands)
        anchors_by_band = [self._embedding(fused_bands[name]) for name in names]
        positives_by_band = [
            torch.stack((self._embedding(ir_bands[name]), self._embedding(vi_bands[name])), dim=1)
            for name in names
        ]
        batch = fused.shape[0]
        candidates = torch.stack(positives_by_band, dim=0).flatten(0, 2)
        rows = torch.arange(batch, device=fused.device)
        losses = []

        for band_index, anchors in enumerate(anchors_by_band):
            logits = anchors @ candidates.t() / self.temperature
            positive_mask = torch.zeros_like(logits, dtype=torch.bool)
            offset = band_index * batch * 2
            positive_mask[rows, offset + rows * 2] = True
            positive_mask[rows, offset + rows * 2 + 1] = True
            log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
            losses.append(-(log_prob * positive_mask).sum(dim=1) / 2)

        return torch.cat(losses).mean()
