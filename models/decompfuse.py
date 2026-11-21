import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder
from .modules.adaptive_compensation import AdaptiveSourceDetailCompensation
from .modules.decomposition import AdaptiveIterativeDecomposition
from .modules.frequency import DeformableFrequencyDecoupling
from .modules.fusion import FusionLayer
from .modules.illumination import IlluminationSaliencyWeight


class DecompFuse(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, use_aid=True, use_dfd=True, use_skip=False,
                 use_adaptive_intensity=False, use_source_reconstruction=False,
                 source_reconstruction_mode='features', aid_high_only=False,
                 aid_config=None, dfd_config=None, use_global=False, global_pool=8, global_heads=4,
                 output_anchor=False, anchor_residual_scale=0.1,
                 use_adaptive_compensation=False, compensation_config=None):
        super().__init__()
        if source_reconstruction_mode not in ('features', 'fused'):
            raise ValueError("source_reconstruction_mode must be 'features' or 'fused'")
        self.use_aid = use_aid
        self.use_dfd = use_dfd
        self.use_skip = use_skip
        self.use_adaptive_intensity = use_adaptive_intensity
        self.use_source_reconstruction = use_source_reconstruction
        self.source_reconstruction_mode = source_reconstruction_mode
        self.aid_high_only = aid_high_only
        self.output_anchor = output_anchor
        encoder_config = {
            'in_ch': in_ch, 'base_ch': base_ch, 'use_global': use_global,
            'global_pool': global_pool, 'global_heads': global_heads,
        }
        self.encoder_ir = Encoder(**encoder_config)
        self.encoder_vi = Encoder(**encoder_config)
        self.aid = AdaptiveIterativeDecomposition(base_ch, **(aid_config or {})) if use_aid else None
        if use_dfd:
            self.fusion = DeformableFrequencyDecoupling(base_ch, **(dfd_config or {}))
        else:
            self.fusion = FusionLayer(base_ch)
        self.decoder = Decoder(
            base_ch=base_ch, use_skip=use_skip, output_anchor=output_anchor,
            residual_scale=anchor_residual_scale,
        )
        self.source_decoder_ir = Decoder(base_ch=base_ch) if use_source_reconstruction else None
        self.source_decoder_vi = Decoder(base_ch=base_ch) if use_source_reconstruction else None
        self.intensity_weight = IlluminationSaliencyWeight() if use_adaptive_intensity else None
        self.adaptive_compensation = (AdaptiveSourceDetailCompensation(**(compensation_config or {}))
                                      if use_adaptive_compensation else None)

    def _add_source_reconstruction(self, aux, f_ir, f_vi, fused):
        if self.use_source_reconstruction:
            if self.source_reconstruction_mode == 'fused':
                f_ir = f_vi = fused
            aux['source_ir'] = self.source_decoder_ir(f_ir)
            aux['source_vi'] = self.source_decoder_vi(f_vi)

    def forward(self, ir, vi, return_aux=False, return_steps=False):
        f0_ir, _, f_ir = self.encoder_ir(ir)
        f0_vi, _, f_vi = self.encoder_vi(vi)
        aux = {}
        if self.use_adaptive_intensity and return_aux:
            weight, prior = self.intensity_weight(ir, vi)
            aux['intensity_weight'] = weight
            aux['intensity_prior'] = prior
        low_frequency = (f_ir, f_vi) if self.aid_high_only else None
        if self.use_aid:
            f_ir, f_vi, aid_aux = self.aid(f_ir, f_vi, return_steps=return_steps)
            aux['aid'] = aid_aux
        if self.use_dfd:
            fused, dfd_aux = self.fusion(
                f_ir, f_vi, return_aux=True, low_frequency=low_frequency
            )
            aux['dfd'] = dfd_aux
        else:
            fused = self.fusion(f_ir, f_vi)
        out = self.decoder(fused, f0_ir, f0_vi, anchor=0.5 * (ir + vi)) if self.use_skip else self.decoder(fused, anchor=0.5 * (ir + vi))
        if self.adaptive_compensation is not None:
            out, compensation_aux = self.adaptive_compensation(out, ir, vi)
            aux.update(compensation_aux)
        if return_aux:
            self._add_source_reconstruction(aux, f_ir, f_vi, fused)
            return out, aux
        return out

    def forward_iterations(self, ir, vi):
        if not self.use_aid or not self.use_dfd:
            raise ValueError('Iteration supervision requires AID and DFD')
        f0_ir, _, f_ir = self.encoder_ir(ir)
        f0_vi, _, f_vi = self.encoder_vi(vi)
        low_frequency = (f_ir, f_vi) if self.aid_high_only else None
        _, _, aid_aux = self.aid(f_ir, f_vi, return_steps=True)
        aux = {}
        if self.use_adaptive_intensity:
            weight, prior = self.intensity_weight(ir, vi)
            aux['intensity_weight'] = weight
            aux['intensity_prior'] = prior
        aux['aid'] = aid_aux
        candidates = []
        for step_ir, step_vi in aid_aux['iteration_features']:
            fused, dfd_aux = self.fusion(
                step_ir, step_vi, return_aux=True, low_frequency=low_frequency
            )
            out = self.decoder(fused, f0_ir, f0_vi, anchor=0.5 * (ir + vi)) if self.use_skip else self.decoder(fused, anchor=0.5 * (ir + vi))
            step_aux = dict(aux)
            if self.adaptive_compensation is not None:
                out, compensation_aux = self.adaptive_compensation(out, ir, vi)
                step_aux.update(compensation_aux)
            step_aux['dfd'] = dfd_aux
            self._add_source_reconstruction(step_aux, step_ir, step_vi, fused)
            candidates.append((out, step_aux))
        return candidates
