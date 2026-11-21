import sys
from pathlib import Path

import torch
import torch.nn as nn


class CDDFuseTeacher(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        project_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(project_root / 'external' / 'MMIF-CDDFuse'))
        from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction

        self.encoder = nn.DataParallel(Restormer_Encoder())
        self.decoder = nn.DataParallel(Restormer_Decoder())
        self.base_fuse = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8))
        self.detail_fuse = nn.DataParallel(DetailFeatureExtraction(num_layers=1))
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.encoder.load_state_dict(checkpoint.get('CDDF_Encoder', checkpoint['DIDF_Encoder']))
        self.decoder.load_state_dict(checkpoint.get('CDDF_Decoder', checkpoint['DIDF_Decoder']))
        self.base_fuse.load_state_dict(checkpoint['BaseFuseLayer'])
        self.detail_fuse.load_state_dict(checkpoint['DetailFuseLayer'])
        for module in (self.encoder, self.decoder, self.base_fuse, self.detail_fuse):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def forward(self, ir, vi):
        feature_v_b, feature_v_d, _ = self.encoder(vi)
        feature_i_b, feature_i_d, _ = self.encoder(ir)
        feature_f_b = self.base_fuse(feature_v_b + feature_i_b)
        feature_f_d = self.detail_fuse(feature_v_d + feature_i_d)
        fused, _ = self.decoder(vi, feature_f_b, feature_f_d)
        data_min, data_max = torch.min(fused), torch.max(fused)
        return (fused - data_min) / (data_max - data_min).clamp_min(1e-8)
