"""
自适应双专家融合模块

模块清单:
  - SpatialGateFusion:      特征空间的自适应门控融合（多尺度）
  - PixelGateFusion:        像素空间的自适应门控融合（图像级，bias=-0.8 起步）
  - MultiScaleFeatureFuser: 多尺度融合协调器
  - LightweightFeatureExtractor: 共享轻量级特征提取适配器
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class SpatialGateFusion(nn.Module):
    """
    特征空间自适应门控融合
    初始化: softmax 输出接近 [0.5, 0.5]
    """

    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 2, 1, bias=True),
        )
        for m in self.gate.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels == 2:
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feat_gopro: torch.Tensor, feat_wind: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        concat = torch.cat([feat_gopro, feat_wind], dim=1)
        gates = torch.softmax(self.gate(concat), dim=1)
        gate_gopro = gates[:, 0:1, :, :]
        gate_wind = gates[:, 1:2, :, :]
        fused = gate_gopro * feat_gopro + gate_wind * feat_wind
        return fused, gate_wind


class PixelGateFusion(nn.Module):
    """
    像素空间自适应门控融合

    安全机制:
      alpha_clamp 限制风机权重的最大值
      bias=-0.8 → Sigmoid(-0.8) ≈ 0.31，接近手动调的最优 alpha=0.2
      网络从贴近经验值的起点开始优化，收敛更快
    """

    def __init__(self, in_channels: int = 6):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )
        self._init_bias()

    def _init_bias(self):
        """bias=-0.8, 使初始 alpha ≈ 0.31, 贴近手动调的最优 0.2"""
        for m in self.gate.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels == 1:
                nn.init.zeros_(m.weight)
                nn.init.constant_(m.bias, -0.8)

    def forward(self, img_gopro: torch.Tensor, img_wind: torch.Tensor,
                alpha_clamp: float = 0.4
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        concat = torch.cat([img_gopro, img_wind], dim=1)
        alpha = self.gate(concat)
        alpha = torch.clamp(alpha, max=alpha_clamp)
        fused_img = (1.0 - alpha) * img_gopro + alpha * img_wind
        return fused_img, alpha


class MultiScaleFeatureFuser(nn.Module):
    """多尺度特征融合协调器"""

    def __init__(self, channel_list: List[int]):
        super().__init__()
        self.fusers = nn.ModuleList([
            SpatialGateFusion(ch) for ch in channel_list
        ])
        self.num_scales = len(channel_list)

    def forward(self, feats_gopro: List[torch.Tensor],
                feats_wind: List[torch.Tensor]
                ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        assert len(feats_gopro) == len(feats_wind) == self.num_scales, \
            f"尺度数不匹配: 期望{self.num_scales}, " \
            f"GoPro={len(feats_gopro)}, Wind={len(feats_wind)}"
        fused_feats, alpha_maps = [], []
        for i in range(self.num_scales):
            fused, alpha = self.fusers[i](feats_gopro[i], feats_wind[i])
            fused_feats.append(fused)
            alpha_maps.append(alpha)
        return fused_feats, alpha_maps


class LightweightFeatureExtractor(nn.Module):
    """
    共享轻量级特征提取适配器
    将 RGB 去模糊图 → 多尺度特征 (格式转换器，不替代专家)
    """

    def __init__(self, in_channels: int = 3, channel_list: List[int] = None):
        super().__init__()
        if channel_list is None:
            channel_list = [64, 128, 256, 512]
        self.channel_list = channel_list
        self.stages = nn.ModuleList()
        in_ch = in_channels
        for out_ch in channel_list:
            self.stages.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(min(32, out_ch), out_ch),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.GroupNorm(min(32, out_ch), out_ch),
                nn.SiLU(inplace=True),
            ))
            in_ch = out_ch

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        h = x
        for stage in self.stages:
            h = stage(h)
            features.append(h)
        return features
