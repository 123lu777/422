"""
双专家自适应融合扩散去模糊 —— 完整集成管线

组装关系:
  你的 MISCFilterNet(冻结) → PixelGateFusion + LightweightFeatExtractor
                           → MultiScaleFeatureFuser
                           → 你的 SimpleCondUNet(可训练)
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict

from models.adaptive_expert_fusion import (
    PixelGateFusion,
    MultiScaleFeatureFuser,
    LightweightFeatureExtractor,
)


class DualExpertDiffusionPipeline(nn.Module):

    def __init__(
        self,
        expert_gopro: nn.Module,
        expert_wind: nn.Module,
        unet: nn.Module,
        unet_feat_channels: List[int],
        alpha_clamp: float = 0.4,
        fuse_flows: str = 'average',
    ):
        super().__init__()
        self.alpha_clamp = alpha_clamp
        self.fuse_flows = fuse_flows

        # 你的已有组件
        self.expert_gopro = expert_gopro
        self.expert_wind = expert_wind
        self.unet = unet

        # 新增融合模块
        self.pixel_gate = PixelGateFusion(in_channels=6)
        self.shared_feat_extractor = LightweightFeatureExtractor(
            in_channels=3, channel_list=unet_feat_channels
        )
        self.multi_scale_fuser = MultiScaleFeatureFuser(unet_feat_channels)

    def extract_expert_outputs(self, blurry_image: torch.Tensor) -> Dict:
        """
        调用两个冻结专家

        ★★★ 根据你的 MISCFilterNet_Deform 实际接口修改此处 ★★★

        方式A (直接推理):
            base_img = self.expert_gopro(blurry_image)

        方式B (window 分块推理):
            base_img = self._forward_expert_windowed(self.expert_gopro, blurry_image)
        """
        with torch.no_grad():
            # ====== 请替换为你的实际推理代码 ======
            base_img_gopro = self.expert_gopro(blurry_image)
            base_img_wind = self.expert_wind(blurry_image)

            # 如果需要 window 分块推理, 注释掉上面两行, 取消注释下面:
            # base_img_gopro = self._forward_expert_windowed(
            #     self.expert_gopro, blurry_image, win_size=256)
            # base_img_wind = self._forward_expert_windowed(
            #     self.expert_wind, blurry_image, win_size=256)

        return {
            'base_img_gopro': base_img_gopro,
            'base_img_wind': base_img_wind,
            # 如果需要光流, 取消注释:
            # 'flow_gopro': flow_gopro,
            # 'flow_wind': flow_wind,
        }

    def _forward_expert_windowed(self, model, inp, win_size=256):
        """Window 分块推理 (需要 from models.layers import window_partitionx, window_reversex)"""
        from models.layers import window_partitionx, window_reversex
        _, _, Hx, Wx = inp.shape
        input_re, batch_list = window_partitionx(inp, win_size)
        restored, _ = model(input_re)
        restored = restored[0]
        restored = window_reversex(restored, win_size, Hx, Wx, batch_list)
        restored = torch.clamp(restored, 0, 1)
        return restored

    def fuse_flows_fn(self, flow_gopro, flow_wind):
        if flow_gopro is None and flow_wind is None:
            return None
        if self.fuse_flows == 'average':
            if flow_gopro is not None and flow_wind is not None:
                return (flow_gopro + flow_wind) / 2.0
            return flow_gopro if flow_gopro is not None else flow_wind
        elif self.fuse_flows == 'gopro_only':
            return flow_gopro
        elif self.fuse_flows == 'wind_only':
            return flow_wind
        return flow_gopro

    def forward(self, noisy_image, t, blurry_image, return_alpha_maps=False):
        # Step 1: 冻结专家推理
        expert_out = self.extract_expert_outputs(blurry_image)
        base_img_gopro = expert_out['base_img_gopro']
        base_img_wind = expert_out['base_img_wind']

        # Step 2: 像素级融合
        fused_base_img, pixel_alpha = self.pixel_gate(
            base_img_gopro, base_img_wind, alpha_clamp=self.alpha_clamp
        )

        # Step 3: 多尺度特征融合
        feats_gopro = self.shared_feat_extractor(base_img_gopro)
        feats_wind = self.shared_feat_extractor(base_img_wind)
        fused_feats, feat_alphas = self.multi_scale_fuser(feats_gopro, feats_wind)

        # Step 4: 运动场融合
        fused_flow = self.fuse_flows_fn(
            expert_out.get('flow_gopro'), expert_out.get('flow_wind')
        )

        # Step 5: 调用你的 SimpleCondUNet
        # ★★★ 替换为你的 UNet 实际调用签名 ★★★
        noise_pred = self.unet(
            noisy_image, t,
            blur_img=blurry_image,
            base_img=fused_base_img,
            flow=fused_flow,
            extra_condition=fused_feats,
        )

        if return_alpha_maps:
            return noise_pred, {
                'pixel_alpha': pixel_alpha,
                'feat_alphas': feat_alphas,
            }
        return noise_pred
