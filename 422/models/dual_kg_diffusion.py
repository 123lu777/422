import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.MISCFilterNet_Deform import MISCKernelNet_Deform as MISCDeformNet


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

def _add_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k if k.startswith("module.") else f"module.{k}": v for k, v in state_dict.items()}

def _extract_state_dict(ckpt: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt

def _load_expert_weights(model: nn.Module, weights_path: str, device: torch.device) -> None:
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = _extract_state_dict(ckpt)
    model_is_dp = isinstance(model, nn.DataParallel)
    has_module = any(k.startswith("module.") for k in state_dict.keys())

    if model_is_dp and not has_module:
        state_dict = _add_module_prefix(state_dict)
    elif (not model_is_dp) and has_module:
        state_dict = _strip_module_prefix(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[DualKG] Missing keys while loading {weights_path}: {len(missing)}")
    if unexpected:
        print(f"[DualKG] Unexpected keys while loading {weights_path}: {len(unexpected)}")

def _freeze_model(model: nn.Module) -> None:
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

def _flow_at(flows: Sequence[torch.Tensor], idx: int) -> Optional[torch.Tensor]:
    return flows[idx] if len(flows) > idx else None

def _to_neg_one_to_one(x: torch.Tensor) -> torch.Tensor:
    if x.ndim <= 1:
        x_min = x.detach().amin()
        x_max = x.detach().amax()
        if x_min >= 0.0 and x_max <= 1.0:
            return x * 2.0 - 1.0
        return torch.clamp(x, -1.0, 1.0)

    reduce_dims = tuple(range(1, x.ndim))
    x_min = x.detach().amin(dim=reduce_dims, keepdim=True)
    x_max = x.detach().amax(dim=reduce_dims, keepdim=True)
    in_zero_one = (x_min >= 0.0) & (x_max <= 1.0)
    scaled = x * 2.0 - 1.0
    clipped = torch.clamp(x, -1.0, 1.0)
    return torch.where(in_zero_one, scaled, clipped)

def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    if dim < 2:
        raise ValueError("timestep embedding dimension must be >= 2 to generate sinusoidal embeddings")
    half_dim = dim // 2
    exponent = -math.log(10000.0) * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
    exponent = exponent / half_dim
    emb = timesteps.float().unsqueeze(1) * torch.exp(exponent).unsqueeze(0)
    emb = torch.cat([emb.sin(), emb.cos()], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class PriorExtractor(nn.Module):
    def __init__(
        self,
        general_weights_path: str,
        kinematic_weights_path: str,
        use_deform_in_feat: bool = True,
        use_deform_in_encoder: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.general_expert = MISCDeformNet(
            inference=False,
            use_deform_in_feat=use_deform_in_feat,
            use_deform_in_encoder=use_deform_in_encoder,
        ).to(self.device)
        self.kinematic_expert = MISCDeformNet(
            inference=False,
            use_deform_in_feat=use_deform_in_feat,
            use_deform_in_encoder=use_deform_in_encoder,
        ).to(self.device)

        _load_expert_weights(self.general_expert, general_weights_path, self.device)
        _load_expert_weights(self.kinematic_expert, kinematic_weights_path, self.device)
        _freeze_model(self.general_expert)
        _freeze_model(self.kinematic_expert)

    @staticmethod
    def _extract_primary_output(model_out) -> torch.Tensor:
        if torch.is_tensor(model_out):
            return model_out
        if isinstance(model_out, (tuple, list)):
            first = model_out[0]
            if torch.is_tensor(first):
                return first
            if isinstance(first, (tuple, list)) and len(first) > 0 and torch.is_tensor(first[0]):
                return first[0]
        raise TypeError("Unsupported expert output format: expected Tensor or tuple/list with Tensor output")

    @torch.no_grad()
    def forward(self, blur_img: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        general_out = self.general_expert(blur_img)
        base_img = self._extract_primary_output(general_out)

        kinematic_out = self.kinematic_expert(blur_img, return_flows=True)
        if isinstance(kinematic_out, (tuple, list)) and len(kinematic_out) == 3:
            _, _, flows = kinematic_out
        elif isinstance(kinematic_out, (tuple, list)) and len(kinematic_out) == 2:
            _, flows = kinematic_out
        else:
            raise TypeError(
                "Unsupported kinematic expert output format: expected tuple/list with flows (length 2 or 3)"
            )

        return base_img, flows


class KinematicModulation(nn.Module):
    def __init__(self, feat_channels: int, flow_channels: int = 2, hidden_channels: Optional[int] = None) -> None:
        super().__init__()
        hidden_channels = hidden_channels or feat_channels
        self.flow_encoder = nn.Sequential(
            nn.Conv2d(flow_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, feat_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.affine = nn.Conv2d(feat_channels * 2, feat_channels * 2, kernel_size=1)

    def forward(self, feat: torch.Tensor, flow: Optional[torch.Tensor]) -> torch.Tensor:
        if flow is None:
            return feat
        flow_feat = self.flow_encoder(flow)
        gamma_beta = self.affine(torch.cat([feat, flow_feat], dim=1))
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return feat * (1.0 + gamma) + beta


class ResidualTimeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, t_embed_dim: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(t_embed_dim, out_channels)
        self.act = nn.SiLU(inplace=True)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.act(self.conv2(h))
        return h + self.skip(x)


class SimpleCondUNet(nn.Module):
    """
    条件 UNet，支持可选的多尺度融合特征残差注入。
    若不给 extra_condition_channels，则完全等同于原版。
    """
    def __init__(
        self,
        in_channels: int = 3,
        cond_channels: int = 6,
        base_channels: int = 64,
        t_embed_dim: int = 128,
        extra_condition_channels: Optional[List[int]] = None,  # ★ 新增
    ) -> None:
        super().__init__()
        self.t_embed_dim = t_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(t_embed_dim, t_embed_dim * 2),
            nn.SiLU(inplace=True),
            nn.Linear(t_embed_dim * 2, t_embed_dim),
        )

        self.input_proj = nn.Conv2d(in_channels + cond_channels, base_channels, 3, padding=1)

        self.enc1 = ResidualTimeBlock(base_channels, base_channels, t_embed_dim)
        self.enc2 = ResidualTimeBlock(base_channels * 2, base_channels * 2, t_embed_dim)
        self.mid = ResidualTimeBlock(base_channels * 2, base_channels * 2, t_embed_dim)
        self.dec_low = ResidualTimeBlock(base_channels * 4, base_channels * 2, t_embed_dim)
        self.dec_high = ResidualTimeBlock(base_channels * 2, base_channels, t_embed_dim)
        self.refine = ResidualTimeBlock(base_channels, base_channels, t_embed_dim)

        self.down = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)
        self.up = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.out = nn.Conv2d(base_channels, in_channels, 3, padding=1)

        self.mod1 = KinematicModulation(base_channels, flow_channels=2)
        self.mod2 = KinematicModulation(base_channels * 2, flow_channels=2)
        self.mod3 = KinematicModulation(base_channels * 2, flow_channels=2)

        # ★ 新增：多尺度融合特征投影层
        if extra_condition_channels is not None:
            # 两个投影层，分别对应 skip2 (e2, base*2) 和 skip1 (e1, base)
            self.extra_projections = nn.ModuleList([
                nn.Conv2d(extra_condition_channels[0], base_channels * 2, 1),
                nn.Conv2d(extra_condition_channels[1], base_channels, 1),
            ])
        else:
            self.extra_projections = None

    @staticmethod
    def _resize_flow(flow: Optional[torch.Tensor], target: torch.Tensor) -> Optional[torch.Tensor]:
        if flow is None:
            return None
        if flow.shape[-2:] == target.shape[-2:]:
            return flow
        return F.interpolate(flow, size=target.shape[-2:], mode="bilinear", align_corners=False)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        cond_img: torch.Tensor,
        flows: Optional[Sequence[torch.Tensor]] = None,
        extra_condition: Optional[List[torch.Tensor]] = None,  # ★ 新增
    ) -> torch.Tensor:
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        if timestep.dim() == 1 and timestep.shape[0] == 1 and noisy_latent.shape[0] > 1:
            timestep = timestep.repeat(noisy_latent.shape[0])

        t_emb = timestep_embedding(timestep, self.t_embed_dim)
        t_emb = self.time_mlp(t_emb)

        flows = list(flows) if flows is not None else []
        f1 = _flow_at(flows, 0)
        f2 = _flow_at(flows, 1)
        f3 = _flow_at(flows, 2)

        x = torch.cat([noisy_latent, cond_img], dim=1)
        x0 = self.input_proj(x)

        # ---- encoder ----
        e1 = self.enc1(x0, t_emb)
        e1 = self.mod1(e1, self._resize_flow(f1, e1))

        d = self.down(e1)
        e2 = self.enc2(d, t_emb)
        e2 = self.mod2(e2, self._resize_flow(f2, e2))

        m = self.mid(e2, t_emb)
        m = self.mod3(m, self._resize_flow(f3, m))

        # ★ 新增：在 skip2 (e2) 处注入融合特征
        if self.extra_projections is not None and extra_condition is not None and len(extra_condition) > 0:
            cond = extra_condition[0]
            if cond.shape[-2:] != e2.shape[-2:]:
                cond = F.interpolate(cond, size=e2.shape[-2:], mode='bilinear', align_corners=False)
            e2 = e2 + self.extra_projections[0](cond)

        # ---- decoder ----
        low = torch.cat([m, e2], dim=1)
        low = self.dec_low(low, t_emb)

        u = self.up(low)
        if u.shape[-2:] != e1.shape[-2:]:
            u = F.interpolate(u, size=e1.shape[-2:], mode='bilinear', align_corners=False)

        # ★ 新增：在 skip1 (e1) 处注入融合特征
        if self.extra_projections is not None and extra_condition is not None and len(extra_condition) > 1:
            cond = extra_condition[1]
            if cond.shape[-2:] != e1.shape[-2:]:
                cond = F.interpolate(cond, size=e1.shape[-2:], mode='bilinear', align_corners=False)
            e1 = e1 + self.extra_projections[1](cond)

        u = torch.cat([u, e1], dim=1)
        u = self.dec_high(u, t_emb)
        u = self.refine(u, t_emb)
        return self.out(u)


class DualKGDiffusionModel(nn.Module):
    def __init__(
        self,
        general_weights_path: str,
        kinematic_weights_path: str,
        use_deform_in_feat: bool = True,
        use_deform_in_encoder: bool = True,
        unet_base_channels: int = 64,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prior_extractor = PriorExtractor(
            general_weights_path=general_weights_path,
            kinematic_weights_path=kinematic_weights_path,
            use_deform_in_feat=use_deform_in_feat,
            use_deform_in_encoder=use_deform_in_encoder,
            device=self.device,
        )
        self.denoiser = SimpleCondUNet(base_channels=unet_base_channels)

    def forward(
        self,
        blur_img: torch.Tensor,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        base_img, flows = self.prior_extractor(blur_img)
        blur_cond = _to_neg_one_to_one(blur_img)
        base_img = _to_neg_one_to_one(base_img)
        cond_img = torch.cat([blur_cond, base_img], dim=1)
        pred_noise = self.denoiser(noisy_latent, timestep, cond_img, flows)
        flow_s1 = _flow_at(flows, 0)
        flow_s2 = _flow_at(flows, 1)
        flow_s3 = _flow_at(flows, 2)
        return {
            "pred_noise": pred_noise,
            "base_img": base_img,
            "flow_s1": flow_s1,
            "flow_s2": flow_s2,
            "flow_s3": flow_s3,
        }
