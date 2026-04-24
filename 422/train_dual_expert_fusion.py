"""
双专家融合扩散模型训练脚本

训练策略:
  1. 两个 MISCFilterNet 专家完全冻结
  2. 线性噪声调度 (与已有预训练兼容)
  3. 损失函数: warmup 阶段仅 L2, 之后加 Charbonnier
  4. 差异化学习率: 融合模块 2x, UNet 1x
  5. warmup 结束后自动保存 alpha 可视化图
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ================================================================
# 1. 线性噪声调度
# ================================================================

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def q_sample(x_start, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    sqrt_a = sqrt_alphas_cumprod[t][:, None, None, None]
    sqrt_1ma = sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
    return sqrt_a * x_start + sqrt_1ma * noise


# ================================================================
# 2. 损失函数 (先 L2 warmup, 后加 Charbonnier)
# ================================================================

class StableDeblurLoss(nn.Module):
    def __init__(self, warmup_epochs=50):
        super().__init__()
        self.warmup_epochs = warmup_epochs

    @staticmethod
    def charbonnier(pred, target, eps=1e-6):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + eps * eps))

    def forward(self, pred, target, current_epoch=0):
        loss = F.mse_loss(pred, target)
        if current_epoch >= self.warmup_epochs:
            loss = loss + 0.5 * self.charbonnier(pred, target)
        return loss


# ================================================================
# 3. 数据集
# ================================================================

class DeblurPairDataset(Dataset):
    def __init__(self, blur_dir, sharp_dir, crop_size=256):
        self.blur_paths = sorted([
            os.path.join(blur_dir, f) for f in os.listdir(blur_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        self.sharp_paths = sorted([
            os.path.join(sharp_dir, f) for f in os.listdir(sharp_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        assert len(self.blur_paths) == len(self.sharp_paths)
        self.crop_size = crop_size

    def __len__(self):
        return len(self.blur_paths)

    def __getitem__(self, idx):
        blur = Image.open(self.blur_paths[idx]).convert('RGB')
        sharp = Image.open(self.sharp_paths[idx]).convert('RGB')

        if self.crop_size > 0:
            w, h = blur.size
            cs = min(self.crop_size, w, h)
            i = torch.randint(0, h - cs + 1, (1,)).item()
            j = torch.randint(0, w - cs + 1, (1,)).item()
            blur = blur.crop((j, i, j + cs, i + cs))
            sharp = sharp.crop((j, i, j + cs, i + cs))
            if torch.rand(1).item() > 0.5:
                blur = blur.transpose(Image.FLIP_LEFT_RIGHT)
                sharp = sharp.transpose(Image.FLIP_LEFT_RIGHT)

        blur = transforms.ToTensor()(blur) * 2.0 - 1.0
        sharp = transforms.ToTensor()(sharp) * 2.0 - 1.0
        return {'blurry': blur, 'sharp': sharp}


# ================================================================
# 4. Alpha 可视化工具
# ================================================================

def save_alpha_maps(alpha_maps_dict, save_dir, tag=''):
    """保存像素级和多尺度 alpha 热力图"""
    import cv2
    import numpy as np
    os.makedirs(save_dir, exist_ok=True)

    pixel_alpha = alpha_maps_dict['pixel_alpha'][0, 0].cpu().numpy()
    pixel_uint8 = (pixel_alpha * 255).astype(np.uint8)
    h, w = pixel_uint8.shape
    heatmap = cv2.applyColorMap(pixel_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(save_dir, f'alpha_pixel_{tag}.png'), heatmap)

    for i, fa in enumerate(alpha_maps_dict.get('feat_alphas', [])):
        fa_np = (fa[0, 0].cpu().numpy() * 255).astype(np.uint8)
        fa_up = cv2.resize(fa_np, (w, h))
        hm = cv2.applyColorMap(fa_up, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(save_dir, f'alpha_scale{i}_{tag}.png'), hm)

    print(f"  📊 Alpha 图已保存至: {save_dir}/alpha_*_{tag}.png")


# ================================================================
# 5. 模型构建
# ================================================================

def build_pipeline(config):
    from models.MISCFilterNet_Deform import MISCKernelNet_Deform
    from models.dual_expert_diffusion import DualExpertDiffusionPipeline
    from models.dual_kg_diffusion import SimpleCondUNet  # 包含 extra_condition 支持的版本

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建专家
    expert_gopro = MISCKernelNet_Deform(
        inference=False,
        use_deform_in_feat=config.get('use_deform_in_feat', True),
        use_deform_in_encoder=config.get('use_deform_in_encoder', True),
    )
    expert_wind = MISCKernelNet_Deform(
        inference=False,
        use_deform_in_feat=config.get('use_deform_in_feat', True),
        use_deform_in_encoder=config.get('use_deform_in_encoder', True),
    )

    # 加载权重辅助函数
    def load_weights(model, path, label=""):
        ckpt = torch.load(path, map_location='cpu')
        sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        new_sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        if missing:
            print(f"  [{label}] 缺失: {len(missing)}个")
        if unexpected:
            print(f"  [{label}] 多余: {len(unexpected)}个")
        print(f"  ✔ [{label}] 已加载: {path}")

    print("加载专家权重:")
    load_weights(expert_gopro, config['gopro_ckpt'], 'GoPro')
    load_weights(expert_wind, config['wind_ckpt'], '风机')

    # 冻结专家
    for p in expert_gopro.parameters():
        p.requires_grad = False
    for p in expert_wind.parameters():
        p.requires_grad = False
    expert_gopro.eval()
    expert_wind.eval()
    print("✔ 两个专家已完全冻结\n")

    # ─── 构建 UNet ───
    # ★ 关键：extra_condition_channels 必须与管道中传入的 extra_condition 列表通道数一致
    #    这里使用 unet_feat_channels[0] (即第一个尺度的通道数) 作为投影通道
    feat0_channels = config['unet_feat_channels'][0]  # 例如 64
    unet = SimpleCondUNet(
        in_channels=3,               # 输出 RGB
        cond_channels=6,             # blur (3) + fused_base_img (3) = 6
        base_channels=64,            # 基础通道数
        t_embed_dim=128,
        extra_condition_channels=[feat0_channels, feat0_channels],  # 对应 e2 和 e1 的注入
    )

    if 'unet_ckpt' in config:
        load_weights(unet, config['unet_ckpt'], 'UNet')

    # ─── 组装总管线 ───
    pipeline = DualExpertDiffusionPipeline(
        expert_gopro=expert_gopro,
        expert_wind=expert_wind,
        unet=unet,
        unet_feat_channels=config['unet_feat_channels'],
        alpha_clamp=config.get('alpha_clamp', 0.4),
        fuse_flows=config.get('fuse_flows', 'average'),
    )

    # 参数统计
    total = sum(p.numel() for p in pipeline.parameters())
    trainable = sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    # ★ 修正括号错误
    frozen = total - trainable
    print(f"参数: Total={total/1e6:.2f}M, Trainable={trainable/1e6:.2f}M, "
          f"Frozen={frozen/1e6:.2f}M\n")

    pipeline = pipeline.to(device)
    if torch.cuda.device_count() > 1:
        pipeline = nn.DataParallel(pipeline)

    return pipeline, device


# ================================================================
# 6. 主训练循环
# ================================================================

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    T = config.get('timesteps', 1000)
    batch_size = config.get('batch_size', 4)
    epochs = config.get('epochs', 500)
    base_lr = config.get('lr', 2e-4)
    warmup_epochs = config.get('warmup_epochs', 50)
    save_every = config.get('save_every', 50)

    # 调度器
    betas = linear_beta_schedule(T).to(device)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # 损失
    criterion = StableDeblurLoss(warmup_epochs=warmup_epochs).to(device)

    # 数据
    dataset = DeblurPairDataset(
        config['blur_dir'], config['sharp_dir'],
        crop_size=config.get('crop_size', 256),
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True, drop_last=True,
    )
    print(f"数据: {len(dataset)} 对, batch={batch_size}, "
          f"每epoch {len(dataloader)} 步\n")

    # 模型
    model, _ = build_pipeline(config)
    # model 已经在 build_pipeline 里 .to(device) 了，这里可以省略
    model = model.to(device)

    # 差异化学习率
    optimizer = torch.optim.AdamW([
        {"params": model.pixel_gate.parameters(), "lr": base_lr * 2.0},
        {"params": model.shared_feat_extractor.parameters(), "lr": base_lr},
        {"params": model.multi_scale_fuser.parameters(), "lr": base_lr * 2.0},
        {"params": model.unet.parameters(), "lr": base_lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=base_lr * 0.01
    )

    alpha_save_dir = config.get('alpha_save_dir', 'visualize/alpha')

    print(f"{'=' * 60}")
    print(f"开始训练: {epochs} epochs, T={T}, lr={base_lr}")
    print(f"{'=' * 60}\n")

    for epoch in range(epochs):
        model.train()
        is_warmup = epoch < warmup_epochs
        phase = "WARMUP(L2)" if is_warmup else "FULL(L2+Charb)"
        total_loss = 0.0

        for step, batch in enumerate(dataloader):
            blurry = batch['blurry'].to(device)
            sharp = batch['sharp'].to(device)
            B = blurry.shape[0]

            t = torch.randint(0, T, (B,), device=device).long()
            noise = torch.randn_like(sharp)
            noisy = q_sample(sharp, t, noise,
                             sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)

            noise_pred = model(noisy, t, blurry)      # 管线前向: (noisy_image, t, blurry_image)
            loss = criterion(noise_pred, noise, current_epoch=epoch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()

            total_loss += loss.item()

            if step % 50 == 0:
                print(f"  [{phase}] Ep[{epoch}/{epochs}] "
                      f"St[{step}/{len(dataloader)}] "
                      f"Loss={loss.item():.6f}")

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"\n  ► Epoch {epoch} | Avg Loss: {avg_loss:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}\n")

        # 保存 alpha 可视化 (warmup 结束后每 20 个 epoch)
        if epoch >= warmup_epochs and epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                test_blur = blurry[:1]
                test_t = torch.randint(0, T, (1,), device=device)
                test_noisy = torch.randn_like(test_blur)
                _, alpha_maps = model(test_noisy, test_t, test_blur,
                                      return_alpha_maps=True)
                save_alpha_maps(alpha_maps, alpha_save_dir, tag=f'e{epoch}')
            model.train()

        # 保存权重
        if (epoch + 1) % save_every == 0:
            save_path = os.path.join(
                config.get('save_dir', 'checkpoints/dual_expert'),
                f'model_epoch_{epoch + 1}.pth'
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"  💾 已保存: {save_path}\n")


# ================================================================
# 7. 配置与启动
# ================================================================

if __name__ == '__main__':

    config = {
        # 数据 (请替换为你的实际路径)
        'blur_dir': '/path/to/your/mixed_train_blur',
        'sharp_dir': '/path/to/your/mixed_train_sharp',
        'crop_size': 256,

        # 专家权重 (已填写你的路径)
        'gopro_ckpt': '/media/JYJ/新加卷/ZJL/checkpoints_deform/GoPro/MISCFilter_Deform_GoPro/model_epoch_1162.pth',
        'wind_ckpt': '/media/JYJ/新加卷/ZJL/340checkpoints/1/2/model_epoch_564.pth',
        # 'unet_ckpt': '/path/to/your/pretrained_unet.pth',   # 可选

        # 模型参数
        'use_deform_in_feat': True,
        'use_deform_in_encoder': True,
        'unet_feat_channels': [64, 128, 256, 512],      # 与 LightweightFeatureExtractor 一致
        'alpha_clamp': 0.4,

        # 训练参数
        'timesteps': 1000,
        'batch_size': 4,
        'epochs': 500,
        'lr': 2e-4,
        'warmup_epochs': 50,
        'save_every': 50,
        'save_dir': 'checkpoints/dual_expert',
        'alpha_save_dir': 'visualize/alpha',
        'num_workers': 4,
    }

    train(config)
