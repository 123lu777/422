import argparse
import os
import random
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from models.dual_kg_diffusion import DualKGDiffusionModel


class RealDeblurDataset(Dataset):
    def __init__(self, data_dir: str, meta_file: str, image_size: int = 256, is_train: bool = True) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.is_train = is_train
        self.samples = []

        with open(meta_file, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 2:
                    blur_path = parts[0] if os.path.isabs(parts[0]) else os.path.join(data_dir, parts[0])
                    sharp_path = parts[1] if os.path.isabs(parts[1]) else os.path.join(data_dir, parts[1])
                    self.samples.append((blur_path, sharp_path))

        print(f"Successfully loaded {len(self.samples)} image pairs from {meta_file}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        blur_path, sharp_path = self.samples[idx]

        blur_img = cv2.imread(blur_path)
        sharp_img = cv2.imread(sharp_path)

        if blur_img is None or sharp_img is None:
            return self.__getitem__(random.randint(0, len(self.samples) - 1))

        blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
        sharp_img = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB)

        h, w, _ = blur_img.shape

        if self.is_train:
            if h > self.image_size and w > self.image_size:
                top = random.randint(0, h - self.image_size)
                left = random.randint(0, w - self.image_size)
                blur_img = blur_img[top: top + self.image_size, left: left + self.image_size, :]
                sharp_img = sharp_img[top: top + self.image_size, left: left + self.image_size, :]
            else:
                blur_img = cv2.resize(blur_img, (self.image_size, self.image_size))
                sharp_img = cv2.resize(sharp_img, (self.image_size, self.image_size))

            if random.random() < 0.5:
                blur_img = np.flip(blur_img, axis=1)
                sharp_img = np.flip(sharp_img, axis=1)
            if random.random() < 0.5:
                blur_img = np.flip(blur_img, axis=0)
                sharp_img = np.flip(sharp_img, axis=0)
        else:
            blur_img = cv2.resize(blur_img, (self.image_size, self.image_size))
            sharp_img = cv2.resize(sharp_img, (self.image_size, self.image_size))

        blur_tensor = torch.from_numpy(np.ascontiguousarray(blur_img)).permute(2, 0, 1).float() / 255.0
        sharp_tensor = torch.from_numpy(np.ascontiguousarray(sharp_img)).permute(2, 0, 1).float() / 255.0

        return {"blur": blur_tensor, "sharp": sharp_tensor}


def normalize_to_neg_one_to_one(x: torch.Tensor) -> torch.Tensor:
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


def build_linear_beta_schedule(
        timesteps: int,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alpha_bars


def q_sample(clean_img: torch.Tensor, t: torch.Tensor, alpha_bars: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    sqrt_ab = torch.sqrt(alpha_bars[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_ab = torch.sqrt(1.0 - alpha_bars[t]).view(-1, 1, 1, 1)
    return sqrt_ab * clean_img + sqrt_one_minus_ab * noise


def train_one_step(
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: optim.Optimizer,
        alpha_bars: torch.Tensor,
        num_timesteps: int,
        device: torch.device,
) -> Dict[str, float]:
    blur = normalize_to_neg_one_to_one(batch["blur"].to(device))
    sharp = normalize_to_neg_one_to_one(batch["sharp"].to(device))

    sampled_timesteps = torch.randint(0, num_timesteps, (blur.size(0),), device=device)
    noise = torch.randn_like(sharp)
    noisy_latent = q_sample(sharp, sampled_timesteps, alpha_bars, noise)

    out = model(blur_img=blur, noisy_latent=noisy_latent, timestep=sampled_timesteps)
    pred_noise = out["pred_noise"]
    loss_noise = F.mse_loss(pred_noise, noise)
    loss = loss_noise

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "loss_noise": loss_noise.item(),
    }


def save_checkpoint(
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        save_path: str,
        args: argparse.Namespace,
) -> None:
    """保存完整的训练状态"""
    # 脱掉 DataParallel 的壳
    state_dict_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": state_dict_to_save,
        "optimizer_state_dict": optimizer.state_dict(),
        "args": args,
    }
    torch.save(checkpoint, save_path)
    print(f"💾 Checkpoint saved to {save_path}")


def load_checkpoint(
        checkpoint_path: str,
        model: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
) -> int:
    """加载完整的训练状态，返回起始epoch"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 判断checkpoint格式
    if "model_state_dict" in checkpoint:
        # 完整checkpoint格式
        model_state_dict = checkpoint["model_state_dict"]
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"✅ Full checkpoint loaded from {checkpoint_path}")
    elif "state_dict" in checkpoint:
        # 你的格式：{"state_dict": ...}
        model_state_dict = checkpoint["state_dict"]
        # 从文件名提取epoch
        import re
        match = re.search(r'epoch_(\d+)', checkpoint_path)
        start_epoch = int(match.group(1)) if match else 0
        print(f"✅ Model weights loaded from {checkpoint_path}")
    else:
        # 直接是state_dict
        model_state_dict = checkpoint
        # 从文件名提取epoch
        import re
        match = re.search(r'epoch_(\d+)', checkpoint_path)
        start_epoch = int(match.group(1)) if match else 0
        print(f"✅ Model weights loaded from {checkpoint_path}")

    # 加载模型权重
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)

    print(f"📊 Resuming from epoch {start_epoch + 1}")
    return start_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-Expert Guided Diffusion training scaffold")
    parser.add_argument("--general_weights", type=str, required=True, help="Path to GoPro-trained expert weights")
    parser.add_argument("--kinematic_weights", type=str, required=True, help="Path to kinematic expert weights")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1, help="Total epochs to run (including resumed ones)")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DualKGDiffusionModel(
        general_weights_path=args.general_weights,
        kinematic_weights_path=args.kinematic_weights,
        use_deform_in_feat=True,
        use_deform_in_encoder=True,
        unet_base_channels=64,
        device=device,
    ).to(device)

    # 开启多卡并行训练
    if torch.cuda.device_count() > 1:
        print(f"🔥 检测到 {torch.cuda.device_count()} 张显卡，开启多卡并行训练！")
        model = nn.DataParallel(model)

    dataset = RealDeblurDataset(
        data_dir="/media/JYJ/新加卷/ZJL/FanBlade_Dataset_MultiOmega",
        meta_file="/media/JYJ/新加卷/ZJL/FanBlade_Dataset_MultiOmega/fan_blade_train_list.txt",
        image_size=args.image_size,
        is_train=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    # 判断模型是否被 DataParallel 包裹，正确提取 denoiser 网络传给优化器
    unet_model = model.module.denoiser if isinstance(model, nn.DataParallel) else model.denoiser
    optimizer = optim.AdamW(unet_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ================= 断点续训逻辑 =================
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        start_epoch = load_checkpoint(args.resume, model, optimizer, device)
    else:
        if args.resume:
            print(f"⚠️ Checkpoint {args.resume} not found, starting from scratch")
    # ==============================================

    _, alpha_bars = build_linear_beta_schedule(args.timesteps, device=device)

    # 准备保存权重的文件夹
    save_dir = "checkpoints420"
    os.makedirs(save_dir, exist_ok=True)
    print(f"📁 训练权重将保存在: {os.path.abspath(save_dir)}")

    model.train()
    for epoch in range(start_epoch, args.epochs):
        for step, batch in enumerate(loader, 1):
            metrics = train_one_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                alpha_bars=alpha_bars,
                num_timesteps=args.timesteps,
                device=device,
            )
            print(
                f"[epoch {epoch + 1}/{args.epochs} step {step}/{len(loader)}] "
                f"loss={metrics['loss']:.6f} "
                f"noise={metrics['loss_noise']:.6f}"
            )

        # ================= 保存完整的 checkpoint =================
        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pth")
        save_checkpoint(model, optimizer, epoch, checkpoint_path, args)

        # 同时保存独立的模型权重文件（保持兼容性）
        model_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth")
        state_dict_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save({"state_dict": state_dict_to_save}, model_path)
        print(f"💾 Epoch {epoch + 1} 权重已成功保存至: {model_path}")
        print("-" * 60)


if __name__ == "__main__":
    main()