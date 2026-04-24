import argparse
import os
import glob
from tqdm import tqdm

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models.dual_kg_diffusion import DualKGDiffusionModel


# ================= 辅助函数 =================

def normalize_to_neg_one_to_one(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


def unnormalize_to_zero_to_one(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)


def build_linear_beta_schedule(
        timesteps: int,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device = torch.device("cpu"),
):
    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def pad_img(img_tensor: torch.Tensor, pad_size: int = 16):
    """确保长宽是 pad_size 的整数倍，防止 U-Net 下采样时维度不匹配"""
    _, _, h, w = img_tensor.shape
    pad_h = (pad_size - h % pad_size) % pad_size
    pad_w = (pad_size - w % pad_size) % pad_size
    if pad_h == 0 and pad_w == 0:
        return img_tensor, (0, 0, 0, 0)
    # F.pad format: (left, right, top, bottom)
    padding = (0, pad_w, 0, pad_h)
    padded_img = F.pad(img_tensor, padding, mode='reflect')
    return padded_img, padding


# ================= DDPM 采样核心 =================

@torch.no_grad()
def sample_ddpm(
        model: DualKGDiffusionModel,
        blur_img: torch.Tensor,
        timesteps: int,
        device: torch.device
) -> torch.Tensor:
    b, c, h, w = blur_img.shape

    # 1. 提取物理专家先验 (只需提取一次，大大加速推理)
    base_img, flows = model.prior_extractor(blur_img)
    blur_cond = normalize_to_neg_one_to_one(blur_img)
    base_img_cond = normalize_to_neg_one_to_one(base_img)
    cond_img = torch.cat([blur_cond, base_img_cond], dim=1)

    # 2. 准备 DDPM 参数
    betas, alphas, alpha_bars = build_linear_beta_schedule(timesteps, device=device)

    # 3. 从纯高斯噪声开始
    x_t = torch.randn_like(blur_cond)

    # 4. 逐步去噪循环 (T-1 down to 0)
    print("开始 DDPM 去噪采样...")
    for i in tqdm(reversed(range(timesteps)), total=timesteps, desc="DDPM Sampling"):
        t_tensor = torch.full((b,), i, device=device, dtype=torch.long)

        # U-Net 预测噪声
        pred_noise = model.denoiser(x_t, t_tensor, cond_img, flows)

        alpha_t = alphas[i]
        alpha_bar_t = alpha_bars[i]
        beta_t = betas[i]

        # 计算均值 (mu)
        mu = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise)

        if i > 0:
            # 如果没到最后一步，加上随机方差
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt(beta_t)
            x_t = mu + sigma * noise
        else:
            # 最后一步直接输出均值
            x_t = mu

    # 5. 反归一化回到 [0, 1]
    final_img = unnormalize_to_zero_to_one(x_t)
    return final_img


# ================= 主函数 =================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--general_weights", type=str, required=True, help="通用专家权重")
    parser.add_argument("--kinematic_weights", type=str, required=True, help="旋转专家权重")
    parser.add_argument("--checkpoint", type=str, required=True, help="我们训练出来的 pth 权重")
    parser.add_argument("--input_dir", type=str, required=True, help="测试模糊图像文件夹")
    parser.add_argument("--output_dir", type=str, default="./results", help="去模糊结果保存文件夹")
    parser.add_argument("--timesteps", type=int, default=1000, help="DDPM 步数")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=> 正在初始化模型...")
    model = DualKGDiffusionModel(
        general_weights_path=args.general_weights,
        kinematic_weights_path=args.kinematic_weights,
        use_deform_in_feat=True,
        use_deform_in_encoder=True,
        unet_base_channels=64,
        device=device,
    ).to(device)

    print(f"=> 正在加载训练好的权重: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 寻找输入文件夹下所有图片
    exts = ('*.jpg', '*.jpeg', '*.png')
    img_paths = []
    for ext in exts:
        img_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
        img_paths.extend(glob.glob(os.path.join(args.input_dir, ext.upper())))

    print(f"=> 找到 {len(img_paths)} 张测试图片")

    for idx, img_path in enumerate(sorted(img_paths)):
        img_name = os.path.basename(img_path)
        print(f"\n[{idx + 1}/{len(img_paths)}] 正在处理: {img_name}")

        # 读取并预处理
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 为了测试快一点，如果在验证阶段可以 resize 到 256x256
        # img = cv2.resize(img, (256, 256))

        blur_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        blur_tensor = blur_tensor.unsqueeze(0).to(device)

        # 尺寸对齐 (Padding 到 16 的倍数)
        blur_tensor_pad, padding = pad_img(blur_tensor, pad_size=16)

        # 核心：DDPM 生成
        out_tensor_pad = sample_ddpm(model, blur_tensor_pad, args.timesteps, device)

        # 去除 Padding
        _, _, h, w = blur_tensor.shape
        out_tensor = out_tensor_pad[:, :, :h, :w]

        # 转换为 numpy 并保存
        out_img = out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_img = (out_img * 255.0).clip(0, 255).astype(np.uint8)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

        save_path = os.path.join(args.output_dir, img_name)
        cv2.imwrite(save_path, out_img)
        print(f"✅ 已保存至: {save_path}")


if __name__ == "__main__":
    main()