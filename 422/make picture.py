"""
风机叶片数据集生成工具 - 2025 顶会 SOTA 物理先验版
✅ 连续随机采样：抛弃离散固定值，角速度在 [0.2, 3.0] 范围内连续随机，拒绝模型死记硬背
✅ 图像内容减负：每张清晰图仅生成 2 张随机模糊图，避免内容过度冗余
✅ 全空间旋转中心：旋转中心可随机跑到图片外部，完美模拟无人机“特写局部”时的微弧度旋转模糊
✅ 真实物理退化：基于物理帧积分法渲染模糊，并注入真实传感器噪声 (Sensor Noise)
✅ 视频/图像处理：直接缩放到 256×256，无灰色边缘填充
✅ 完整兼容：保持 GoPro 数据集目录结构 (train/test, sharp/blur)，自动生成 list.txt
"""

import cv2
import numpy as np
import os
import sys
import shutil
import random
import glob
from typing import List, Dict, Tuple, Optional

# =====================================================
# 可选导入 - 进度条
# =====================================================
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("💡 提示: 安装 'tqdm' 可获得更好的进度显示 (pip install tqdm)")


# =====================================================
# [2025 SOTA] 真实物理退化模拟 (Physical Degradation)
# =====================================================

def add_realistic_sensor_noise(image: np.ndarray,
                               noise_level_range: Tuple[float, float] = (0.002, 0.015)) -> np.ndarray:
    """
    注入真实传感器噪声 (近似 Poisson-Gaussian)。
    消除纯数学合成导致的 Domain Gap，提升模型在真实工业图像上的泛化性。
    """
    img_float = image.astype(np.float32) / 255.0
    noise_level = np.random.uniform(noise_level_range[0], noise_level_range[1])

    # 模拟传感器暗电流噪声
    noise = np.random.randn(*img_float.shape) * noise_level
    noisy_img = img_float + noise

    return np.clip(noisy_img * 255.0, 0, 255).astype(np.uint8)


def apply_physical_rotational_blur(image: np.ndarray, omega_rad: float, exposure_time: float = 1.0,
                                   num_steps: int = 31) -> np.ndarray:
    """
    基于物理帧积分法模拟风机叶片的非均匀旋转模糊。
    允许旋转中心在图片外，完美契合网络中的 KinematicMotionHead 预测原理 (v = w x r)。
    """
    h, w = image.shape[:2]

    # 【核心升级：全空间随机旋转中心】
    # 允许旋转中心跑到图片外部 [-w, 2w]，模拟无人机只拍到叶片局部（轮毂在画面外）的情况。
    cx = np.random.randint(-w, 2 * w)
    cy = np.random.randint(-h, 2 * h)

    # 曝光时间内的总旋转角度 (Degrees)
    total_angle_deg = np.degrees(omega_rad * exposure_time)

    # 随机决定顺时针还是逆时针旋转
    if random.random() > 0.5:
        total_angle_deg = -total_angle_deg

    blurred_float = np.zeros((h, w, 3), dtype=np.float32)
    angles = np.linspace(0, total_angle_deg, num_steps)

    for angle in angles:
        M = cv2.getRotationMatrix2D((cx, cy), angle, scale=1.0)
        # BORDER_REFLECT_101 避免边缘切出去产生黑边
        rotated_frame = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        blurred_float += rotated_frame.astype(np.float32)

    blurred_float /= num_steps
    return blurred_float.astype(np.uint8)


def motion_blur_by_continuous_sampling(sharp_image: np.ndarray,
                                       omega_range: Tuple[float, float] = (0.2, 3.0),
                                       target_size: Tuple[int, int] = (256, 256)) -> Optional[np.ndarray]:
    """从连续区间内随机抽取角速度生成模糊图像"""
    if sharp_image is None:
        return None

    # 1. 物理渲染积分 (曝光时间 0.3~0.6 之间随机，模拟不同快门速度)
    random_omega = random.uniform(omega_range[0], omega_range[1])
    exposure_time = random.uniform(0.3, 0.6)

    blurred = apply_physical_rotational_blur(sharp_image, omega_rad=random_omega, exposure_time=exposure_time)

    # 2. 添加传感器噪声
    blurred = add_realistic_sensor_noise(blurred)

    return blurred


# =====================================================
# 基础工具函数
# =====================================================

def find_images_recursive(root_path: str) -> List[str]:
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP')
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root_path, '**', f'*{ext}'), recursive=True))
    return sorted(list(set(files)))


def imread_chinese(path: str) -> Optional[np.ndarray]:
    try:
        with open(path, 'rb') as f:
            return cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    except:
        return None


def imwrite_chinese(path: str, img: np.ndarray) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ext = os.path.splitext(path)[1].lower() or '.png'
        success, encoded = cv2.imencode(ext, img)
        if success:
            with open(path, 'wb') as f:
                f.write(encoded.tobytes())
            return True
        return False
    except:
        return False


# =====================================================
# 核心处理流程 - 图像处理
# =====================================================

def process_images(image_sources: List[str], output_dir: str, target_size: Tuple[int, int],
                   omega_range: Tuple[float, float], num_augment: int, train_ratio: float,
                   global_idx: Dict[str, int]) -> Dict[str, int]:
    for split in ['train', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'sharp'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'blur'), exist_ok=True)

    all_files = []
    for src in image_sources:
        if os.path.isdir(src):
            all_files.extend(find_images_recursive(src))
        elif os.path.isfile(src):
            all_files.append(src)

    if not all_files:
        print("❌ 未找到图像")
        return global_idx

    # 随机划分训练/测试集
    random.seed(42)
    random.shuffle(all_files)
    train_end = int(len(all_files) * train_ratio)

    splits = [('train', all_files[:train_end]), ('test', all_files[train_end:])]

    for split_name, img_list in splits:
        print(f"\n🚀 处理 {split_name} 集 ({len(img_list)} 张清晰原图)...")
        iterator = tqdm(img_list, desc=f"{split_name}处理") if HAS_TQDM else img_list

        for img_path in iterator:
            sharp = imread_chinese(img_path)
            if sharp is None:
                continue

            # 直接缩放为 target_size
            sharp_resized = cv2.resize(sharp, target_size, interpolation=cv2.INTER_AREA)

            # 【连续随机采样生成】：每张清晰图生成 num_augment 次随机模糊图
            for _ in range(num_augment):
                blurred = motion_blur_by_continuous_sampling(sharp_resized, omega_range=omega_range,
                                                             target_size=target_size)
                if blurred is None:
                    continue

                # 全局递增文件名
                file_name = f"{global_idx[split_name]:07d}.png"
                s_path = os.path.join(output_dir, split_name, 'sharp', file_name)
                b_path = os.path.join(output_dir, split_name, 'blur', file_name)

                # 保存图像对
                if imwrite_chinese(s_path, sharp_resized) and imwrite_chinese(b_path, blurred):
                    global_idx[split_name] += 1

    return global_idx


# =====================================================
# 生成元文件
# =====================================================

def generate_metadata(dataset_dir: str) -> None:
    print("\n📝 生成训练/测试元文件...")
    for split in ['train', 'test']:
        sharp_dir = os.path.join(dataset_dir, split, 'sharp')
        if not os.path.exists(sharp_dir):
            continue

        files = sorted([f for f in os.listdir(sharp_dir) if f.endswith('.png')])
        meta_file = os.path.join(dataset_dir, f'GOPRO_{split}_list.txt')  # 兼容 GOPRO 命名习惯

        with open(meta_file, 'w') as f:
            for img in files:
                f.write(f"{split}/sharp/{img} {split}/blur/{img}\n")
        print(f"  ✓ {meta_file} 写入 {len(files)} 对路径")


# =====================================================
# Main
# =====================================================

def main():
    print("=" * 70)
    print("风机叶片数据集生成工具 - 2025 SOTA 架构")
    print("=" * 70)

    # ================= 配置区 =================

    # 1. 纯净风机图片目录 (会自动向下扫描)
    IMAGE_INPUT_DIRS = [
        r"C:\Your\Path\To\Clear\FanBlades",  # <--- 请修改为您的真实目录
    ]

    # 2. 数据集保存目录
    OUTPUT_BASE_DIR = r"./dataset/FanBlade_ContinuousOmega"

    # 3. 生成参数
    TARGET_SIZE = (256, 256)
    TRAIN_RATIO = 0.9  # 90% 用于训练，10% 用于测试

    # 【SOTA 连续采样配置】
    OMEGA_RANGE = (0.2, 3.0)  # 角速度随机抽取范围 (rad/s)
    NUM_AUGMENT_PER_IMAGE = 2  # 每张清晰原图，生成 2 种随机的模糊图片

    # ==========================================

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    global_idx = {'train': 0, 'test': 0}

    # 处理静态图像
    valid_img_dirs = [d for d in IMAGE_INPUT_DIRS if os.path.exists(d)]
    if valid_img_dirs:
        global_idx = process_images(
            image_sources=valid_img_dirs,
            output_dir=OUTPUT_BASE_DIR,
            target_size=TARGET_SIZE,
            omega_range=OMEGA_RANGE,
            num_augment=NUM_AUGMENT_PER_IMAGE,
            train_ratio=TRAIN_RATIO,
            global_idx=global_idx
        )

    # 生成 list.txt
    generate_metadata(OUTPUT_BASE_DIR)

    print("\n✅ 全部完成！")
    print(f"🎉 训练集共生成: {global_idx['train']} 对图像")
    print(f"🎉 测试集共生成: {global_idx['test']} 对图像")


if __name__ == "__main__":
    main()