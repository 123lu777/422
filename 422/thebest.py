import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import glob
import torch.multiprocessing as mp
from tqdm import tqdm

# 请确保您的模型路径正确
from models.MISCFilterNet_WindTurbine import MISCKernelNet as myNet


# ================= 自定义 PSNR 和 SSIM 计算函数 =================
def calculate_psnr(img1, img2):
    """纯 numpy 实现的 PSNR"""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim_channel(img1, img2):
    """单通道 SSIM 计算"""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    """支持 RGB 三通道的 SSIM 计算"""
    if len(img1.shape) == 3:
        return np.mean([calculate_ssim_channel(img1[:, :, i], img2[:, :, i]) for i in range(img1.shape[2])])
    else:
        return calculate_ssim_channel(img1, img2)


# ===============================================================

def test_weights_worker(gpu_id, weights_list, ref_img_path, blur_img_path, return_dict):
    """
    单个 GPU 的工作进程
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 读取并预处理图片
    ref_img = cv2.imread(ref_img_path)
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

    blur_img = cv2.imread(blur_img_path)
    blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)

    if ref_img.shape != blur_img.shape:
        ref_img = cv2.resize(ref_img, (blur_img.shape[1], blur_img.shape[0]))

    input_ = torch.from_numpy(np.float32(blur_img / 255.)).permute(2, 0, 1).unsqueeze(0).cuda()
    _, _, Hx, Wx = input_.shape

    pad_h = (8 - Hx % 8) % 8
    pad_w = (8 - Wx % 8) % 8
    input_padded = F.pad(input_, (0, pad_w, 0, pad_h), mode='reflect')

    # 初始化模型
    model_restoration = myNet(inference=False)
    model_restoration.cuda()
    model_restoration.eval()

    results = []

    with torch.no_grad():
        for weight_path in tqdm(weights_list, desc=f"GPU {gpu_id} 进度", position=gpu_id, leave=True):
            try:
                # 1. 加载权重
                checkpoint = torch.load(weight_path, map_location='cuda')
                state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

                new_state_dict = {}
                for k, v in state_dict.items():
                    new_key = k.replace('module.', '') if k.startswith('module.') else k
                    new_state_dict[new_key] = v

                model_restoration.load_state_dict(new_state_dict, strict=True)

                # 2. 推理
                restored_list, _ = model_restoration(input_padded)
                restored_padded = restored_list[0]

                # 3. 恢复尺寸 & 后处理
                restored = restored_padded[:, :, :Hx, :Wx]
                restored = torch.clamp(restored, 0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
                restored_img = (restored * 255.0).astype(np.uint8)

                # 4. 使用自定义函数计算 PSNR 和 SSIM
                psnr_val = calculate_psnr(ref_img, restored_img)
                ssim_val = calculate_ssim(ref_img, restored_img)

                results.append({
                    "weight_name": os.path.basename(weight_path),
                    "psnr": psnr_val,
                    "ssim": ssim_val
                })

            except Exception as e:
                print(f"\n[GPU {gpu_id} Error] 测试权重 {os.path.basename(weight_path)} 出错: {e}")

    return_dict[gpu_id] = results


def main():
    # 配置路径
    image1_path = "/media/JYJ/新加卷/ZJL/1015/1_fusion_a0.20.jpg"  # 参考图（清晰）
    image2_path = "/media/JYJ/新加卷/ZJL/MOHU2/1.jpg"  # 输入图（模糊）
    weights_dir = "/media/JYJ/新加卷/ZJL/420/checkpoints/GoPro/GoPro_Test_VSS/"  # 权重所在文件夹

    weight_files = glob.glob(os.path.join(weights_dir, "*.pth"))
    if not weight_files:
        raise FileNotFoundError(f"在 {weights_dir} 中没有找到任何 .pth 文件！")

    print(f"===> 总共找到 {len(weight_files)} 个权重文件。准备启动双卡并行测试...")

    # 分配任务
    weights_gpu0 = weight_files[0::2]
    weights_gpu1 = weight_files[1::2]

    # 启动多进程
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    return_dict = manager.dict()

    p0 = mp.Process(target=test_weights_worker, args=(0, weights_gpu0, image1_path, image2_path, return_dict))
    p1 = mp.Process(target=test_weights_worker, args=(1, weights_gpu1, image1_path, image2_path, return_dict))

    p0.start()
    p1.start()

    p0.join()
    p1.join()

    # 汇总与排序
    all_results = []
    if 0 in return_dict:
        all_results.extend(return_dict[0])
    if 1 in return_dict:
        all_results.extend(return_dict[1])

    if not all_results:
        print("没有任何权重测试成功。")
        return

    results_sorted = sorted(all_results, key=lambda x: x["psnr"], reverse=True)

    print("\n" + "=" * 50)
    print("🏆 表现最接近参考图的前 10 个权重 (按 PSNR 排序):")
    print("=" * 50)
    print(f"{'排名':<4} | {'权重文件名':<35} | {'PSNR':<8} | {'SSIM':<8}")
    print("-" * 65)

    top_k = min(10, len(results_sorted))
    for i in range(top_k):
        res = results_sorted[i]
        print(f"{i + 1:<4} | {res['weight_name']:<35} | {res['psnr']:.4f}  | {res['ssim']:.4f}")


if __name__ == '__main__':
    main()