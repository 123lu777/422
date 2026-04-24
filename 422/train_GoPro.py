import os
import torch

torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import numpy as np

import utils
from data.data_RGB import get_training_data, get_validation_data
from models.MISCFilterNet_WindTurbine import MISCKernelNet as myNet
from loss import losses
from warmup_scheduler import GradualWarmupScheduler
from tools.get_parameter_number import get_parameter_number
import kornia
import argparse

# ================= DDP 分布式训练库 =================
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
start_epoch = 1

parser = argparse.ArgumentParser(description='Image Deblurring')
parser.add_argument('--train_dir', default='./dataset/GOPRO_Large', type=str)
parser.add_argument('--train_meta', default='./dataset/GOPRO_Large/GOPRO_train_list.txt', type=str)
parser.add_argument('--val_dir', default='./dataset/GOPRO_Large', type=str)
parser.add_argument('--val_meta', default='./dataset/GOPRO_Large/GOPRO_test_list.txt', type=str)
parser.add_argument('--model_save_dir', default='./checkpoints', type=str)
parser.add_argument('--dataset', default='GoPro', type=str)
parser.add_argument('--session', default='MISCFilter_GoPro', type=str)
parser.add_argument('--patch_size', default=256, type=int)
parser.add_argument('--num_epochs', default=5000, type=int)
parser.add_argument('--batch_size', default=16, type=int, help='总 Batch Size (将自动平分给两张卡)')
parser.add_argument('--val_epochs', default=10, type=int, help='每隔多少轮计算一次 PSNR/SSIM')
parser.add_argument('--print_epochs', default=1, type=int)
parser.add_argument('--pretrain_weights', default='', type=str)
parser.add_argument('--local_rank', default=-1, type=int)
args = parser.parse_args()

# ================= DDP 环境初始化 =================
local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')
device = torch.device('cuda', local_rank)
is_master = (local_rank == 0)
# =================================================

dataset = args.dataset
session = args.session
patch_size = args.patch_size

model_dir = os.path.join(args.model_save_dir, dataset, session)
log_dir = os.path.join(args.model_save_dir, dataset, session, 'log.txt')
if is_master:
    utils.mkdir(model_dir)

train_dir = args.train_dir
val_dir = args.val_dir
train_meta = args.train_meta
val_meta = args.val_meta

num_epochs = args.num_epochs
world_size = dist.get_world_size()
per_gpu_batch_size = max(1, args.batch_size // world_size)

val_epochs = args.val_epochs
print_epochs = args.print_epochs
start_lr = 2e-4
end_lr = 1e-6

######### Model ###########
model_restoration = myNet().to(device)

if is_master:
    total_num, trainable_num = get_parameter_number(model_restoration)
    with open(log_dir, "a+", encoding="utf-8") as f:
        f.write('Total: {}\n'.format(total_num))
        f.write('Trainable: {}\n'.format(trainable_num))

# 【关键修复】先加载纯净版权重，然后再用 DDP 包裹！
model_pre_dir = args.pretrain_weights.strip()
if model_pre_dir:
    utils.load_checkpoint(model_restoration, model_pre_dir)
    try:
        start_epoch = utils.load_start_epoch(model_pre_dir) + 1
    except:
        start_epoch = 1
    if is_master:
        print(f"\n==> 成功加载权重: {model_pre_dir}")
        print(f"==> 将无缝从 Epoch {start_epoch} 继续训练!\n")

model_restoration = DDP(model_restoration, device_ids=[local_rank], output_device=local_rank,
                        find_unused_parameters=False)

optimizer = optim.Adam(model_restoration.parameters(), lr=start_lr, betas=(0.9, 0.999), eps=1e-8)

if model_pre_dir:
    try:
        utils.load_optim(optimizer, model_pre_dir)
    except:
        pass

######### Scheduler ###########
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs - warmup_epochs, eta_min=end_lr)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

# 恢复优化器的学习率退火进度
if start_epoch > 1:
    for i in range(1, start_epoch):
        scheduler.step()

######### Loss ###########
criterion_char = losses.CharbonnierLoss().to(device)
criterion_edge = losses.EdgeLoss().to(device)
criterion_fft = losses.fftLoss().to(device)

######### DataLoaders ###########
train_dataset = get_training_data(train_dir, train_meta, {'patch_size': patch_size})
train_sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(dataset=train_dataset, batch_size=per_gpu_batch_size, sampler=train_sampler, num_workers=4,
                          drop_last=True, pin_memory=True)

val_dataset = get_validation_data(val_dir, val_meta, {'patch_size': patch_size})
val_sampler = DistributedSampler(val_dataset, shuffle=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=per_gpu_batch_size, sampler=val_sampler, num_workers=4,
                        drop_last=False, pin_memory=True)

# 尝试获取历史 best_psnr
best_psnr = 0
best_epoch = 0
try:
    if is_master and os.path.exists(os.path.join(model_dir, "model_best.pth")):
        chk = torch.load(os.path.join(model_dir, "model_best.pth"), map_location='cpu')
        best_epoch = chk.get('epoch', 0)
        best_psnr = 25.4367  # 根据您的日志手动兜底
except:
    pass

for epoch in range(start_epoch, num_epochs + 1):
    train_sampler.set_epoch(epoch)

    epoch_start_time = time.time()
    epoch_loss = 0
    iter = 0

    model_restoration.train()
    for i, data in enumerate(train_loader, 0):
        optimizer.zero_grad()

        target_ = data[0].to(device)
        input_ = data[1].to(device)
        target = kornia.geometry.transform.build_pyramid(target_, 3)
        restored, restored_inter = model_restoration(input_)

        loss_fft = criterion_fft(restored[0], target[0]) + criterion_fft(restored[1], target[1]) + criterion_fft(
            restored[2], target[2])
        loss_char = criterion_char(restored[0], target[0]) + criterion_char(restored[1], target[1]) + criterion_char(
            restored[2], target[2])
        loss_edge = criterion_edge(restored[0], target[0]) + criterion_edge(restored[1], target[1]) + criterion_edge(
            restored[2], target[2])
        loss_char_inter = criterion_char(restored_inter[0], target[0]) + criterion_char(restored_inter[1],
                                                                                        target[1]) + criterion_char(
            restored_inter[2], target[2])

        loss = loss_char + loss_char_inter + 0.01 * loss_fft + 0.05 * loss_edge
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model_restoration.parameters(), max_norm=0.01)
        optimizer.step()

        epoch_loss += loss.item()
        iter += 1

        if is_master and iter % 20 == 0:
            print(f'Epoch [{epoch}] Iter [{iter}] Total Loss: {loss.item():.4f}')

    scheduler.step()

    # ================= 强制每轮保存当前权重 =================
    if is_master:
        torch.save({
            'epoch': epoch,
            'state_dict': model_restoration.module.state_dict(),
            'optimizer': optimizer.state_dict()
        }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))

        torch.save({
            'epoch': epoch,
            'state_dict': model_restoration.module.state_dict(),
            'optimizer': optimizer.state_dict()
        }, os.path.join(model_dir, "model_latest.pth"))

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                                  epoch_loss / iter,
                                                                                  scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")

    #### Evaluation (每 10 轮验证一次 PSNR 和 SSIM) ####
    if epoch % val_epochs == 0:
        model_restoration.eval()
        psnr_val_rgb = []
        ssim_val_rgb = []

        for ii, data_val in enumerate(val_loader, 0):
            target = data_val[0].to(device)
            input_ = data_val[1].to(device)

            with torch.no_grad():
                restored, _ = model_restoration(input_)

            # 遍历 Batch 计算指标
            for res, tar in zip(restored[0], target):
                psnr_val_rgb.append(utils.torchPSNR(res, tar))

                res_c = torch.clamp(res, 0.0, 1.0).unsqueeze(0)
                tar_c = torch.clamp(tar, 0.0, 1.0).unsqueeze(0)
                ssim_score = kornia.metrics.ssim(res_c, tar_c, window_size=11, max_val=1.0).mean()
                ssim_val_rgb.append(ssim_score)

        local_psnr = torch.stack(psnr_val_rgb).mean()
        local_ssim = torch.stack(ssim_val_rgb).mean()

        dist.reduce(local_psnr, dst=0, op=dist.ReduceOp.AVG)
        dist.reduce(local_ssim, dst=0, op=dist.ReduceOp.AVG)

        if is_master:
            avg_psnr = local_psnr.item()
            avg_ssim = local_ssim.item()

            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                best_epoch = epoch
                torch.save({'epoch': epoch, 'state_dict': model_restoration.module.state_dict(),
                            'optimizer': optimizer.state_dict()}, os.path.join(model_dir, "model_best.pth"))

            print(
                f"[验证阶段] Epoch {epoch} | PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} --- [当前最佳] Epoch {best_epoch} Best_PSNR {best_psnr:.4f}")
            with open(log_dir, "a+", encoding="utf-8") as f:
                f.write(
                    f"[验证阶段] Epoch {epoch} | PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} --- [当前最佳] Epoch {best_epoch} Best_PSNR {best_psnr:.4f}\n")