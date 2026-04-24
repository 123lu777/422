import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import *
import models.MISCKernel_cuda as misckernel


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class LightVSSBlock(nn.Module):
    def __init__(self, channels):
        super(LightVSSBlock, self).__init__()
        self.norm = LayerNorm2d(channels)
        self.in_proj = nn.Conv2d(channels, channels * 2, kernel_size=1, stride=1, padding=0, bias=True)

        self.strip_h = nn.Conv2d(channels, channels, kernel_size=(1, 9), stride=1, padding=(0, 4), groups=channels,
                                 bias=True)
        self.strip_v = nn.Conv2d(channels, channels, kernel_size=(9, 1), stride=1, padding=(4, 0), groups=channels,
                                 bias=True)

        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.act = nn.SiLU(inplace=False)

        # 【稳定锁1】零初始化输出投影，使得初始阶段该模块表现为严格的 Identity 映射，防止初期梯度爆炸
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        identity = x
        x = self.norm(x)
        x = self.in_proj(x)

        x_state, x_gate = torch.chunk(x, 2, dim=1)

        x_state = self.act(x_state)
        x_state = self.strip_h(x_state)
        x_state = self.strip_v(x_state)

        x = x_state * torch.sigmoid(x_gate)
        x = self.out_proj(x)
        return x + identity


class KinematicMotionHead(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, max_omega_prior=2.0):
        super(KinematicMotionHead, self).__init__()
        self.max_omega_prior = max_omega_prior

        self.translation_branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.SiLU(inplace=False),
            nn.Conv2d(hidden_channels, 2, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.rotation_branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.SiLU(inplace=False),
            nn.Conv2d(hidden_channels, 3, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        # 【稳定锁2】物理头零初始化：保证在 Epoch 1 时输出的流场绝对为 0，杜绝底层 C++ 算子非法内存访问！
        nn.init.zeros_(self.translation_branch[-1].weight)
        nn.init.zeros_(self.translation_branch[-1].bias)
        nn.init.zeros_(self.rotation_branch[-1].weight)
        nn.init.zeros_(self.rotation_branch[-1].bias)

    def _meshgrid(self, h, w, device, dtype):
        ys = torch.arange(0, h, device=device, dtype=dtype)
        xs = torch.arange(0, w, device=device, dtype=dtype)
        if 'indexing' in torch.meshgrid.__code__.co_varnames:
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        else:
            grid_y, grid_x = torch.meshgrid(ys, xs)
        return grid_x, grid_y

    def forward(self, feat):
        b, _, h, w = feat.size()

        trans = self.translation_branch(feat)
        trans = torch.tanh(self.pool(trans)) * float(max(h, w))

        rot = self.rotation_branch(feat)
        rot = self.pool(rot)
        omega = torch.tanh(rot[:, 0:1, :, :]) * self.max_omega_prior

        cx_offset = torch.tanh(rot[:, 1:2, :, :]) * float(w)
        cy_offset = torch.tanh(rot[:, 2:3, :, :]) * float(h)

        cx = (float(w - 1) * 0.5) + cx_offset
        cy = (float(h - 1) * 0.5) + cy_offset

        grid_x, grid_y = self._meshgrid(h, w, feat.device, feat.dtype)
        grid_x = grid_x.view(1, 1, h, w)
        grid_y = grid_y.view(1, 1, h, w)

        rel_x = grid_x - cx
        rel_y = grid_y - cy

        rot_vx = -omega * rel_y
        rot_vy = omega * rel_x
        rot_flow = torch.cat([rot_vx, rot_vy], dim=1)

        trans_flow = trans.expand(b, 2, h, w)
        flow = trans_flow + rot_flow

        # 严格钳制流场
        flow = torch.clamp(flow, min=-256.0, max=256.0)
        return flow


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8, ResBlock=ResBlock):
        super(EBlock, self).__init__()
        layers = [ResBlock(out_channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8, ResBlock=ResBlock):
        super(DBlock, self).__init__()
        layers = [ResBlock(channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel, BasicConv=BasicConv):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class SCM(nn.Module):
    def __init__(self, out_plane, BasicConv=BasicConv, inchannel=3):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(inchannel, out_plane // 4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane - inchannel, kernel_size=1, stride=1, relu=True)
        )
        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel, BasicConv=BasicConv):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out


def CharbonnierFunc(data, epsilon=0.001):
    return torch.mean(torch.sqrt(data ** 2 + epsilon ** 2))


def flow_warp(x, flow, interpolation='bilinear', padding_mode='zeros', align_corners=True):
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(
            f'The spatial sizes of input ({x.size()[-2:]}) and flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    device = flow.device

    if 'indexing' in torch.meshgrid.__code__.co_varnames:
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h, device=device, dtype=x.dtype),
                                        torch.arange(0, w, device=device, dtype=x.dtype), indexing='ij')
    else:
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h, device=device, dtype=x.dtype),
                                        torch.arange(0, w, device=device, dtype=x.dtype))

    grid = torch.stack((grid_x, grid_y), 2)
    grid.requires_grad = False

    grid_flow = grid + flow
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    grid_flow = grid_flow.type(x.type())
    output = F.grid_sample(x, grid_flow, mode=interpolation, padding_mode=padding_mode, align_corners=align_corners)
    return output


class MISCKernelNet(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=32, num_blocks=[12, 12, 12], num_blocks_kernel=[1, 1, 1],
                 kernel_size=7, inference=False):
        super(MISCKernelNet, self).__init__()
        self.inference = inference
        self.dim = dim
        self.kernel_size = kernel_size
        self.kernel_pad = int((self.kernel_size - 1) / 2.0)

        if not inference:
            BasicConv = BasicConv_do
        else:
            BasicConv = BasicConv_do_eval

        ResBlock = LightVSSBlock
        base_channel = dim

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_blocks[0], ResBlock=ResBlock),
            EBlock(base_channel * 2, num_blocks[1], ResBlock=ResBlock),
            EBlock(base_channel * 4, num_blocks[2], ResBlock=ResBlock),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(inp_channels, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 4 * 2, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 2 * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_blocks[2], ResBlock=ResBlock),
            DBlock(base_channel * 2, num_blocks[1], ResBlock=ResBlock),
            DBlock(base_channel, num_blocks[0], ResBlock=ResBlock)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel * 1, BasicConv=BasicConv),
            AFF(base_channel * 7, base_channel * 2, BasicConv=BasicConv)
        ])

        self.FAM1 = FAM(base_channel * 4, BasicConv=BasicConv)
        self.SCM1 = SCM(base_channel * 4, BasicConv=BasicConv)
        self.FAM2 = FAM(base_channel * 2, BasicConv=BasicConv)
        self.SCM2 = SCM(base_channel * 2, BasicConv=BasicConv)

        self.softmax = nn.Softmax(1)
        self.modulePad = torch.nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad])
        self.moduleKernel = misckernel.FunctionKernel.apply

        self.KernelPredictFlow = nn.ModuleList([
            BasicConv(base_channel * 4, 2, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, 2, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel, 2, kernel_size=3, relu=False, stride=1),
        ])

        self.KinematicHeads = nn.ModuleList([
            KinematicMotionHead(base_channel * 4, hidden_channels=max(16, base_channel // 2)),
            KinematicMotionHead(base_channel * 2, hidden_channels=max(16, base_channel // 2)),
            KinematicMotionHead(base_channel, hidden_channels=max(16, base_channel // 2)),
        ])

        self.flowup = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.KernelPredictFlowMask = nn.ModuleList([
            BasicConv(base_channel * 4, 1, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, 1, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel, 1, kernel_size=3, relu=False, stride=1),
        ])
        self.sigmoid = nn.Sigmoid()

        self.KernelOutBias = nn.ModuleList([
            BasicConv(base_channel * 4, out_channels, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, out_channels, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel, out_channels, kernel_size=3, relu=False, stride=1),
        ])

        self.KernelOutWeight = nn.ModuleList([
            BasicConv(base_channel * 4 * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2 * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
        ])

        self.KernelOutkernelx = nn.ModuleList([
            BasicConv(base_channel * 4 * 2, kernel_size, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2 * 2, kernel_size, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, kernel_size, kernel_size=3, relu=False, stride=1),
        ])

        self.KernelOutkernely = nn.ModuleList([
            BasicConv(base_channel * 4 * 2, kernel_size, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2 * 2, kernel_size, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, kernel_size, kernel_size=3, relu=False, stride=1),
        ])

        self.KernelOutAlpha = nn.ModuleList([
            BasicConv(base_channel * 4 * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2 * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
        ])

        self.KernelOutBeta = nn.ModuleList([
            BasicConv(base_channel * 4 * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2 * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, kernel_size ** 2, kernel_size=3, relu=False, stride=1),
        ])

    def safe_module_kernel(self, pad_x, posx, posy, alpha, beta, weight):
        torch.cuda.synchronize(device=pad_x.device)
        out = self.moduleKernel(
            pad_x.contiguous(),
            posx.contiguous(),
            posy.contiguous(),
            alpha.contiguous(),
            beta.contiguous(),
            weight.contiguous()
        )
        torch.cuda.synchronize(device=pad_x.device)
        return out

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)

        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs_fil = list()
        outputs = list()
        Kernal_Loss = 0

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        z = self.Decoder[0](z)

        # ---------------- Scale 1/4 ----------------
        s3_kernal_flow_data = self.KernelPredictFlow[0](z)
        s3_kernal_flow_prior = self.KinematicHeads[0](z)
        s3_kernal_flow = s3_kernal_flow_data + s3_kernal_flow_prior

        s3_kernal_flowmask = self.KernelPredictFlowMask[0](z)
        s3_kernal_flowmask = self.sigmoid(s3_kernal_flowmask)

        zx4 = torch.cat([z, x_4], 1)
        s3_kernal_flowfeat0, x_4_0 = torch.split(flow_warp(zx4, s3_kernal_flow.permute(0, 2, 3, 1)), self.dim * 4,
                                                 dim=1)
        s3_kernal_flowfeat1, x_4_1 = torch.split(flow_warp(zx4, -s3_kernal_flow.permute(0, 2, 3, 1)), self.dim * 4,
                                                 dim=1)
        x_4 = x_4_0 * s3_kernal_flowmask + x_4_1 * (1 - s3_kernal_flowmask)

        s3_kernal_bias = self.KernelOutBias[0](z)

        z = torch.cat([z, s3_kernal_flowfeat0 * s3_kernal_flowmask + s3_kernal_flowfeat1 * (1 - s3_kernal_flowmask)], 1)
        s3_kernal_weight = self.KernelOutWeight[0](z)
        s3_kernal_weight = self.softmax(s3_kernal_weight)
        s3_kernal_alpha = self.KernelOutAlpha[0](z)
        s3_kernal_beta = self.KernelOutBeta[0](z)

        # 【稳定锁3】严格亚像素钳制
        s3_kernal_posx = torch.clamp(self.KernelOutkernelx[0](z), -0.99, 0.99)
        s3_kernal_posy = torch.clamp(self.KernelOutkernely[0](z), -0.99, 0.99)

        z = self.feat_extract[3](z)

        out3 = self.safe_module_kernel(
            self.modulePad(torch.cat([x_4, x_4.new_ones(x_4.size(0), 1, x_4.size(2), x_4.size(3))], 1)),
            s3_kernal_posx, s3_kernal_posy, s3_kernal_alpha, s3_kernal_beta, s3_kernal_weight)

        out3_norm = out3[:, -1:, :, :]
        out3_norm[out3_norm.abs() < 0.01] = 1.0
        out3 = out3[:, :-1, :, :] / out3_norm
        out3 += s3_kernal_bias

        if not self.inference:
            outputs.append(out3)
            outputs_fil.append(x_4)

            s3_Alpha = torch.mean(s3_kernal_weight * s3_kernal_alpha, dim=1, keepdim=True)
            s3_Beta = torch.mean(s3_kernal_weight * s3_kernal_beta, dim=1, keepdim=True)
            loss_s3_Alpha = CharbonnierFunc(s3_Alpha[:, :, :, :-1] - s3_Alpha[:, :, :, 1:]) + CharbonnierFunc(
                s3_Alpha[:, :, :-1, :] - s3_Alpha[:, :, 1:, :])
            loss_s3_Beta = CharbonnierFunc(s3_Beta[:, :, :, :-1] - s3_Beta[:, :, :, 1:]) + CharbonnierFunc(
                s3_Beta[:, :, :-1, :] - s3_Beta[:, :, 1:, :])
            Kernal_Loss += loss_s3_Alpha
            Kernal_Loss += loss_s3_Beta

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)

        # ---------------- Scale 1/2 ----------------
        s2_kernal_flow_data = self.KernelPredictFlow[1](z) + self.flowup(s3_kernal_flow) * 2
        s2_kernal_flow_prior = self.KinematicHeads[1](z)
        s2_kernal_flow = s2_kernal_flow_data + s2_kernal_flow_prior

        s2_kernal_flowmask = self.KernelPredictFlowMask[1](z)
        s2_kernal_flowmask = self.sigmoid(s2_kernal_flowmask)

        zx2 = torch.cat([z, x_2], 1)
        s2_kernal_flowfeat0, x_2_0 = torch.split(flow_warp(zx2, s2_kernal_flow.permute(0, 2, 3, 1)), self.dim * 2,
                                                 dim=1)
        s2_kernal_flowfeat1, x_2_1 = torch.split(flow_warp(zx2, -s2_kernal_flow.permute(0, 2, 3, 1)), self.dim * 2,
                                                 dim=1)
        x_2 = x_2_0 * s2_kernal_flowmask + x_2_1 * (1 - s2_kernal_flowmask)

        s2_kernal_bias = self.KernelOutBias[1](z)

        z = torch.cat([z, s2_kernal_flowfeat0 * s2_kernal_flowmask + s2_kernal_flowfeat1 * (1 - s2_kernal_flowmask)], 1)
        s2_kernal_weight = self.KernelOutWeight[1](z)
        s2_kernal_weight = self.softmax(s2_kernal_weight)
        s2_kernal_alpha = self.KernelOutAlpha[1](z)
        s2_kernal_beta = self.KernelOutBeta[1](z)

        # 【稳定锁3】严格亚像素钳制
        s2_kernal_posx = torch.clamp(self.KernelOutkernelx[1](z), -0.99, 0.99)
        s2_kernal_posy = torch.clamp(self.KernelOutkernely[1](z), -0.99, 0.99)

        z = self.feat_extract[4](z)

        out2 = self.safe_module_kernel(
            self.modulePad(torch.cat([x_2, x_2.new_ones(x_2.size(0), 1, x_2.size(2), x_2.size(3))], 1)),
            s2_kernal_posx, s2_kernal_posy, s2_kernal_alpha, s2_kernal_beta, s2_kernal_weight)

        out2_norm = out2[:, -1:, :, :]
        out2_norm[out2_norm.abs() < 0.01] = 1.0
        out2 = out2[:, :-1, :, :] / out2_norm
        out2 += s2_kernal_bias

        if not self.inference:
            outputs.append(out2)
            outputs_fil.append(x_2)

            s2_Alpha = torch.mean(s2_kernal_weight * s2_kernal_alpha, dim=1, keepdim=True)
            s2_Beta = torch.mean(s2_kernal_weight * s2_kernal_beta, dim=1, keepdim=True)
            loss_s2_Alpha = CharbonnierFunc(s2_Alpha[:, :, :, :-1] - s2_Alpha[:, :, :, 1:]) + CharbonnierFunc(
                s2_Alpha[:, :, :-1, :] - s2_Alpha[:, :, 1:, :])
            loss_s2_Beta = CharbonnierFunc(s2_Beta[:, :, :, :-1] - s2_Beta[:, :, :, 1:]) + CharbonnierFunc(
                s2_Beta[:, :, :-1, :] - s2_Beta[:, :, 1:, :])
            Kernal_Loss += loss_s2_Alpha
            Kernal_Loss += loss_s2_Beta

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)

        z = self.Decoder[2](z)

        # ---------------- Full Scale ----------------
        s1_kernal_flow_data = self.KernelPredictFlow[2](z) + self.flowup(s2_kernal_flow) * 2
        s1_kernal_flow_prior = self.KinematicHeads[2](z)
        s1_kernal_flow = s1_kernal_flow_data + s1_kernal_flow_prior

        s1_kernal_flowmask = self.KernelPredictFlowMask[2](z)
        s1_kernal_flowmask = self.sigmoid(s1_kernal_flowmask)

        zx = torch.cat([z, x], 1)
        s1_kernal_flowfeat0, x_0 = torch.split(flow_warp(zx, s1_kernal_flow.permute(0, 2, 3, 1)), self.dim, dim=1)
        s1_kernal_flowfeat1, x_1 = torch.split(flow_warp(zx, -s1_kernal_flow.permute(0, 2, 3, 1)), self.dim, dim=1)
        x = x_0 * s1_kernal_flowmask + x_1 * (1 - s1_kernal_flowmask)

        s1_kernal_bias = self.KernelOutBias[2](z)
        z = torch.cat([z, s1_kernal_flowfeat0 * s1_kernal_flowmask + s1_kernal_flowfeat1 * (1 - s1_kernal_flowmask)], 1)
        s1_kernal_weight = self.KernelOutWeight[2](z)
        s1_kernal_weight = self.softmax(s1_kernal_weight)
        s1_kernal_alpha = self.KernelOutAlpha[2](z)
        s1_kernal_beta = self.KernelOutBeta[2](z)

        # 【稳定锁3】严格亚像素钳制
        s1_kernal_posx = torch.clamp(self.KernelOutkernelx[2](z), -0.99, 0.99)
        s1_kernal_posy = torch.clamp(self.KernelOutkernely[2](z), -0.99, 0.99)

        out = self.safe_module_kernel(self.modulePad(torch.cat([x, x.new_ones(x.size(0), 1, x.size(2), x.size(3))], 1)),
                                      s1_kernal_posx, s1_kernal_posy, s1_kernal_alpha, s1_kernal_beta, s1_kernal_weight)

        out_norm = out[:, -1:, :, :]
        out_norm[out_norm.abs() < 0.01] = 1.0
        out = out[:, :-1, :, :] / out_norm
        out += s1_kernal_bias

        if not self.inference:
            outputs.append(out)
            outputs_fil.append(x)

            s1_Alpha = torch.mean(s1_kernal_weight * s1_kernal_alpha, dim=1, keepdim=True)
            s1_Beta = torch.mean(s1_kernal_weight * s1_kernal_beta, dim=1, keepdim=True)
            loss_s1_Alpha = CharbonnierFunc(s1_Alpha[:, :, :, :-1] - s1_Alpha[:, :, :, 1:]) + CharbonnierFunc(
                s1_Alpha[:, :, :-1, :] - s1_Alpha[:, :, 1:, :])
            loss_s1_Beta = CharbonnierFunc(s1_Beta[:, :, :, :-1] - s1_Beta[:, :, :, 1:]) + CharbonnierFunc(
                s1_Beta[:, :, :-1, :] - s1_Beta[:, :, 1:, :])
            Kernal_Loss += loss_s1_Alpha
            Kernal_Loss += loss_s1_Beta

            return outputs[::-1], outputs_fil[::-1]
        else:
            return out