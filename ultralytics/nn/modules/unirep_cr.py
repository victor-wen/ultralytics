from __future__ import annotations

import functools
import math
import warnings
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import C2f
from ultralytics.nn.modules.conv import Conv

__all__ = (
    "SEBlock",
    "ECABlock",
    "SpatialGate",
    "DilatedReparamBlock",
    "ContextResidualLargeKernelBlock",
    "UniRepBottleneck_CR",
    "C3k2_UniRep_CR",
    "build_pose_supervision_mask",
    "reparameterize_unirep_model",
)


DILATE_SETTINGS: Dict[int, Tuple[List[int], List[int]]] = {
    17: ([5, 9, 3, 3, 3], [1, 2, 4, 5, 7]),
    15: ([5, 7, 3, 3, 3], [1, 2, 3, 5, 7]),
    13: ([5, 7, 3, 3, 3], [1, 2, 3, 4, 5]),
    11: ([5, 5, 3, 3, 3], [1, 2, 3, 4, 5]),
    9: ([5, 5, 3, 3], [1, 2, 3, 4]),
    7: ([7, 5, 3], [1, 1, 1]),
    5: ([3, 3], [1, 2]),
}

FUSION_TOL: float = 1e-4


@functools.lru_cache(maxsize=1)
def _get_lk_impl():
    try:
        from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM as _Impl

        return _Impl
    except ImportError:
        return None
    except Exception as e:
        warnings.warn(f"加载第三方大核实现时异常: {e}", UserWarning)
        return None


def _to_2tuple(x):
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
        return int(x[0]), int(x[1])
    return int(x), int(x)


def _validate_positive_odd(name: str, value: int) -> int:
    value = int(value)
    if value <= 0 or value % 2 == 0:
        raise ValueError(f"{name} 必须为正奇数，当前为 {value}")
    return value


def get_conv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding=None,
    dilation=1,
    groups=1,
    bias=False,
    attempt_use_lk_impl=True,
):
    kernel_size = _to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = _to_2tuple(padding)

    need_large = (
        kernel_size[0] == kernel_size[1]
        and kernel_size[0] > 5
        and padding == (kernel_size[0] // 2, kernel_size[1] // 2)
    )

    if attempt_use_lk_impl and need_large:
        lk_impl = _get_lk_impl()
        if lk_impl is not None and in_channels == out_channels == groups and stride == 1 and dilation == 1:
            try:
                return lk_impl(in_channels, kernel_size[0], bias=bias)
            except Exception:
                pass

    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
    )


def fuse_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    conv_bias = torch.zeros_like(bn.running_mean) if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    fused_w = conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1)
    fused_b = bn.bias + (conv_bias - bn.running_mean) * bn.weight / std
    return fused_w, fused_b


def convert_dilated_to_nondilated(kernel, dilate_rate):
    if dilate_rate == 1:
        return kernel.clone()
    identity = torch.ones(1, 1, 1, 1, device=kernel.device, dtype=kernel.dtype)
    return F.conv_transpose2d(kernel, identity, stride=dilate_rate)


def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    equiv_ks = dilated_r * (dilated_kernel.size(2) - 1) + 1
    equiv_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    pad = large_k // 2 - equiv_ks // 2
    if pad < 0:
        raise ValueError(f"等价核 {equiv_ks} > 大核 {large_k}")
    return large_kernel + F.pad(equiv_kernel, [pad] * 4)


def _apply_cdc_theta(weight, theta):
    w = weight.clone()
    center = w.size(2) // 2
    w_sum = weight.sum(dim=[1, 2, 3])
    w[:, 0, center, center] -= theta * w_sum
    return w


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


class ECABlock(nn.Module):
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        k = max(k, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


def build_attention(channels: int, attn: str = "se", **kwargs) -> nn.Module:
    attn = attn.lower().strip()
    if attn == "se":
        return SEBlock(channels, **kwargs)
    if attn == "eca":
        return ECABlock(channels, **kwargs)
    if attn in ("none", "identity", ""):
        return nn.Identity()
    raise ValueError(f"不支持的注意力: '{attn}'，可选 'se'/'eca'/'none'")


class SpatialGate(nn.Module):
    def __init__(self, channels: int, gate_k: int = 5, reduction: int = 4, max_scale: float = 0.5):
        super().__init__()
        gate_k = _validate_positive_odd("gate_k", gate_k)
        if max_scale <= 0:
            raise ValueError(f"max_scale 必须 > 0，当前为 {max_scale}")

        mid = max(channels // reduction, 8)
        self.channels = channels
        self.max_scale = float(max_scale)

        self.spatial = nn.Sequential(
            nn.Conv2d(channels, channels, gate_k, padding=gate_k // 2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=False),
        )
        self.channel = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.SiLU(inplace=False),
            nn.Conv2d(mid, 1, 1, bias=True),
        )

        nn.init.normal_(self.channel[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.channel[-1].bias)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.tanh(self.channel(self.spatial(x)))
        scale = self.max_scale * torch.tanh(self.scale)
        return x * (1.0 + scale * gate)

    def extra_repr(self) -> str:
        k = self.spatial[0].kernel_size[0]
        mid = self.channel[0].out_channels
        return f"channels={self.channels}, gate_k={k}, mid={mid}, max_scale={self.max_scale}"


class DilatedReparamBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        deploy=False,
        use_sync_bn=False,
        attempt_use_lk_impl=True,
        cdc_kernel_size=3,
    ):
        super().__init__()
        self.deploy = deploy
        self.attempt_use_lk_impl = attempt_use_lk_impl
        self.kernel_size = kernel_size
        self.channels = channels
        self.cdc_kernel_size = _validate_positive_odd("cdc_kernel_size", cdc_kernel_size)

        if kernel_size not in DILATE_SETTINGS:
            raise ValueError(f"kernel_size 须在 {list(DILATE_SETTINGS.keys())} 中")
        self.kernel_sizes, self.dilates = DILATE_SETTINGS[kernel_size]

        self.lk_origin = get_conv2d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            dilation=1,
            groups=channels,
            bias=deploy,
            attempt_use_lk_impl=deploy and attempt_use_lk_impl,
        )

        if not deploy:
            bn_cls = nn.SyncBatchNorm if use_sync_bn else nn.BatchNorm2d
            self.origin_bn = bn_cls(channels)
            self.dil_branches = nn.ModuleList()
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.dil_branches.append(
                    nn.Sequential(
                        nn.Conv2d(
                            channels,
                            channels,
                            k,
                            stride=1,
                            padding=r * (k - 1) // 2,
                            dilation=r,
                            groups=channels,
                            bias=False,
                        ),
                        bn_cls(channels),
                    )
                )

            self.branch_1x1 = nn.Sequential(
                nn.Conv2d(channels, channels, 1, stride=1, padding=0, groups=channels, bias=False),
                bn_cls(channels),
            )
            self.branch_cdc = nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    self.cdc_kernel_size,
                    stride=1,
                    padding=self.cdc_kernel_size // 2,
                    dilation=1,
                    groups=channels,
                    bias=False,
                ),
                bn_cls(channels),
            )
            self.theta_cdc = nn.Parameter(torch.zeros(channels))

    @staticmethod
    def _cdc_conv(conv, x, theta):
        y = conv(x)
        w_sum = conv.weight.sum(dim=[1, 2, 3])
        correction = (theta * w_sum).view(1, -1, 1, 1) * x
        return y - correction

    def forward(self, x):
        if self.deploy:
            return self.lk_origin(x)

        out = self.origin_bn(self.lk_origin(x))
        for branch in self.dil_branches:
            out = out + branch(x)
        out = out + self.branch_1x1(x)
        out = out + self.branch_cdc[1](self._cdc_conv(self.branch_cdc[0], x, self.theta_cdc))
        return out

    @torch.no_grad()
    def merge_dilated_branches(self, force_plain_conv=False):
        if self.deploy:
            return

        origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)

        for r, branch in zip(self.dilates, self.dil_branches):
            br_k, br_b = fuse_bn(branch[0], branch[1])
            origin_k = merge_dilated_into_large_kernel(origin_k, br_k, r)
            origin_b += br_b

        k_1x1, b_1x1 = fuse_bn(self.branch_1x1[0], self.branch_1x1[1])
        origin_k = merge_dilated_into_large_kernel(origin_k, k_1x1, 1)
        origin_b += b_1x1

        cdc_k, cdc_b = fuse_bn(self.branch_cdc[0], self.branch_cdc[1])
        cdc_k = _apply_cdc_theta(cdc_k, self.theta_cdc.detach())
        origin_k = merge_dilated_into_large_kernel(origin_k, cdc_k, 1)
        origin_b += cdc_b

        merged = get_conv2d(
            origin_k.size(0),
            origin_k.size(0),
            origin_k.size(2),
            stride=1,
            padding=origin_k.size(2) // 2,
            dilation=1,
            groups=origin_k.size(0),
            bias=True,
            attempt_use_lk_impl=(self.attempt_use_lk_impl and not force_plain_conv),
        ).to(device=origin_k.device, dtype=origin_k.dtype)

        if isinstance(merged, nn.Conv2d):
            merged.weight.copy_(origin_k)
            merged.bias.copy_(origin_b)
        else:
            if hasattr(merged, "weight"):
                merged.weight.copy_(origin_k)
            else:
                raise RuntimeError("LK impl 缺少 weight 属性，无法融合")

            if hasattr(merged, "bias") and merged.bias is not None:
                merged.bias.copy_(origin_b)
            elif origin_b.abs().max() > FUSION_TOL:
                warnings.warn(
                    f"LK impl 不支持 bias，但 bias 范数为 {origin_b.abs().max():.4e}，融合结果可能不等价！",
                    UserWarning,
                )

        self.lk_origin = merged
        for attr in ("origin_bn", "dil_branches", "branch_1x1", "branch_cdc", "theta_cdc"):
            if hasattr(self, attr):
                delattr(self, attr)
        self.deploy = True

    def fuse_convs(self, force_plain_conv=False):
        self.merge_dilated_branches(force_plain_conv=force_plain_conv)

    def extra_repr(self):
        return (
            f"channels={self.channels}, kernel_size={self.kernel_size}, "
            f"cdc_kernel_size={self.cdc_kernel_size}, deploy={self.deploy}"
        )


class ContextResidualLargeKernelBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        deploy: bool = False,
        use_sync_bn: bool = False,
        attempt_use_lk_impl: bool = True,
        cdc_kernel_size: int = 3,
        attn: str = "se",
        use_gate: bool = False,
        gate_k: int = 5,
        local_k: int = 3,
        gamma_per_channel: bool = True,
    ):
        super().__init__()
        local_k = _validate_positive_odd("local_k", local_k)
        bn_cls = nn.SyncBatchNorm if use_sync_bn else nn.BatchNorm2d

        self.channels = channels
        self.kernel_size = kernel_size
        self.deploy = deploy
        self.use_gate = use_gate

        self.local_dw = nn.Sequential(
            nn.Conv2d(channels, channels, local_k, stride=1, padding=local_k // 2, groups=channels, bias=False),
            bn_cls(channels),
        )

        self.rep_lk = DilatedReparamBlock(
            channels,
            kernel_size,
            deploy=deploy,
            use_sync_bn=use_sync_bn,
            attempt_use_lk_impl=attempt_use_lk_impl,
            cdc_kernel_size=cdc_kernel_size,
        )
        self.attn_g = build_attention(channels, attn)
        self.post_bn_g = bn_cls(channels)

        if use_gate:
            self.spatial_gate_g = SpatialGate(channels, gate_k=gate_k)

        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1) if gamma_per_channel else torch.zeros(1))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        l = u + self.local_dw(u)
        g = self.rep_lk(u)
        g = self.attn_g(g)
        g = self.post_bn_g(g)
        if self.use_gate:
            g = self.spatial_gate_g(g)
        return l + self.gamma * g

    def fuse_convs(self, force_plain_conv: bool = False):
        self.rep_lk.fuse_convs(force_plain_conv=force_plain_conv)
        self.deploy = True

    def extra_repr(self):
        return f"channels={self.channels}, kernel_size={self.kernel_size}, deploy={self.deploy}, use_gate={self.use_gate}"


class UniRepBottleneck_CR(nn.Module):
    def __init__(
        self,
        c1,
        c2,
        shortcut=True,
        e=0.5,
        k=13,
        deploy=False,
        attn="se",
        use_gate=False,
        gate_k=5,
        local_k=3,
        use_sync_bn=False,
        attempt_use_lk_impl=True,
        cdc_kernel_size=3,
        gamma_per_channel=True,
    ):
        super().__init__()
        self.deploy = deploy
        c_ = int(c2 * e)

        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.ctx = ContextResidualLargeKernelBlock(
            channels=c_,
            kernel_size=k,
            deploy=deploy,
            use_sync_bn=use_sync_bn,
            attempt_use_lk_impl=attempt_use_lk_impl,
            cdc_kernel_size=cdc_kernel_size,
            attn=attn,
            use_gate=use_gate,
            gate_k=gate_k,
            local_k=local_k,
            gamma_per_channel=gamma_per_channel,
        )
        self.act = nn.SiLU(inplace=True)
        self.cv2 = Conv(c_, c2, 1, 1, act=False)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.cv2(self.act(self.ctx(self.cv1(x))))
        return x + out if self.add else out

    def fuse_convs(self, force_plain_conv=False):
        self.ctx.fuse_convs(force_plain_conv=force_plain_conv)
        self.deploy = True


class C3k2_UniRep_CR(C2f):
    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=False,
        e=0.5,
        k=13,
        deploy=False,
        attn="se",
        use_gate=False,
        gate_k=5,
        local_k=3,
    ):
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=1, e=e)
        self.deploy = deploy
        self.m = nn.ModuleList(
            UniRepBottleneck_CR(
                self.c,
                self.c,
                shortcut=shortcut,
                e=1.0,
                k=k,
                deploy=deploy,
                attn=attn,
                use_gate=use_gate,
                gate_k=gate_k,
                local_k=local_k,
            )
            for _ in range(n)
        )

    def fuse_convs(self, force_plain_conv=False):
        for m in self.m:
            if hasattr(m, "fuse_convs"):
                m.fuse_convs(force_plain_conv=force_plain_conv)
        self.deploy = True


@torch.no_grad()
def build_pose_supervision_mask(
    anchor_points: torch.Tensor,
    stride_tensor: torch.Tensor,
    gt_bboxes: torch.Tensor,
    mask_gt: torch.Tensor,
    fg_mask: torch.Tensor,
    target_gt_idx: torch.Tensor,
    allowed_strides=(8, 16),
    center_radius: float = 1.5,
    extra_weight: float = 0.5,
):
    b, _, _ = gt_bboxes.shape
    a = anchor_points.shape[0]
    stride = stride_tensor.view(-1) if stride_tensor.ndim == 2 else stride_tensor
    valid_gt = mask_gt.squeeze(-1).bool() if mask_gt.ndim == 3 else mask_gt.bool()

    anchor_points_img = anchor_points * stride.unsqueeze(-1)
    gt_centers = 0.5 * (gt_bboxes[..., 0:2] + gt_bboxes[..., 2:4])

    stride_ok = torch.zeros_like(stride, dtype=torch.bool)
    for s in allowed_strides:
        stride_ok |= (stride == float(s)) | (stride == int(s))
    stride_ok = stride_ok.view(1, a, 1)

    dx = (anchor_points_img[None, :, None, 0] - gt_centers[:, None, :, 0]).abs()
    dy = (anchor_points_img[None, :, None, 1] - gt_centers[:, None, :, 1]).abs()
    r = center_radius * stride.view(1, a, 1)
    center_3x3 = (dx <= r) & (dy <= r) & stride_ok & valid_gt[:, None, :]
    center_3x3 = center_3x3 & (~fg_mask[:, :, None])

    dist2 = dx.square() + dy.square()
    dist2 = dist2.masked_fill(~center_3x3, float("inf"))
    nearest_gt = dist2.argmin(dim=-1)
    extra_pose_mask = center_3x3.any(dim=-1)

    pose_mask = fg_mask | extra_pose_mask
    pose_gt_idx = target_gt_idx.clone()
    pose_gt_idx[extra_pose_mask] = nearest_gt[extra_pose_mask]
    pose_weight = fg_mask.float() + extra_pose_mask.float() * float(extra_weight)
    return pose_mask, pose_gt_idx, pose_weight, extra_pose_mask


def reparameterize_unirep_model(model, force_plain_conv_for_export=True):
    was_training = model.training
    model.eval()

    for m in model.modules():
        if isinstance(m, DilatedReparamBlock) and not m.deploy:
            m.merge_dilated_branches(force_plain_conv=force_plain_conv_for_export)

    for mod in model.modules():
        if isinstance(
            mod,
            (
                DilatedReparamBlock,
                ContextResidualLargeKernelBlock,
                UniRepBottleneck_CR,
                C3k2_UniRep_CR,
            ),
        ):
            mod.deploy = True

    if was_training:
        warnings.warn("模型已不可逆融合，请保持 eval 模式。", UserWarning)
    return model
