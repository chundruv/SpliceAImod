# Unpadded ("valid") execution of the fused SpliceAI ensemble.
#
# Every conv in SpliceAI uses "same" zero padding, so each of the 32 dilated
# convs computes all L input positions, and the network then discards
# CL//2 = 5000 positions from each end. A retained output position i (with
# CL//2 <= i < L - CL//2) has a full receptive field inside the sequence at
# every layer, so its value does not depend on the padding at all. Running the
# convs unpadded and trimming the residual/skip streams as the receptive field
# grows therefore computes exactly the same retained window while skipping the
# positions that would be cropped. The w=41/ar=25 units evaluate ~1k-9k
# positions instead of 11001: about 32% fewer FLOPs overall at -D 500.
#
# Two further, also exact, transformations are available:
#
#   fold_bn   -- in each residual unit conv1's output feeds batchnorm2 and
#                nothing else, so bn2 (a per-channel affine in eval mode) is
#                folded into conv1's weights. One fewer elementwise pass per RU.
#
#   conv2d    -- a dilated 1-D conv is identical to a dense 2-D conv with kernel
#                (w, 1) on the sequence viewed as (L/ar, ar): element (h, r) of
#                that view is position h*ar + r, so a kernel tap k at row h+k
#                reads position (h+k)*ar + r = i + k*ar for i = h*ar + r. This
#                is a zero-copy reshape (plus a right pad to a multiple of ar,
#                which only affects trimmed positions). cuDNN's dense conv
#                kernels are far better tuned than its dilated ones, and the
#                output view interleaves the phases back with no permute.
#
#   nhwc      -- cuDNN's tensor-core conv kernels are NHWC-native. Given NCW
#                input they transpose in and out around EVERY conv
#                (nchwToNhwcKernel / nhwcToNchwKernel: ~17% of GPU time on an
#                A100 at fp16). Running the network as conv2d over an
#                (N, C, 1, L) tensor held in channels_last memory format lets
#                cuDNN consume and produce NHWC directly, so those transposes
#                disappear. Elementwise ops preserve the format, so the whole
#                forward stays NHWC. Numerically identical.
#
# All are validated against the padded fused model in tests/ (max abs
# deviation at fp32 is reassociation-level, ~1e-6).

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from spliceai.batch.fused_inference import FusedSpliceAIEnsemble, _module_layout

CONV_IMPLS = ('padded', 'valid', 'valid2d', 'valid_nhwc')


def _bn_affine(bn):
    a = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    b = bn.bias - a * bn.running_mean
    return a, b


def _valid_conv_nhwc(x4, weight4, bias, ar, groups):
    """Unpadded conv over (N, C, 1, L) channels_last with kernel (1, w), dilation (1, ar)."""
    return F.conv2d(x4, weight4, bias, dilation=(1, ar), groups=groups)


def _dilated_valid_conv(x, weight, bias, ar, groups, use_2d):
    """Unpadded conv1d with dilation `ar`; optionally via the (L/ar, ar) 2-D view."""
    if ar == 1 or not use_2d:
        return F.conv1d(x, weight, bias, dilation=ar, groups=groups)
    N, C, L = x.shape
    w = weight.shape[-1]
    Lp = int(math.ceil(L / ar)) * ar
    if Lp != L:
        x = F.pad(x, (0, Lp - L))
    x2 = x.view(N, C, Lp // ar, ar)
    y2 = F.conv2d(x2, weight.unsqueeze(-1), bias, groups=groups)   # (N, Cout, Lp/ar-(w-1), ar)
    y = y2.reshape(N, y2.shape[1], -1)
    Lout = L - (w - 1) * ar
    return y[:, :, :Lout]


class ValidFusedSpliceAIEnsemble(nn.Module):
    """Wraps a loaded FusedSpliceAIEnsemble and runs it unpadded.

    Shares the source module's parameters (no copy) except where bn2 is folded
    into conv1, which materialises new weight/bias tensors at build time.
    forward(x) -> (N, 3, L - CL) fp32, identical to FusedSpliceAIEnsemble.
    """

    def __init__(self, fused: FusedSpliceAIEnsemble, use_2d=True, fold_bn=True, nhwc=False):
        super().__init__()
        self.fused = fused
        self.n_models = fused.n_models
        self.groups = fused.n_models
        self.use_2d = use_2d and not nhwc
        self.fold_bn = fold_bn
        self.nhwc = nhwc
        self.CL = fused.CL
        self.layout = _module_layout()

        self.folded_w = nn.ParameterList()
        self.folded_b = nn.ParameterList()
        if fold_bn:
            with torch.no_grad():
                for mi, spec in enumerate(self.layout):
                    if spec[0] != 'RU':
                        continue
                    ru = fused.residual_units[mi]
                    a, b = _bn_affine(ru.bn2)
                    a = a.to(ru.conv1.weight.dtype)
                    b = b.to(ru.conv1.weight.dtype)
                    w = (ru.conv1.weight * a[:, None, None]).contiguous()
                    bb = (ru.conv1.bias * a + b).contiguous()
                    self.folded_w.append(nn.Parameter(w, requires_grad=False))
                    self.folded_b.append(nn.Parameter(bb, requires_grad=False))

        # NHWC: pre-build 4-D (Cout, Cin/g, 1, w) channels_last weights once.
        self.w4 = {}
        if nhwc:
            cl = torch.channels_last
            with torch.no_grad():
                def to4(w):
                    return nn.Parameter(w.detach().unsqueeze(2).contiguous(memory_format=cl),
                                        requires_grad=False)
                self.w4_initial = to4(fused.initial_conv.weight)
                self.w4_initial_skip = to4(fused.initial_skip.conv.weight)
                self.w4_final = to4(fused.final_conv.weight)
                self.w4_conv1 = nn.ParameterList()
                self.w4_conv2 = nn.ParameterList()
                self.w4_skip = nn.ParameterList()
                ru_i = 0
                for mi, spec in enumerate(self.layout):
                    m = fused.residual_units[mi]
                    if spec[0] == 'RU':
                        w1 = self.folded_w[ru_i] if fold_bn else m.conv1.weight
                        self.w4_conv1.append(to4(w1))
                        self.w4_conv2.append(to4(m.conv2.weight))
                        ru_i += 1
                    else:
                        self.w4_skip.append(to4(m.conv.weight))

    def _forward_nhwc(self, x):
        f = self.fused
        N = x.shape[0]
        cl = torch.channels_last
        x = x.unsqueeze(2).contiguous(memory_format=cl)             # (N, 4, 1, L)
        x = F.conv2d(x, self.w4_initial, f.initial_conv.bias)
        skip = F.conv2d(x, self.w4_initial_skip, f.initial_skip.conv.bias, groups=self.groups)
        ru_i = 0
        sk_i = 0
        for mi, spec in enumerate(self.layout):
            m = f.residual_units[mi]
            if spec[0] == 'RU':
                _, w, ar = spec
                p = (w - 1) * ar // 2
                out = torch.relu(F.batch_norm(x, m.bn1.running_mean, m.bn1.running_var,
                                              m.bn1.weight, m.bn1.bias, False, 0.0, m.bn1.eps))
                if self.fold_bn:
                    out = _valid_conv_nhwc(out, self.w4_conv1[ru_i], self.folded_b[ru_i], ar, self.groups)
                    out = torch.relu(out)
                else:
                    out = _valid_conv_nhwc(out, self.w4_conv1[ru_i], m.conv1.bias, ar, self.groups)
                    out = torch.relu(F.batch_norm(out, m.bn2.running_mean, m.bn2.running_var,
                                                  m.bn2.weight, m.bn2.bias, False, 0.0, m.bn2.eps))
                out = _valid_conv_nhwc(out, self.w4_conv2[ru_i], m.conv2.bias, ar, self.groups)
                x = x[:, :, :, 2 * p:x.shape[-1] - 2 * p] + out
                ru_i += 1
            else:
                d = (skip.shape[-1] - x.shape[-1]) // 2
                if d > 0:
                    skip = skip[:, :, :, d:skip.shape[-1] - d]
                skip = F.conv2d(x, self.w4_skip[sk_i], m.conv.bias, groups=self.groups) + skip
                sk_i += 1
        out = F.conv2d(skip, self.w4_final, f.final_conv.bias, groups=self.groups)  # (N, 3n, 1, Lout)
        Lout = out.shape[-1]
        out = out.reshape(N, self.n_models, 3, Lout)
        out = F.softmax(out, dim=2)
        return out.float().mean(dim=1)

    def forward(self, x):
        if self.nhwc:
            return self._forward_nhwc(x)
        f = self.fused
        N = x.shape[0]
        x = f.initial_conv(x)
        skip = f.initial_skip.conv(x)
        ru_i = 0
        for mi, spec in enumerate(self.layout):
            m = f.residual_units[mi]
            if spec[0] == 'RU':
                _, w, ar = spec
                p = (w - 1) * ar // 2
                out = torch.relu(m.bn1(x))
                if self.fold_bn:
                    out = _dilated_valid_conv(out, self.folded_w[ru_i], self.folded_b[ru_i],
                                              ar, self.groups, self.use_2d)
                    out = torch.relu(out)
                else:
                    out = _dilated_valid_conv(out, m.conv1.weight, m.conv1.bias,
                                              ar, self.groups, self.use_2d)
                    out = torch.relu(m.bn2(out))
                out = _dilated_valid_conv(out, m.conv2.weight, m.conv2.bias,
                                          ar, self.groups, self.use_2d)
                x = x[:, :, 2 * p:x.shape[-1] - 2 * p] + out
                ru_i += 1
            else:
                d = (skip.shape[-1] - x.shape[-1]) // 2
                if d > 0:
                    skip = skip[:, :, d:skip.shape[-1] - d]
                skip = m.conv(x) + skip
        out = f.final_conv(skip)                          # (N, 3*n_models, Lout)
        Lout = out.shape[-1]
        out = out.view(N, self.n_models, 3, Lout)
        out = F.softmax(out, dim=2)
        return out.float().mean(dim=1)


def wrap_conv_impl(fused, conv_impl):
    """Return a callable module for the requested conv implementation."""
    if conv_impl == 'padded':
        return fused
    if conv_impl == 'valid':
        return ValidFusedSpliceAIEnsemble(fused, use_2d=False, fold_bn=True)
    if conv_impl == 'valid2d':
        return ValidFusedSpliceAIEnsemble(fused, use_2d=True, fold_bn=True)
    if conv_impl == 'valid_nhwc':
        return ValidFusedSpliceAIEnsemble(fused, fold_bn=True, nhwc=True)
    raise ValueError(f"unknown conv_impl {conv_impl!r}; choose from {CONV_IMPLS}")
