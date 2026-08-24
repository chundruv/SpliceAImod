# Fused grouped-convolution SpliceAI ensemble + optimized inference path.
#
# The 5 SpliceAI models share an identical architecture and differ only in their
# weights. Running them as 5 separate forward passes uses only L=32 channels per
# conv -- far too narrow to saturate a modern GPU's tensor cores, and it pays the
# Python/kernel-launch overhead 5x.
#
# This module stacks all 5 into a single network whose convolutions use
# ``groups=5`` (160 channels wide). A grouped convolution computes exactly the
# same thing as N independent convolutions, so the output is mathematically
# identical to averaging the 5 separate models -- but it runs as ONE kernel
# sequence, with 5x the arithmetic intensity per launch.
#
# It is fully hardware-agnostic: grouped convs are standard cuDNN/MKLDNN ops, so
# this runs unchanged on any NVIDIA GPU (and on CPU). No custom CUDA required.
#
# Accumulation note: each model applies softmax over its 3 output channels, then
# the 5 are averaged. We replicate that exactly (per-group softmax -> mean), and
# the mean is taken in fp32 regardless of compute dtype so fp16/bf16 inference
# does not bias the ensemble average.

import torch
import torch.nn as nn
import torch.nn.functional as F

# SpliceAI-10k hyperparameters (must match spliceai.utils.SpliceAI)
_W  = [11, 11, 11, 11, 11, 11, 11, 11, 21, 21, 21, 21, 41, 41, 41, 41]
_AR = [1, 1, 1, 1, 4, 4, 4, 4, 10, 10, 10, 10, 25, 25, 25, 25]
_L = 32
_N_MODELS = 5


def _module_layout():
    """Return the ordered residual_units module layout matching SpliceAI.

    Yields ('RU', w, ar) for each residual unit and ('SKIP',) after every 4th,
    exactly as spliceai.utils.SpliceAI builds its ModuleList.
    """
    layout = []
    for i, (w, ar) in enumerate(zip(_W, _AR)):
        layout.append(('RU', w, ar))
        if (i + 1) % 4 == 0:
            layout.append(('SKIP',))
    return layout


def _stack_conv(per_model_weights, per_model_biases, groups):
    """Build a grouped Conv1d-ready (weight, bias) by stacking N per-model tensors.

    For groups=N each model occupies one group; concatenating the per-model
    weight tensors along the output-channel axis yields the block-diagonal
    grouped-conv weight. For the initial conv (groups=1, 4 input channels) the
    same concatenation gives a plain wide conv.
    """
    weight = torch.cat([w for w in per_model_weights], dim=0).contiguous()
    bias = torch.cat([b for b in per_model_biases], dim=0).contiguous()
    return weight, bias


def _stack_bn(states, prefix):
    """Concatenate BatchNorm parameters from N models into one wide BN."""
    weight = torch.cat([s[prefix + '.weight'] for s in states])
    bias = torch.cat([s[prefix + '.bias'] for s in states])
    rmean = torch.cat([s[prefix + '.running_mean'] for s in states])
    rvar = torch.cat([s[prefix + '.running_var'] for s in states])
    return weight, bias, rmean, rvar


class _GroupedResidualUnit(nn.Module):
    def __init__(self, ln, w, ar, groups):
        super().__init__()
        padding = (w - 1) * ar // 2
        self.bn1 = nn.BatchNorm1d(ln)
        self.bn2 = nn.BatchNorm1d(ln)
        self.conv1 = nn.Conv1d(ln, ln, w, dilation=ar, padding=padding, groups=groups)
        self.conv2 = nn.Conv1d(ln, ln, w, dilation=ar, padding=padding, groups=groups)

    def forward(self, x, skip):
        out = self.conv1(torch.relu(self.bn1(x)))
        out = self.conv2(torch.relu(self.bn2(out)))
        return x + out, skip


class _GroupedSkip(nn.Module):
    def __init__(self, ln, groups):
        super().__init__()
        self.conv = nn.Conv1d(ln, ln, 1, groups=groups)

    def forward(self, x, skip):
        return x, self.conv(x) + skip


class FusedSpliceAIEnsemble(nn.Module):
    """The 5-model SpliceAI ensemble fused into one grouped-conv network.

    forward(x): x is (N, 4, Lseq) one-hot DNA. Returns (N, 3, Lout) -- the mean
    of the 5 per-model softmax outputs, identical to the sequential ensemble.
    """

    def __init__(self, n_models=_N_MODELS, ln=_L):
        super().__init__()
        self.n_models = n_models
        self.ln = ln
        self.wide = ln * n_models
        self.CL = 2 * sum(ar * (w - 1) for ar, w in zip(_AR, _W))
        self.crop_amount = self.CL // 2

        # initial conv: 4 -> wide, groups=1 (shared 4-channel input, per-model out)
        self.initial_conv = nn.Conv1d(4, self.wide, 1)
        self.initial_skip = _GroupedSkip(self.wide, n_models)

        self.residual_units = nn.ModuleList()
        for spec in _module_layout():
            if spec[0] == 'RU':
                _, w, ar = spec
                self.residual_units.append(_GroupedResidualUnit(self.wide, w, ar, n_models))
            else:
                self.residual_units.append(_GroupedSkip(self.wide, n_models))

        self.final_conv = nn.Conv1d(self.wide, 3 * n_models, 1, groups=n_models)

    @torch.no_grad()
    def load_from_state_dicts(self, state_dicts):
        """Populate grouped weights from the 5 original per-model state dicts."""
        assert len(state_dicts) == self.n_models, "expected %d state dicts" % self.n_models
        sd = state_dicts

        # initial_conv (groups=1): stack out-channels -> (wide, 4, 1)
        w, b = _stack_conv([s['initial_conv.weight'] for s in sd],
                           [s['initial_conv.bias'] for s in sd], groups=1)
        self.initial_conv.weight.copy_(w)
        self.initial_conv.bias.copy_(b)

        # initial_skip conv (groups=N)
        w, b = _stack_conv([s['initial_skip.conv.weight'] for s in sd],
                           [s['initial_skip.conv.bias'] for s in sd], groups=self.n_models)
        self.initial_skip.conv.weight.copy_(w)
        self.initial_skip.conv.bias.copy_(b)

        for mi, spec in enumerate(_module_layout()):
            mod = self.residual_units[mi]
            base = 'residual_units.%d' % mi
            if spec[0] == 'RU':
                for bn_name in ('bn1', 'bn2'):
                    src = base + ('.batchnorm1' if bn_name == 'bn1' else '.batchnorm2')
                    gw, gb, rm, rv = _stack_bn(sd, src)
                    bn = getattr(mod, bn_name)
                    bn.weight.copy_(gw); bn.bias.copy_(gb)
                    bn.running_mean.copy_(rm); bn.running_var.copy_(rv)
                for cn, key in (('conv1', '.conv1'), ('conv2', '.conv2')):
                    w, b = _stack_conv([s[base + key + '.weight'] for s in sd],
                                       [s[base + key + '.bias'] for s in sd], groups=self.n_models)
                    conv = getattr(mod, cn)
                    conv.weight.copy_(w); conv.bias.copy_(b)
            else:  # SKIP
                w, b = _stack_conv([s[base + '.conv.weight'] for s in sd],
                                   [s[base + '.conv.bias'] for s in sd], groups=self.n_models)
                mod.conv.weight.copy_(w); mod.conv.bias.copy_(b)

        # final_conv (groups=N): each model 3 out -> 3*N total
        w, b = _stack_conv([s['final_conv.weight'] for s in sd],
                           [s['final_conv.bias'] for s in sd], groups=self.n_models)
        self.final_conv.weight.copy_(w)
        self.final_conv.bias.copy_(b)
        return self

    def forward(self, x):
        N = x.shape[0]
        x = self.initial_conv(x)
        # skip starts at integer 0 (matches the original; avoids a per-call
        # tensor allocation, which keeps CUDA-graph capture clean).
        x, skip = self.initial_skip(x, 0)
        for m in self.residual_units:
            x, skip = m(x, skip)
        c = self.crop_amount
        skip = skip[:, :, c:-c]
        out = self.final_conv(skip)                      # (N, 3*n_models, Lout)
        Lout = out.shape[-1]
        out = out.view(N, self.n_models, 3, Lout)
        # per-model softmax over the 3 channels, then ensemble mean in fp32
        out = F.softmax(out, dim=2)
        out = out.float().mean(dim=1)                    # (N, 3, Lout) fp32
        return out


def build_fused_ensemble(state_dicts, device, dtype):
    """Convenience: build, load, move, set eval/precision."""
    model = FusedSpliceAIEnsemble().to(device)
    model.load_from_state_dicts([{k: v.to(device) for k, v in s.items()} for s in state_dicts])
    model.eval()
    if dtype in (torch.float16, torch.bfloat16):
        # Convert conv/BN weights; the forward upcasts the final mean to fp32.
        model = model.to(dtype)
    return model
