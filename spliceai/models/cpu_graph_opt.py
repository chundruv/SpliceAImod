"""Output-preserving graph rewrites for CPU inference.

Both rewrites below are algebraically exact: they change which arithmetic is
performed, not what the model computes. They are CPU-only because they trade
away the GPU's preference for uniform tensor shapes.

A) fold_bn2 -- In ResidualUnit.forward, conv1's output feeds batchnorm2 and
   nothing else. In eval mode a BatchNorm1d is a per-channel affine map
   y = a*z + b with a = gamma/sqrt(var+eps), b = beta - a*mu, so
   bn2(conv1(z)) == conv1'(z) for W' = a[:,None,None]*W, bias' = a*bias + b.
   batchnorm1 is NOT foldable: its input is the residual stream x = x_prev+out,
   whose producer is an add with three consumers (bn1, the residual add, and
   the 1x1 Skip conv), so there is no single preceding conv to absorb it into.

B) ValidSpliceAI -- every conv in SpliceAI uses "same" zero padding, so each of
   the 32 dilated convs computes all L input positions. SpliceAI then discards
   CL//2 = 5000 positions from each end. Zero-padding contamination propagates
   inward by exactly sum(AR*(W-1)) = 5000 per side, which is exactly the crop,
   so the retained centre window does not depend on the padded values at all.
   Running the convs unpadded and trimming the residual/skip streams as the
   receptive field grows computes only positions that survive the crop. The
   w=41/ar=25 units then evaluate ~3k positions instead of the full width.

Measured on the target cluster (Intel Xeon E5-2640 v3/v4, AVX2, 16 cores,
torch 2.13.0) against a full-precision oracle of the unmodified graph, 5-model
ensemble, 11001 bp windows:

    batch   as-shipped   valid+fold   speedup
        1      4.34/s       1.46/s      0.34x
        2      3.76/s       2.76/s      0.73x
        4      4.17/s       5.00/s      1.20x
        8      4.07/s       7.09/s      1.74x
       16      4.07/s       8.92/s      2.19x
       32      4.85/s       9.02/s      1.86x

The rewrite LOSES below batch 4 and wins above it, so `enable_cpu_graph_opt`
applies it only at batch >= CPU_GRAPH_OPT_MIN_BATCH. Max abs deviation from the
oracle over all batch sizes was 3.0e-7 (fp32 reassociation only); no position
changed at the 2-decimal precision SpliceAI scores are reported to.
"""
import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from spliceai.models.pytorch_model import ResidualUnit, Skip

logger = logging.getLogger(__name__)

# Decision thresholds for enable_cpu_graph_opt(); both were measured, see the
# table in that function's docstring.
#   MIN_BATCH: at or above this batch size the rewrite wins at any thread count.
#   MAX_THREADS_SMALL_BATCH: at or below this thread count the rewrite wins even
#     at batch 1, which is the --cpu-processes layout (many single-threaded
#     workers). Between the two the padded graph parallelises better.
CPU_GRAPH_OPT_MIN_BATCH = 4
CPU_GRAPH_OPT_MAX_THREADS_SMALL_BATCH = 2


def _bn_affine(bn):
    """Per-channel (scale, shift) of a BatchNorm1d in eval mode."""
    a = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    b = bn.bias - a * bn.running_mean
    return a, b


class FoldedResidualUnit(nn.Module):
    """A ResidualUnit with batchnorm2 absorbed into conv1's weight and bias."""

    def __init__(self, unit):
        super().__init__()
        self.batchnorm1 = copy.deepcopy(unit.batchnorm1)
        self.conv1 = copy.deepcopy(unit.conv1)
        self.conv2 = copy.deepcopy(unit.conv2)
        a, b = _bn_affine(unit.batchnorm2)
        with torch.no_grad():
            self.conv1.weight.mul_(a.view(-1, 1, 1))
            self.conv1.bias.mul_(a).add_(b)

    def forward(self, x, skip):
        out = self.conv1(torch.relu(self.batchnorm1(x)))
        out = self.conv2(torch.relu(out))
        return x + out, skip


def fold_bn2(model):
    """Return a copy of `model` with batchnorm2 folded in every ResidualUnit."""
    m = copy.deepcopy(model)
    units = nn.ModuleList()
    for u in m.residual_units:
        units.append(FoldedResidualUnit(u) if isinstance(u, ResidualUnit) else u)
    m.residual_units = units
    m.eval()
    return m


class ValidSpliceAI(nn.Module):
    """SpliceAI evaluated with unpadded convolutions over the kept window.

    Produces the same (N, 3, L - CL) softmax output as the wrapped model, up to
    fp32 reassociation, without computing positions the crop discards.
    """

    def __init__(self, model, fold=True):
        super().__init__()
        src = fold_bn2(model) if fold else copy.deepcopy(model)
        self.fold = fold
        self.initial_conv = src.initial_conv
        self.initial_skip = src.initial_skip
        self.units = nn.ModuleList()
        trims = []
        for u in src.residual_units:
            self.units.append(u)
            if isinstance(u, (ResidualUnit, FoldedResidualUnit)):
                pad = u.conv1.padding[0]
                u.conv1.padding = (0,)
                u.conv2.padding = (0,)
                # each of the two convs eats `pad` positions per side
                trims.append(2 * pad)
            else:
                trims.append(0)
        self.trims = trims
        self.final_conv = src.final_conv
        self.CL = int(model.CL)
        self.eval()

    def forward(self, x):
        if x.dim() != 3 or x.shape[1] != 4:
            raise ValueError(f"expected (N, 4, L) input, got {tuple(x.shape)}")
        if x.shape[2] < self.CL + 1:
            raise ValueError(
                f"SpliceAI-{self.CL // 2} requires length >= {self.CL + 1}, "
                f"got {x.shape[2]}")
        x = self.initial_conv(x)
        skip = self.initial_skip.conv(x)
        for u, trim in zip(self.units, self.trims):
            if trim:
                out = u.conv1(torch.relu(u.batchnorm1(x)))
                if self.fold:
                    out = u.conv2(torch.relu(out))
                else:
                    out = u.conv2(torch.relu(u.batchnorm2(out)))
                x = x[:, :, trim:-trim] + out
                skip = skip[:, :, trim:-trim]
            else:
                # Skip module: 1x1 pointwise, no receptive-field growth
                skip = u.conv(x) + skip
        return F.softmax(self.final_conv(skip), dim=1)


def enable_cpu_graph_opt(models, batch_size, num_threads=None):
    """Wrap each model in ValidSpliceAI when the configuration makes it a win.

    Returns (models, applied). `models` is returned unchanged when the rewrite
    would be slower, so callers can apply this unconditionally on CPU.

    The rewrite does strictly less arithmetic, but the padded graph gives MKLDNN
    a longer sequence to parallelise over, so which one wins depends on how much
    per-call work there is relative to the thread count. Measured on a 16-core
    Xeon E5-2640 v3, 11001 bp windows, 5-model ensemble (predictions/s):

        threads=16, batch  1: padded 4.34  vs rewrite 1.46   -> padded
        threads=16, batch  4: padded 4.17  vs rewrite 5.00   -> rewrite
        threads=16, batch 16: padded 4.07  vs rewrite 8.92   -> rewrite (2.2x)
        threads= 2, batch  1: padded 6.87  vs rewrite 9.76   -> rewrite (1.4x)
        threads= 1, batch  1: padded 7.28  vs rewrite 18.32  -> rewrite (2.5x)

    (the low-thread rows are per-process aggregates over 16 cores, i.e. the
    --cpu-processes configuration.) So the decision needs both knobs: apply
    when there is enough work per call (batch >= 4) OR when few enough threads
    are competing for it (<= 2), which covers the multi-process CPU layout where
    each worker is single-threaded.
    """
    if num_threads is None:
        num_threads = torch.get_num_threads()
    enough_batch = batch_size >= CPU_GRAPH_OPT_MIN_BATCH
    few_threads = num_threads <= CPU_GRAPH_OPT_MAX_THREADS_SMALL_BATCH
    if not (enough_batch or few_threads):
        logger.info(
            "CPU graph rewrite skipped: batch_size=%d < %d and num_threads=%d "
            "> %d -- the padded graph parallelises better in this regime",
            batch_size, CPU_GRAPH_OPT_MIN_BATCH, num_threads,
            CPU_GRAPH_OPT_MAX_THREADS_SMALL_BATCH)
        return models, False
    out = [ValidSpliceAI(m, fold=True) for m in models]
    logger.info("CPU graph rewrite active: valid convolutions + batchnorm2 "
                "folding (batch_size=%d, num_threads=%d)",
                batch_size, num_threads)
    return out, True
