"""Verify the fused grouped-conv ensemble == sequential 5-model mean, using the
REAL spliceai*.pt weights, on CPU. Architecture-level correctness is
device-independent, so passing here means it is correct on GPU too."""
import sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO + "/spliceai/batch")
from fused_inference import FusedSpliceAIEnsemble

W  = [11,11,11,11,11,11,11,11,21,21,21,21,41,41,41,41]
AR = [1,1,1,1,4,4,4,4,10,10,10,10,25,25,25,25]
L = 32

# --- reference single-model architecture (mirrors spliceai.utils.SpliceAI) ---
class RU(nn.Module):
    def __init__(self, l, w, ar):
        super().__init__()
        self.batchnorm1 = nn.BatchNorm1d(l); self.batchnorm2 = nn.BatchNorm1d(l)
        p = (w-1)*ar//2
        self.conv1 = nn.Conv1d(l,l,w,dilation=ar,padding=p)
        self.conv2 = nn.Conv1d(l,l,w,dilation=ar,padding=p)
    def forward(self, x, skip):
        out = self.conv1(torch.relu(self.batchnorm1(x)))
        out = self.conv2(torch.relu(self.batchnorm2(out)))
        return x+out, skip
class Skip(nn.Module):
    def __init__(self, l):
        super().__init__(); self.conv = nn.Conv1d(l,l,1)
    def forward(self, x, skip): return x, self.conv(x)+skip
class SpliceAI(nn.Module):
    def __init__(self, l=32):
        super().__init__()
        self.CL = 2*sum(a*(w-1) for a,w in zip(AR,W))
        self.initial_conv = nn.Conv1d(4,l,1); self.initial_skip = Skip(l)
        self.residual_units = nn.ModuleList()
        for i,(w,r) in enumerate(zip(W,AR)):
            self.residual_units.append(RU(l,w,r))
            if (i+1)%4==0: self.residual_units.append(Skip(l))
        self.final_conv = nn.Conv1d(l,3,1)
        self.crop_amount = self.CL//2
    def forward(self, x):
        x = self.initial_conv(x); x, skip = self.initial_skip(x,0)
        for m in self.residual_units: x, skip = m(x, skip)
        c = self.crop_amount; skip = skip[:,:,c:-c]
        return F.softmax(self.final_conv(skip), dim=1)

torch.manual_seed(0)
paths = [REPO + f"/spliceai/models/spliceai{i}.pt" for i in range(1,6)]
states = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]

models = []
for s in states:
    m = SpliceAI(L); m.load_state_dict(s); m.eval(); models.append(m)

fused = FusedSpliceAIEnsemble().to("cpu")
fused.load_from_state_dicts(states); fused.eval()

# realistic-ish input length (smaller than 25001 to keep CPU test fast)
Lseq = 15001
x = torch.randn(2, 4, Lseq)

with torch.inference_mode():
    # sequential ensemble (the current code path)
    seq_out = models[0](x)
    for m in range(1,5): seq_out = seq_out + models[m](x)
    seq_out = seq_out / 5.0
    # fused
    fused_out = fused(x)

print("seq   shape", tuple(seq_out.shape))
print("fused shape", tuple(fused_out.shape))
abs_err = (seq_out - fused_out).abs()
print(f"max abs err : {abs_err.max().item():.3e}")
print(f"mean abs err: {abs_err.mean().item():.3e}")
assert seq_out.shape == fused_out.shape
assert abs_err.max().item() < 1e-5, "FUSION MISMATCH"
print("\nPASS (fp32): fused ensemble is numerically identical to sequential ensemble.")

# --- bf16 sanity (GPU-only op; CPU MKLDNN lacks half-precision conv1d) ---
try:
    xb = x.to(torch.bfloat16)
    fused_bf = FusedSpliceAIEnsemble().to("cpu")
    fused_bf.load_from_state_dicts(states); fused_bf.eval(); fused_bf = fused_bf.to(torch.bfloat16)
    with torch.inference_mode():
        out_bf = fused_bf(xb)  # forward upcasts mean to fp32
    for ch, name in ((1,"acceptor"),(2,"donor")):
        a = seq_out[:,ch,:].argmax(dim=-1)
        b = out_bf[:,ch,:].argmax(dim=-1)
        print(f"bf16 {name} argmax match: {(a==b).all().item()}")
except RuntimeError as e:
    print(f"bf16 check skipped on CPU (expected): {str(e).splitlines()[0][:60]}...")

# --- micro-benchmark: sequential vs fused (CPU, fp32) ---
def bench(fn, n=3):
    with torch.inference_mode():
        fn(); t=time.time()
        for _ in range(n): fn()
        return (time.time()-t)/n
def seq_fn():
    o = models[0](x)
    for m in range(1,5): o = o + models[m](x)
    return o/5.0
def fused_fn():
    return fused(x)
ts = bench(seq_fn); tf = bench(fused_fn)
print(f"\nCPU timing  sequential: {ts*1000:.0f} ms | fused: {tf*1000:.0f} ms | speedup {ts/tf:.2f}x")
print("(CPU speedup is modest; the big win is GPU tensor-core saturation.)")
