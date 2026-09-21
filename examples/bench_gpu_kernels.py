"""Micro-benchmark of the fused SpliceAI ensemble forward on one GPU.

Answers "is the GPU doing the work efficiently?" independently of the batch
pipeline: builds the fused 5-model ensemble from spliceai/models/*.pt, times
a (T, 4, 11001) forward under several configurations, reports predictions/s
and achieved TFLOPS against the card's fp16 tensor-core peak, and prints the
top CUDA kernels from torch.profiler for the baseline.

Usage (Colab):  python examples/bench_gpu_kernels.py --T 512
"""
import argparse, time, os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from spliceai.batch.fused_inference import FusedSpliceAIEnsemble, _W, _AR, _L, _N_MODELS
from spliceai.models.pytorch_model import create_spliceai_model

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'spliceai', 'models')


def flops_per_position():
    # each residual unit: 2 convs of (L -> L, kernel W); plus 1x1 skips (negligible)
    macs = sum(2 * _L * _L * w for w in _W)
    return 2 * macs * _N_MODELS  # FLOPs, all 5 models


def load_fused(device, dtype):
    sds = []
    for i in range(1, 6):
        sd = torch.load(os.path.join(MODEL_DIR, f'spliceai{i}.pt'), map_location='cpu', weights_only=True)
        sds.append({k: v.float() for k, v in sd.items()})
    m = FusedSpliceAIEnsemble().to(device)
    m.load_from_state_dicts(sds)
    return m.eval().to(dtype)


def load_sequential(device, dtype):
    ms = []
    for i in range(1, 6):
        m = create_spliceai_model(device='cpu')
        m.load_state_dict(torch.load(os.path.join(MODEL_DIR, f'spliceai{i}.pt'), map_location='cpu', weights_only=True))
        ms.append(m.eval().to(device).to(dtype))
    def fwd(x):
        out = ms[0](x)
        for m in ms[1:]:
            out = out + m(x)
        return out / 5.0
    return fwd


def bench(name, fn, x, iters=10, warm=3):
    with torch.inference_mode():
        for _ in range(warm):
            fn(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            fn(x)
        torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    n = x.shape[0]
    tflops = flops_per_position() * x.shape[2] * n / dt / 1e12
    print(f'{name:38s} {dt*1000:8.1f} ms/chunk  {n/dt:8.1f} pred/s  {tflops:6.1f} TFLOPS')
    return dt


def profile(fn, x, top=12):
    from torch.profiler import profile as prof, ProfilerActivity
    with torch.inference_mode():
        fn(x); torch.cuda.synchronize()
        with prof(activities=[ProfilerActivity.CUDA]) as p:
            fn(x); torch.cuda.synchronize()
    print(p.key_averages().table(sort_by='cuda_time_total', row_limit=top))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--T', type=int, default=512)
    ap.add_argument('--L', type=int, default=11001)
    ap.add_argument('--no-profile', action='store_true')
    a = ap.parse_args()

    dev = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    print(torch.__version__, torch.version.cuda, 'cudnn', torch.backends.cudnn.version())
    print(torch.cuda.get_device_name(0))
    print(f'FLOPs/prediction at L={a.L}: {flops_per_position()*a.L/1e9:.1f} GFLOP')

    x16 = torch.randint(0, 2, (a.T, 4, a.L), device=dev).to(torch.float16)

    fused16 = load_fused(dev, torch.float16)
    base = bench('fused fp16 (as shipped)', fused16, x16)

    from spliceai.batch.fused_valid import wrap_conv_impl
    for impl in ('valid', 'valid2d'):
        v = wrap_conv_impl(fused16, impl).eval()
        bench(f'fused fp16 conv_impl={impl}', v, x16)
        try:
            vc = torch.compile(v, mode='default')
            bench(f'fused fp16 conv_impl={impl} + compile', vc, x16, iters=5)
        except Exception as e:
            print(f'compile ({impl}) failed:', e)

    fusedbf = load_fused(dev, torch.bfloat16)
    bench('fused bf16', fusedbf, x16.to(torch.bfloat16))

    seq16 = load_sequential(dev, torch.float16)
    bench('sequential 5 models fp16', seq16, x16)

    try:
        comp = torch.compile(fused16, mode='max-autotune-no-cudagraphs')
        bench('fused fp16 + compile max-autotune', comp, x16, iters=5)
    except Exception as e:
        print('compile max-autotune failed:', e)

    for T in (128, 256, 1024):
        if T == a.T: continue
        try:
            xs = torch.randint(0, 2, (T, 4, a.L), device=dev).to(torch.float16)
            bench(f'fused fp16 T={T}', fused16, xs)
            del xs
        except torch.cuda.OutOfMemoryError:
            print(f'fused fp16 T={T}: OOM'); torch.cuda.empty_cache()

    if not a.no_profile:
        print('\nTop CUDA kernels, fused fp16 baseline (padded):')
        profile(fused16, x16)
        print('\nTop CUDA kernels, conv_impl=valid2d:')
        profile(wrap_conv_impl(fused16, 'valid2d').eval(), x16)


if __name__ == '__main__':
    main()
