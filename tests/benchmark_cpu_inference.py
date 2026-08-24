"""CPU inference benchmark for the SpliceAI 5-model ensemble.

Uses the REAL spliceai{1..5}.pt weights and the production model definition
(spliceai.models.pytorch_model.SpliceAI) to measure predictions/sec under the
same "sequential ensemble" code path batch.py actually runs on CPU
(spliceai/batch/batch.py::_create_ensemble_forward, CPU branch).

Run with:  python tests/benchmark_cpu_inference.py [--skip-compile]

Note: if a particular batch size shows a wildly non-linear jump in time
(much worse than ~2x the previous size), that's almost always thermal
throttling, memory pressure, or another process stealing cores on the
machine running the benchmark -- rerun on an idle machine before drawing
conclusions from it.
"""
import importlib.util
import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import pytorch_model.py directly by file path, bypassing spliceai/__init__.py
# (which calls importlib.metadata.version('spliceai') and requires the package
# to be pip-installed -- unnecessary for this standalone benchmark).
_spec = importlib.util.spec_from_file_location(
    "pytorch_model", os.path.join(REPO, "spliceai", "models", "pytorch_model.py"))
_pytorch_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pytorch_model)
SpliceAI = _pytorch_model.SpliceAI

SPLICEAI_INPUT_LENGTH = 25001
WARMUP_ITERS = 1
TIMED_ITERS = 3


def get_available_cpu_count():
    """Same logic as spliceai.batch.batch.get_available_cpu_count: respects
    cgroup/cpuset/SLURM affinity restrictions, unlike os.cpu_count()."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 8


def get_physical_cpu_count():
    """Same logic as spliceai.batch.batch.get_physical_cpu_count."""
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except ImportError:
        pass
    try:
        with open("/proc/cpuinfo") as f:
            physical_ids = set()
            cur_physical_id = "0"
            for line in f:
                if line.startswith("physical id"):
                    cur_physical_id = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    physical_ids.add((cur_physical_id, line.split(":", 1)[1].strip()))
        if physical_ids:
            return len(physical_ids)
    except OSError:
        pass
    return None


def load_models():
    paths = [os.path.join(REPO, "spliceai", "models", f"spliceai{i}.pt") for i in range(1, 6)]
    models = []
    for p in paths:
        m = SpliceAI(L=32)
        sd = torch.load(p, map_location="cpu", weights_only=True)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    return models


def sequential_ensemble(models, x):
    out = models[0](x)
    for m in range(1, 5):
        out = out + models[m](x)
    return out / 5.0


def bench(models, batch_size, num_threads, seq_len=SPLICEAI_INPUT_LENGTH,
          warmup=WARMUP_ITERS, iters=TIMED_ITERS):
    torch.set_num_threads(num_threads)
    x = torch.randn(batch_size, 4, seq_len)
    with torch.inference_mode():
        for _ in range(warmup):
            sequential_ensemble(models, x)
        t0 = time.perf_counter()
        for _ in range(iters):
            sequential_ensemble(models, x)
        elapsed = time.perf_counter() - t0
    per_batch = elapsed / iters
    preds_per_sec = (batch_size * iters) / elapsed
    return per_batch, preds_per_sec


def main():
    available_cores = get_available_cpu_count()
    physical_cores = get_physical_cpu_count()
    print(f"torch {torch.__version__}, os.cpu_count()={os.cpu_count()}, "
          f"available (affinity-aware)={available_cores}, physical={physical_cores}")
    if os.cpu_count() != available_cores:
        print(f"NOTE: os.cpu_count() ({os.cpu_count()}) != available affinity-restricted "
              f"cores ({available_cores}) -- you are likely in a cgroup/cpuset-limited "
              f"job (container/SLURM/etc). Production code now uses the affinity-aware "
              f"count; this benchmark does too.")
    models = load_models()

    # Thread scaling: cover the low end, the physical core count (production's
    # new cap), the full affinity-visible count, and a bit beyond it -- that
    # last point is what reveals the "too many threads" regression seen on
    # SMT/hyperthreaded machines.
    thread_points = sorted(set(
        t for t in [2, 4, 6, 8, physical_cores, available_cores, available_cores + 4]
        if t and 0 < t <= available_cores + 8
    ))

    print("\n=== A) Thread scaling at batch_size=4 (current CPU default) ===")
    for threads in thread_points:
        per_batch, pps = bench(models, batch_size=4, num_threads=threads)
        tag = " <- physical core count" if threads == physical_cores else \
              " <- affinity-visible count (all cores this job can use)" if threads == available_cores else ""
        print(f"threads={threads:2d}  batch=4   {per_batch*1000:8.1f} ms/batch   {pps:8.1f} preds/s{tag}")

    bench_threads = min(physical_cores or available_cores, available_cores)
    print(f"\n=== B) Batch size scaling at threads={bench_threads} (production's capped count) ===")
    for batch_size in [4, 8, 16]:
        per_batch, pps = bench(models, batch_size=batch_size, num_threads=bench_threads)
        print(f"batch={batch_size:3d}  threads={bench_threads}   {per_batch*1000:8.1f} ms/batch   {pps:8.1f} preds/s")

    if "--skip-compile" in sys.argv:
        return

    print("\n=== C) torch.compile (CPU, mode=default) vs eager, threads=capped, batch=8 ===")
    torch.set_num_threads(bench_threads)
    x = torch.randn(8, 4, SPLICEAI_INPUT_LENGTH)
    with torch.inference_mode():
        for _ in range(WARMUP_ITERS):
            sequential_ensemble(models, x)
        t0 = time.perf_counter()
        for _ in range(TIMED_ITERS):
            sequential_ensemble(models, x)
        eager_elapsed = time.perf_counter() - t0
    eager_pps = (8 * TIMED_ITERS) / eager_elapsed
    print(f"eager     {eager_elapsed/TIMED_ITERS*1000:8.1f} ms/batch   {eager_pps:8.1f} preds/s")

    try:
        compiled_fn = torch.compile(sequential_ensemble, mode="default")
        with torch.inference_mode():
            compile_start = time.perf_counter()
            compiled_fn(models, x)  # trigger compilation
            compile_time = time.perf_counter() - compile_start
            for _ in range(WARMUP_ITERS):
                compiled_fn(models, x)
            t0 = time.perf_counter()
            for _ in range(TIMED_ITERS):
                compiled_fn(models, x)
            compiled_elapsed = time.perf_counter() - t0
        compiled_pps = (8 * TIMED_ITERS) / compiled_elapsed
        print(f"compiled  {compiled_elapsed/TIMED_ITERS*1000:8.1f} ms/batch   {compiled_pps:8.1f} preds/s"
              f"   (first-call compile took {compile_time:.1f}s)")
    except Exception as e:
        print(f"torch.compile failed: {e}")


if __name__ == "__main__":
    main()
