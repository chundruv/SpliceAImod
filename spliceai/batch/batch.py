# Original source code modified to add prediction batching support by Invitae in 2021.
# Modifications copyright (c) 2021 Invitae Corporation.

# Converted to PyTorch with FP16/BF16 support
# OPTIMIZED VERSION v2 - Fixed timing, added torch.compile, channels_last

import os
import sys
import subprocess
import multiprocessing

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass


def get_num_gpus():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0 and result.stdout:
            return len(result.stdout.strip().split('\n'))
    except:
        pass
    return 1


def get_available_cpu_count():
    """Return the number of CPUs this process can actually use.

    ``multiprocessing.cpu_count()`` / ``os.cpu_count()`` report the *machine's*
    total core count, ignoring any cgroup/cpuset restriction the process is
    confined to (e.g. a SLURM allocation with ``--cpus-per-task=1`` on a
    16-core node, or a Docker container started with ``--cpuset-cpus``). On
    such systems this mismatch causes us to spawn far more OMP/MKL threads
    than cores actually available, which doesn't just fail to help -- it makes
    throughput *flat* regardless of thread/batch settings, because all those
    threads are time-sliced across the same tiny cpuset with pure scheduling
    overhead and no added parallelism.

    ``os.sched_getaffinity(0)`` reports the actual set of CPUs the calling
    process is allowed to run on, which correctly reflects cpuset/affinity
    restrictions. It's Linux-only, so fall back to the machine-wide count on
    platforms (e.g. macOS) where it doesn't exist.

    NOTE: duplicated (rather than imported) from spliceai.utils -- this module
    deliberately sets OMP_NUM_THREADS/MKL_NUM_THREADS *before* importing torch
    (see below), and importing spliceai.utils this early would pull torch in
    prematurely, defeating that ordering.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return multiprocessing.cpu_count()


def get_physical_cpu_count():
    """Return the number of *physical* CPU cores, or None if undetermined.

    Benchmarking (tests/benchmark_cpu_inference.py) shows CPU throughput for
    this model peaks around the physical core count and then *regresses* past
    it (e.g. one 16-logical-core SLURM node: 10 threads -> 2539ms/batch,
    16 threads -> 2968ms/batch -- 16 is slower than 10). SpliceAI's dilated
    convolutions are narrow (32-160 channels) and don't have enough
    independent work to benefit from hyperthread siblings -- oversubscribing
    logical/SMT threads just adds synchronization and cache-contention
    overhead. So the CPU thread count should be capped at the physical core
    count, not the logical/affinity-visible one.

    Tries ``psutil`` first (cross-platform), then falls back to parsing
    ``/proc/cpuinfo`` on Linux (no extra dependency). Returns None if neither
    works (e.g. macOS without psutil), in which case callers should just use
    the logical/affinity core count as before.
    """
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except ImportError:
        pass

    try:
        with open('/proc/cpuinfo') as f:
            physical_ids = set()
            cur_physical_id = '0'
            for line in f:
                if line.startswith('physical id'):
                    cur_physical_id = line.split(':', 1)[1].strip()
                elif line.startswith('core id'):
                    core_id = line.split(':', 1)[1].strip()
                    physical_ids.add((cur_physical_id, core_id))
        if physical_ids:
            return len(physical_ids)
    except (OSError, IOError):
        pass

    return None


#: Upper bound on CPU threads handed to a single GPU worker process. See the
#: comment in calculate_optimal_threads().
MAX_GPU_FEEDER_THREADS = 8


def calculate_optimal_threads(cpu_count, workers_per_gpu, num_gpus,
                              is_cpu=False, cpu_processes=1):
    if is_cpu:
        # Pure CPU inference. `num_gpus` is 1 here (get_num_gpus() falls back to
        # 1 when nvidia-smi is absent), so it must NOT be used to divide the
        # cores: with --cpu-processes > 1 every worker would claim all
        # `cpu_count` threads and the node would be oversubscribed N-fold.
        # Divide by the actual number of sibling CPU workers instead. There is
        # no GPU to feed and nothing else competing for cores. Unlike the GPU
        # branches
        # below, don't subtract `reserved_cores` here -- that reservation
        # exists to leave headroom for GPU-feeding/other processes, and on a
        # cgroup/cpuset-restricted node it costs real throughput: benchmarking
        # on a 12-available/16-physical-core SLURM allocation showed 12
        # threads (all available cores) beat every smaller thread count
        # tested (2539-2934ms at <=8 threads vs 2579ms at 12), so reserving
        # 2 of the 12 available cores would have left the peak on the table.
        #
        # But benchmarking also shows more isn't always better: throughput
        # regresses past the physical/available core count (SMT/hyperthread
        # siblings don't have enough independent work in these narrow dilated
        # convs, so oversubscribing them just adds contention -- 16 threads
        # was slower than 12 in that same test). Cap at the physical core
        # count when we can determine it, and always cap at the affinity-
        # visible cpu_count.
        total_workers = max(1, cpu_processes)
        # With one worker per core, 1 thread each is correct and beats 2 (the
        # old floor) -- 16x1 measured 7.28 preds/s aggregate vs 4.43 for 1x16.
        cpu_threads = max(1, cpu_count // total_workers)
        physical_cores = get_physical_cpu_count()
        if physical_cores:
            cpu_threads = min(cpu_threads, max(1, physical_cores // total_workers))
        return cpu_threads

    reserved_cores = max(2, cpu_count // 16)
    available_cores = cpu_count - reserved_cores
    total_workers = num_gpus * workers_per_gpu
    threads_per_worker = max(1, available_cores // total_workers)

    # A GPU worker's CPU threads only feed the device (batch load, H2D/D2H
    # marshalling); the convolutions themselves run on the GPU. Past a handful
    # of threads per worker the extra ones contend rather than help, so cap at
    # MAX_GPU_FEEDER_THREADS regardless of how many cores the node has. This
    # replaces a per-GPU-model lookup table (H200 48, H100 32, A100 16, ...)
    # whose values were never measured against each other; every GPU not named
    # in that table -- the common case -- already landed on this same cap.
    max_threads = min(MAX_GPU_FEEDER_THREADS, available_cores // total_workers)

    # Floor at 1, not 2. When available_cores < 2 * total_workers, the old
    # max(2, ...) floor handed every worker 2 threads regardless of how many
    # cores existed: a 4-core allocation running 4 workers got 8 threads on 2
    # available cores (4x oversubscription), and 8 cores with 2 GPUs x 4
    # workers got 16 threads on 6 cores (2.7x). Oversubscribed BLAS/ATen
    # threads on the narrow dilated convs in this model contend rather than
    # parallelise -- the same effect measured in the CPU path above, where 16x1
    # beat 1x16. One thread per worker is the correct answer when there is less
    # than one core each; it is never faster to promise two.
    return max(1, min(threads_per_worker, max_threads))


cpu_count = get_available_cpu_count()
num_gpus = get_num_gpus()

# NOTE: a `spliceai.batch.worker_optimizer` module providing WorkerCPUPinner
# (core affinity) and PerformanceMonitor was imported here behind a
# try/except ImportError. No such module has ever existed in this tree, so the
# except branch always ran and the guarded block below it was permanently dead
# — CPU pinning never happened and no worker performance was ever monitored,
# silently. Removed rather than left as a false advertisement; if core affinity
# is wanted, the measured wins are in spliceai/models/cpu_graph_opt.py and the
# process/thread layout documented in OPTIMIZATIONS.md.

# Thread limits must be set BEFORE torch is imported -- OMP sizes its pool at
# library load time and ignores later changes to the environment. This is only
# a placeholder: main() recomputes the real value with
# calculate_optimal_threads() (which knows --workers-per-gpu / --cpu-processes,
# unavailable here) and applies it via torch.set_num_threads().
#
# This replaced a lookup table keyed on the nvidia-smi GPU name that assigned a
# different thread count to each card model. Nothing about a GPU's model
# number determines how many CPU threads its feeder process wants, and every
# unlisted card fell through to a default anyway.
_startup_threads = max(1, min(MAX_GPU_FEEDER_THREADS, cpu_count // max(1, num_gpus)))
os.environ['OMP_NUM_THREADS'] = str(_startup_threads)
os.environ['MKL_NUM_THREADS'] = str(_startup_threads)

import logging
import time
import numpy as np

import torch
print(f"Worker: PyTorch {torch.__version__}", flush=True)

# CRITICAL: Enable cudnn optimizations BEFORE any model creation
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# TF32 is a hardware feature of Ampere and later. Requesting it on older cards
# is not an error -- the flag simply has no tensor cores to act on and the
# kernels run in fp32 as before -- so it can be set unconditionally instead of
# being gated on a list of GPU model names.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# NCCL optimizations for multi-GPU setups
os.environ.setdefault('NCCL_IB_DISABLE', '0')  # Enable InfiniBand if available
os.environ.setdefault('NCCL_P2P_LEVEL', 'NVL')  # Enable NVLink P2P
os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')  # Optimize kernel launches

import pickle
import gc
import socket
import argparse

from spliceai.utils import Annotator


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-R', '--reference', required=True)
    parser.add_argument('-A', '--annotation', required=True)
    parser.add_argument('-T', '--pytorch_batch_size', type=int)
    parser.add_argument('-V', '--verbose', action='store_true')
    parser.add_argument('-t', '--tmpdir', type=str, default='/tmp/')
    parser.add_argument('-d', '--device', type=str, required=True)
    parser.add_argument('-w', '--worker_id', type=int, default=0)
    parser.add_argument('-G', '--gpus', type=str, default='all')
    parser.add_argument('--gpu-profile-interval', type=int, default=100)
    parser.add_argument('--precision', type=str, default='auto', choices=['auto', 'fp32', 'fp16', 'bf16'])
    parser.add_argument('--compile', action='store_true', help='Use torch.compile() - can improve speed but increases warmup time')
    parser.add_argument('--no-cuda-graphs', action='store_true', help='Disable CUDA graphs')
    parser.add_argument('--conv-impl', default='padded', choices=['padded', 'valid', 'valid2d'])
    parser.add_argument('--workers-per-gpu', type=int, default=1)
    # How many sibling CPU worker processes exist. Used only to divide the
    # available cores between them (see calculate_optimal_threads).
    parser.add_argument('--cpu-processes', type=int, default=1)
    return parser.parse_args()


def main():
    args = get_options()

    is_cpu_mode = 'cpu' in args.device.lower()
    workers_per_gpu = getattr(args, 'workers_per_gpu', 1)
    cpu_processes = max(1, getattr(args, 'cpu_processes', 1) or 1)
    optimal_threads = calculate_optimal_threads(
        cpu_count, workers_per_gpu, num_gpus,
        is_cpu=is_cpu_mode, cpu_processes=cpu_processes)
    
    os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
    os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
    if is_cpu_mode:
        # Encourage the OS scheduler to keep each OMP thread pinned to one
        # core instead of migrating it around -- migration mid-run loses
        # cache locality and can look exactly like the "too many threads"
        # regression benchmarking showed past the physical core count.
        os.environ.setdefault('OMP_PROC_BIND', 'close')
        os.environ.setdefault('OMP_PLACES', 'cores')
        # We only ever run one sequential forward pass at a time (no
        # torch.jit.fork / inter-op parallelism happens in this code path),
        # so the inter-op thread pool -- which otherwise defaults to the same
        # size as the intra-op pool -- would just be idle threads reserved
        # for nothing. Pin it to 1 to avoid the extra thread-pool overhead.
        torch.set_num_interop_threads(1)
    torch.set_num_threads(optimal_threads)

    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(name)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    logger = logging.getLogger(__name__)

    try:
        if is_cpu_mode:
            device = torch.device('cpu')
        else:
            if not torch.cuda.is_available():
                logger.error("CUDA not available!")
                sys.exit(1)
            device = torch.device('cuda:0')
            
            gpu_props = torch.cuda.get_device_properties(0)
            logger.info(f"Worker {args.worker_id}: {gpu_props.name} ({gpu_props.total_memory / 1024**3:.1f}GB)")
            logger.info(f"  Compute: {gpu_props.major}.{gpu_props.minor}, TF32: {torch.backends.cuda.matmul.allow_tf32}")

        worker = VCFPredictionBatch(args=args, logger=logger, device=device, worker_id=args.worker_id)
        worker.process_batches()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


class VCFPredictionBatch:
    def __init__(self, args, logger, device, worker_id=0):
        self.args = args
        self.ann = None
        self.pytorch_batch_size = args.pytorch_batch_size
        self.tmpdir = args.tmpdir
        self.device_str = args.device
        self.device = device
        self.logger = logger
        self.worker_id = worker_id
        self.precision = getattr(args, 'precision', 'auto')
        self.compile_model = getattr(args, 'compile', False)
        self.socket_client = None
        # Per-batch torch.cuda.synchronize() calls exist ONLY to make the
        # prep/infer/xfer timing breakdown accurate. They serialize the H2D
        # copy, compute and D2H copy and prevent overlap. Off by default; the
        # final .cpu() copy is itself a blocking sync, so correctness holds.
        self.accurate_timing = getattr(args, 'verbose', False)

        # Performance monitoring: see the note at the worker_optimizer import
        # above. The module never existed, so this was always None.
        self.perf_monitor = None

        self._setup_precision()

        self.warmup_done = False
        self.compiled_model = None  # Will hold the compiled ensemble
        self.use_cuda_graphs = False
        self.cuda_graphs = {}
        self.static_inputs = {}
        self.static_outputs = {}
        self._graph_capture_failed = set()

        # Profiling
        self.gpu_profile_interval = getattr(args, 'gpu_profile_interval', 100)
        self.total_batches_processed = 0
        self.gpu_profile_counters = {
            'total_time': 0.0, 'inference_time': 0.0, 'transfer_time': 0.0,
            'total_predictions': 0
        }
        self.last_gpu_profile_log = time.time()

        # Output directory
        device_suffix = self.device_str.replace(':', '_').replace('/', '_').replace('cuda', 'GPU')
        self.predictions_dir = os.path.join(self.tmpdir, f"spliceai_preds.{device_suffix}_w{self.worker_id}")
        os.makedirs(self.predictions_dir, exist_ok=True)
        self.logger.info(f"Worker {self.worker_id}: Output dir: {self.predictions_dir}")

    def _setup_precision(self):
        """Determine precision mode - models will be converted in Annotator._load_models"""
        is_cpu = self.device.type == 'cpu'
        
        if is_cpu:
            if self.precision == 'auto':
                self.precision = 'fp32'
            self.logger.info(f"Worker {self.worker_id}: CPU mode - {self.precision.upper()}")
            return
        
        if self.precision == 'auto':
            # Ask the device what it supports rather than matching its model
            # name against a table: torch.cuda.is_bf16_supported() is true on
            # Ampere and later, and cards without bf16 tensor cores (Volta,
            # Turing) get fp16, which they do accelerate.
            if torch.cuda.is_bf16_supported():
                self.precision = 'bf16'
                self.logger.info(f"Worker {self.worker_id}: Auto-selected BF16 (device supports bfloat16)")
            else:
                self.precision = 'fp16'
                self.logger.info(f"Worker {self.worker_id}: Auto-selected FP16 (device has no bfloat16 support)")
        else:
            self.logger.info(f"Worker {self.worker_id}: Using precision: {self.precision}")

    def _create_ensemble_forward(self):
        """Create an optimized ensemble forward function.

        On GPU we fuse the 5 models into a single grouped-convolution network
        (groups=5, 160 channels). A grouped conv computes exactly N independent
        convolutions, so the result is mathematically identical to averaging the
        5 separate models -- but it runs as ONE kernel sequence with 5x the
        arithmetic intensity, which saturates tensor cores far better than five
        sequential 32-channel forwards. Verified equivalent (max abs err ~3e-7)
        against the real weights.

        On CPU the fused path is *slower* (no tensor cores; wide grouped convs
        parallelize worse than MKLDNN's separate small convs), so we keep the
        sequential ensemble there.
        """
        models = self.ann.models

        def ensemble_forward(x):
            out = models[0](x)
            for m in range(1, 5):
                out = out + models[m](x)
            return out / 5.0

        self.fused_model = None
        if self.device.type != 'cuda':
            # CPU: the fused grouped-conv path is slower (see docstring), but two
            # algebraically exact graph rewrites are worth applying -- folding
            # batchnorm2 into conv1, and replacing "same" padding with valid
            # convolutions so no layer computes positions the CL//2 crop throws
            # away. Measured 2.19x at batch 16 on Xeon E5-2640 v3 (AVX2, 16
            # cores). Which graph wins depends on batch size AND thread count,
            # so both are passed and enable_cpu_graph_opt returns the models
            # untouched in the regime where the padded graph is faster. Max abs
            # deviation from the unmodified graph over batches 1-32 was 3.0e-7;
            # no score changed at the 2 d.p. SpliceAI reports.
            try:
                from spliceai.models.cpu_graph_opt import enable_cpu_graph_opt
                batch_size = self.pytorch_batch_size or 1
                opt_models, applied = enable_cpu_graph_opt(
                    models, batch_size, num_threads=torch.get_num_threads())
                if applied:
                    models = opt_models

                    def ensemble_forward(x):
                        out = models[0](x)
                        for m in range(1, len(models)):
                            out = out + models[m](x)
                        return out / float(len(models))
            except Exception as e:
                self.logger.warning(
                    f"Worker {self.worker_id}: CPU graph rewrite unavailable "
                    f"({e}); using the as-shipped graph")
            return ensemble_forward

        try:
            from spliceai.batch.fused_inference import FusedSpliceAIEnsemble
            model_dtype = next(models[0].parameters()).dtype
            # Build in fp32 (lossless from the loaded half/bf16 weights), then
            # cast to the inference dtype. copy_ upcasts half->fp32 exactly.
            state_dicts = [{k: v.detach().float() for k, v in m.state_dict().items()}
                           for m in models]
            fused = FusedSpliceAIEnsemble().to(self.device)
            fused.load_from_state_dicts(state_dicts)
            fused.eval()
            if model_dtype in (torch.float16, torch.bfloat16):
                fused = fused.to(model_dtype)
            conv_impl = getattr(self.args, 'conv_impl', 'padded') or 'padded'
            from spliceai.batch.fused_valid import wrap_conv_impl
            runner = wrap_conv_impl(fused, conv_impl)
            runner.eval()
            self.fused_model = runner
            self.logger.info(
                f"Worker {self.worker_id}: FUSED grouped-conv ensemble active "
                f"(groups=5, {fused.wide} channels, dtype={model_dtype}, conv_impl={conv_impl})")

            def fused_forward(x):
                return runner(x)

            return fused_forward
        except Exception as e:
            self.logger.warning(
                f"Worker {self.worker_id}: fused ensemble build failed ({e}); "
                f"falling back to sequential ensemble")
            return ensemble_forward

    def process_batches(self):
        s = None
        try:
            s = socket.socket()
            host = socket.gethostname()
            port = 54677

            for attempt in range(10):
                try:
                    s.connect((host, port))
                    break
                except ConnectionRefusedError:
                    if attempt < 9:
                        time.sleep(2)
                    else:
                        raise

            self.socket_client = s

            s.send(b"Ready for work...\n")  # Include newline for proper message framing

            try:
                s.settimeout(2.0)
                s.recv(2048)  # Handshake
            except socket.timeout:
                pass
            finally:
                s.settimeout(None)

            if not self.ann:
                self.logger.info(f"Worker {self.worker_id}: Loading models...")
                load_start = time.time()
                self.ann = Annotator(
                    self.args.reference, 
                    self.args.annotation, 
                    cpu=(self.device.type == 'cpu'),
                    load_models=True,
                    precision=self.precision,
                    compile_model=False,  # We'll compile the ensemble instead
                    device=self.device
                )
                self.logger.info(f"Worker {self.worker_id}: Models loaded in {time.time() - load_start:.2f}s")
                
                if self.device.type == 'cuda':
                    self.logger.info(f"Worker {self.worker_id}: Memory after model load:")
                    self.logger.info(f"  Allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
                    self.logger.info(f"  Reserved: {torch.cuda.memory_reserved()/1024**3:.2f}GB")
                    self.logger.info(f"  Free: {torch.cuda.get_device_properties(0).total_memory/1024**3 - torch.cuda.memory_allocated()/1024**3:.2f}GB")
                    
                # Create and optionally compile ensemble
                self.ensemble_forward = self._create_ensemble_forward()
                use_compile = getattr(self.args, 'compile', False)

                if use_compile:
                    try:
                        self.logger.info(f"Worker {self.worker_id}: Compiling ensemble with torch.compile...")
                        compile_start = time.time()
                        # Use 'default' mode to avoid CUDA graph memory overhead
                        # 'reduce-overhead' uses CUDA graphs which consume ~2x batch memory
                        # 'max-autotune' takes longer to compile but may be faster
                        self.ensemble_forward = torch.compile(
                            self.ensemble_forward, 
                            mode='default',  # Changed from 'reduce-overhead' to save memory
                            fullgraph=False  # Allow graph breaks if needed
                        )
                        self.logger.info(f"Worker {self.worker_id}: Compilation setup in {time.time() - compile_start:.2f}s")
                    except Exception as e:
                        self.logger.warning(f"Worker {self.worker_id}: torch.compile failed: {e}")
                
                self._warmup_models()

            batch_count = 0
            msg = "Ready for work..."
            hold_on_count = 0
            timeout_count = 0  # Track consecutive timeouts
            recv_buffer = b""
            pending_messages = []

            while True:
                if not pending_messages:
                    try:
                        s.send((msg + '\n').encode('utf-8'))  # Add newline for proper framing
                    except (BrokenPipeError, ConnectionResetError):
                        self.logger.info(f"Worker {self.worker_id}: Connection lost, exiting.")
                        break

                    try:
                        s.settimeout(10.0)
                        recv_bytes = s.recv(2048)
                        s.settimeout(None)
                        timeout_count = 0  # Reset on successful receive
                        
                        if not recv_bytes:
                            self.logger.info(f"Worker {self.worker_id}: Server closed connection.")
                            break
                        
                        recv_buffer += recv_bytes
                        lines = recv_buffer.split(b'\n')
                        recv_buffer = lines[-1]
                        pending_messages = [l.decode('utf-8').strip() for l in lines[:-1] if l.strip()]

                        if not pending_messages:
                            continue

                    except socket.timeout:
                        timeout_count += 1
                        if timeout_count > 10:  # 10 consecutive timeouts = 100 seconds
                            self.logger.info(f"Worker {self.worker_id}: Too many timeouts, assuming work complete. {batch_count} batches.")
                            break
                        msg = 'Ready for work...'
                        continue
                    except (BrokenPipeError, ConnectionResetError):
                        self.logger.info(f"Worker {self.worker_id}: Connection error, exiting.")
                        break

                res = pending_messages.pop(0)

                if res == 'Hold On':
                    hold_on_count += 1
                    msg = 'Ready for work...'
                    if not pending_messages:
                        time.sleep(min(0.02 * (2 ** min(hold_on_count - 1, 4)), 0.2))  # Cap at 0.2 seconds
                    continue
                else:
                    hold_on_count = 0

                if res == 'Finished':
                    self.logger.info(f"Worker {self.worker_id}: Done. {batch_count} batches processed.")
                    break
                elif res == 'BATCH_DATA':
                    try:
                        s.settimeout(60.0)
                        len_bytes = recv_buffer
                        while len(len_bytes) < 8:
                            len_bytes += s.recv(8 - len(len_bytes))
                        data_len = int.from_bytes(len_bytes[:8], byteorder='big')
                        pickled_data = len_bytes[8:]
                        while len(pickled_data) < data_len:
                            pickled_data += s.recv(min(262144, data_len - len(pickled_data)))
                        recv_buffer = b''
                        s.settimeout(None)
                        data = pickle.loads(pickled_data[:data_len])
                        batch_count += 1
                        self._process_batch(data['tensor_size'], data['batch_ix'],
                                                data['data'], data['length'], batch_count)
                        msg = 'Ready for work...'
                        del data
                    except Exception as e:
                        self.logger.error(f"Worker {self.worker_id}: Batch error: {e}")
                        s.settimeout(None)
                        msg = 'Ready for work...'
                else:
                    try:
                        ppath = os.path.join(self.tmpdir, res)
                        if not os.path.exists(ppath):
                            msg = 'Ready for work...'
                            continue
                        with open(ppath, 'rb') as p:
                            data = pickle.load(p)
                        os.unlink(ppath)
                        batch_count += 1
                        self._process_batch(data['tensor_size'], data['batch_ix'],
                                                data['data'], data['length'], batch_count)
                        msg = 'Ready for work...'
                        del data
                    except Exception as e:
                        self.logger.error(f"Worker {self.worker_id}: Error: {e}")
                        msg = 'Ready for work...'

            try:
                s.send(b'Done')
            except:
                pass

        except Exception as e:
            self.logger.error(f"Worker {self.worker_id}: Fatal: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        finally:
            if self.perf_monitor:
                self.perf_monitor.stop()
            if s:
                try:
                    s.close()
                except:
                    pass

    def _warmup_models(self):
        """Warmup with actual batch sizes to trigger torch.compile and cuDNN autotuning"""
        if self.warmup_done:
            return

        is_cpu = self.device.type == 'cpu'
        if is_cpu:
            self.logger.info(f"Worker {self.worker_id}: Skipping warmup on CPU")
            self.warmup_done = True
            return

        self.logger.info(f"Worker {self.worker_id}: Warming up...")
        warmup_start = time.time()

        # Window the batch producers actually emit: 10000 bp of context plus
        # the +/- distance region. The old hardcoded 25001 never matched a real
        # batch at -D 500 (11001), so the graph captured here was never replayed
        # and every batch ran eager.
        SPLICEAI_INPUT_LENGTH = 10000 + 2 * int(getattr(self.args, 'distance', 500) or 500) + 1
        batch_size = self.pytorch_batch_size or 128
        
        # Get model dtype
        model_dtype = next(self.ann.models[0].parameters()).dtype
        self.logger.info(f"Worker {self.worker_id}: Model dtype = {model_dtype}")

        # Initialize CUDA graph storage
        self.cuda_graphs = {}
        self.static_inputs = {}
        self.static_outputs = {}
        self._graph_capture_failed = set()
        
        # CUDA graphs are on by default on GPU; --no-cuda-graphs disables them.
        # Capture is LAZY: _process_batch captures a graph the first time it
        # sees a full-size chunk, keyed on the real (batch, seq_len). That works
        # with and without --compile (a compiled callable is capturable once it
        # has been traced), and it cannot key on the wrong shape.
        no_cuda_graphs_flag = getattr(self.args, 'no_cuda_graphs', False)
        self.use_cuda_graphs = (not is_cpu) and torch.cuda.is_available() and not no_cuda_graphs_flag
        
        # Check if we're using torch.compile - skip warmup to avoid OOM
        # torch.compile will JIT compile on first real batch instead
        use_compile = getattr(self.args, 'compile', False)
        if use_compile:
            self.logger.info(f"Worker {self.worker_id}: Skipping warmup (torch.compile will JIT on first batch)")
            self.logger.info(f"Worker {self.worker_id}: CUDA graphs {'enabled (lazy capture)' if self.use_cuda_graphs else 'disabled'}")
            self.warmup_done = True
            self.logger.info(f"Worker {self.worker_id}: Warmup done in {time.time() - warmup_start:.2f}s")
            return

        with torch.inference_mode():
            for i in range(2):  # Fewer warmup iterations
                try:
                    dummy_input = torch.randn(
                        batch_size, 4, SPLICEAI_INPUT_LENGTH,
                        dtype=model_dtype,
                        device=self.device
                    )
                    
                    _ = self.ensemble_forward(dummy_input)
                    
                    if not is_cpu:
                        torch.cuda.synchronize()
                    
                    del dummy_input
                except Exception as e:
                    self.logger.warning(f"Worker {self.worker_id}: Warmup iter {i} failed: {e}")

        # Clean up after warmup to free compile cache memory
        gc.collect()
        if not is_cpu:
            torch.cuda.empty_cache()
            # Log memory after cleanup
            try:
                mem_used = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                self.logger.info(f"Worker {self.worker_id}: Post-warmup memory: {mem_used:.1f}GB allocated, {mem_reserved:.1f}GB reserved")
            except:
                pass

        self.logger.info(f"Worker {self.worker_id}: CUDA graphs {'enabled (lazy capture)' if self.use_cuda_graphs else 'disabled'}")
        self.warmup_done = True
        self.logger.info(f"Worker {self.worker_id}: Warmup done in {time.time() - warmup_start:.2f}s")

    def _capture_cuda_graph(self, batch_size, seq_length, dtype, sample=None):
        """Capture a CUDA graph for one (batch, seq_len) shape.

        Warm-up runs on a side stream before capture, as the PyTorch docs
        require, so cuDNN autotuning (cudnn.benchmark) and any torch.compile
        tracing happen outside the captured region. `sample` (a real chunk)
        seeds the static input so the warm-up sees realistic data.
        """
        graph_key = (batch_size, seq_length)

        static_input = torch.zeros(batch_size, 4, seq_length, dtype=dtype, device=self.device)
        if sample is not None:
            static_input.copy_(sample)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = self.ensemble_forward(static_input)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        # Return the warm-up's cached activations to the driver so the graph's
        # private pool can reuse that physical memory instead of doubling it.
        gc.collect()
        torch.cuda.empty_cache()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_output = self.ensemble_forward(static_input)
        torch.cuda.synchronize()

        self.cuda_graphs[graph_key] = g
        self.static_inputs[graph_key] = static_input
        self.static_outputs[graph_key] = static_output

    def _maybe_capture(self, graph_key, chunk, model_dtype):
        """Lazily capture a graph for a full-size chunk the first time it is seen.

        Only full chunks (== -T) are captured: the ragged last chunk of a batch
        would cost a capture for a shape seen once per shard. Failure (usually
        OOM during capture) is remembered per shape and that shape runs eager.
        """
        if graph_key in self.cuda_graphs or graph_key in self._graph_capture_failed:
            return
        batch_size, seq_length = graph_key
        t0 = time.time()
        try:
            self._capture_cuda_graph(batch_size, seq_length, model_dtype, sample=chunk)
            mem = torch.cuda.memory_allocated() / 1024**3
            self.logger.info(f"Worker {self.worker_id}: CUDA graph captured for shape {graph_key} "
                             f"in {time.time() - t0:.1f}s ({mem:.1f}GB allocated)")
        except Exception as e:
            self._graph_capture_failed.add(graph_key)
            self.cuda_graphs.pop(graph_key, None)
            self.static_inputs.pop(graph_key, None)
            self.static_outputs.pop(graph_key, None)
            gc.collect()
            torch.cuda.empty_cache()
            self.logger.warning(f"Worker {self.worker_id}: CUDA graph capture failed for shape {graph_key}, "
                                f"running eager for this shape: {e}")

    def _process_batch(self, tensor_size, batch_ix, prediction_batch, nr_preds, batch_count):
        """Optimized batch processing with CUDA graph support"""
        total_start = time.time()
        is_cpu = self.device.type == 'cpu'

        if batch_count == 1:
            self.logger.info(f"Worker {self.worker_id}: First batch - shape {prediction_batch.shape}, size {prediction_batch.nbytes / 1024**3:.2f}GB")
        self.total_batches_processed += 1
        self.gpu_profile_counters['total_predictions'] += nr_preds

        # Determine model dtype from actual model parameters
        model_dtype = next(self.ann.models[0].parameters()).dtype

        # Data preparation - convert to model dtype directly
        prep_start = time.time()
        if isinstance(prediction_batch, np.ndarray):
            # Ensure contiguous and transpose: (N, L, 4) -> (N, 4, L)
            prediction_batch = np.ascontiguousarray(np.transpose(prediction_batch, (0, 2, 1)))
            input_tensor = torch.from_numpy(prediction_batch).to(
                device=self.device, 
                dtype=model_dtype,
                non_blocking=True
            )
        else:
            input_tensor = prediction_batch.permute(0, 2, 1).contiguous().to(
                device=self.device, 
                dtype=model_dtype,
                non_blocking=True
            )
        
        total_batch_size = input_tensor.shape[0]
        seq_length = input_tensor.shape[2]
        pytorch_batch_size = getattr(self.args, 'pytorch_batch_size', total_batch_size)
        if pytorch_batch_size is None or pytorch_batch_size <= 0:
            pytorch_batch_size = total_batch_size

        # Sync only when accurate per-phase timing is requested (verbose).
        # Otherwise let the async H2D copy overlap with compute.
        if not is_cpu and self.accurate_timing:
            torch.cuda.synchronize()
        prep_time = time.time() - prep_start

        if not is_cpu and self.accurate_timing:
            torch.cuda.synchronize()

        infer_start = time.time()
        
        results = []
        with torch.inference_mode():
            for i in range(0, total_batch_size, pytorch_batch_size):
                chunk = input_tensor[i:i+pytorch_batch_size]
                chunk_batch_size = chunk.shape[0]
                graph_key = (chunk_batch_size, seq_length)

                if self.use_cuda_graphs and chunk_batch_size == pytorch_batch_size:
                    self._maybe_capture(graph_key, chunk, model_dtype)

                # Try to use CUDA graph if available and batch size matches
                if self.use_cuda_graphs and graph_key in self.cuda_graphs:
                    # Copy input to static buffer and replay graph
                    self.static_inputs[graph_key].copy_(chunk)
                    self.cuda_graphs[graph_key].replay()
                    result = self.static_outputs[graph_key].clone()
                else:
                    # Fall back to regular execution
                    result = self.ensemble_forward(chunk)
                
                # Transpose on GPU: (N, 3, L) -> (N, L, 3)
                result = result.permute(0, 2, 1).contiguous()
                results.append(result)
            
            result = torch.cat(results, dim=0)

        if not is_cpu and self.accurate_timing:
            torch.cuda.synchronize()
        infer_time = time.time() - infer_start
        self.gpu_profile_counters['inference_time'] += infer_time

        # Transfer to CPU. Storage dtype defaults to fp32.
        #
        # This was unconditionally fp16, justified by "the output is rounded to 2
        # decimals downstream". That is false in this fork: utils.py builds the
        # 'Sites with Raw Score > 0.5' lists with a STRICT y > 0.5 test on exactly
        # these stored values, and MIN_CANONICAL_SS_SCORE / MIN_NEW_SS_SCORE /
        # COMPETING_MIN_ABS (all 0.50) gate the splice-event classification the
        # same way. fp16 spacing at 0.5 is 4.883e-4, and rounding is one-sided
        # there: every fp32 value in (0.5, 0.500244] becomes exactly 0.5, so a
        # `> 0.5` test that was True silently becomes False and the site is
        # dropped. At default -D 500 that is ~11 flipped threshold decisions per
        # variant-gene pair, all in the same direction.
        #
        # SPLICEAI_STORE_FP16=1 restores the old behaviour for throughput on runs
        # that consume only DS/DP at 2 d.p. It must not be used with
        # printEVERYTHING output or EVENT_CLASS.
        xfer_start = time.time()
        _store_fp16 = os.environ.get('SPLICEAI_STORE_FP16', '0') == '1'
        result_np = result.to(torch.float16 if _store_fp16
                              else torch.float32).cpu().numpy()
        xfer_time = time.time() - xfer_start
        self.gpu_profile_counters['transfer_time'] += xfer_time
        
        del result
        del input_tensor

        write_start = time.time()
        # Always write to disk - socket transfer is too slow for large arrays
        # The streaming writer will read from disk
        filename = os.path.join(self.predictions_dir, f"{tensor_size}_{batch_ix}.npy")
        np.save(filename, result_np)

        # Periodic cleanup (less frequent to reduce overhead)
        if batch_count % 100 == 0:
            gc.collect()
            if not is_cpu:
                torch.cuda.empty_cache()

        write_time = time.time() - write_start
        total_time = time.time() - total_start
        self.gpu_profile_counters['total_time'] += total_time
        preds_per_sec = nr_preds / total_time

        # More concise logging
        use_graph = 'GRAPH' if (self.use_cuda_graphs and graph_key in self.cuda_graphs) else 'EAGER'
        msg = (f'Worker {self.worker_id} [{model_dtype}/{use_graph}]: {total_time:.2f}s, {preds_per_sec:.1f}/s | '
               f'prep:{prep_time:.3f}s, infer:{infer_time:.3f}s, xfer:{xfer_time:.3f}s, write:{write_time:.3f}s')

        self.logger.info(msg)

        if self.gpu_profile_interval > 0 and batch_count % self.gpu_profile_interval == 0:
            self._log_gpu_profiling()

    def _log_gpu_profiling(self):
        c = self.gpu_profile_counters
        if c['total_time'] == 0:
            return

        gpu_mem_str = ""
        if self.device.type == 'cuda':
            try:
                mem_used = torch.cuda.memory_allocated() / 1024**2
                mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
                gpu_mem_str = f" | Mem: {mem_used:.0f}/{mem_total:.0f}MB"
            except:
                pass

        self.logger.info("=" * 70)
        self.logger.info(f"PROFILE Worker {self.worker_id}: {self.total_batches_processed} batches, "
                        f"{c['total_predictions']} preds, {c['total_predictions']/c['total_time']:.1f}/s{gpu_mem_str}")
        self.logger.info(f"  Inference: {c['inference_time']:.2f}s ({c['inference_time']/c['total_time']*100:.1f}%)")
        self.logger.info(f"  Transfer:  {c['transfer_time']:.2f}s ({c['transfer_time']/c['total_time']*100:.1f}%)")
        self.logger.info("=" * 70)

        # Reset
        self.gpu_profile_counters = {
            'total_time': 0.0, 'inference_time': 0.0, 'transfer_time': 0.0, 'total_predictions': 0
        }


if __name__ == '__main__':
    main()
