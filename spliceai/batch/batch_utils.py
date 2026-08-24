# Original source code modified to add prediction batching support by Invitae in 2021.
# Modifications copyright (c) 2021 Invitae Corporation.
# Converted to PyTorch with spawn multiprocessing fix for pthread errors
# Optimized transcript batching - groups genes by (strand, padding) to reduce model calls

import logging
import os
import numpy as np
import torch
import pickle
import socket
import subprocess
import sys
import multiprocessing as mp
import time

# Use 'spawn' context to avoid pthread_create errors with CUDA
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# NCCL optimizations for multi-GPU
os.environ.setdefault('NCCL_IB_DISABLE', '0')
os.environ.setdefault('NCCL_P2P_LEVEL', 'NVL')
os.environ.setdefault('NCCL_SHM_DISABLE', '0')
os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
os.environ.setdefault('NCCL_SOCKET_IFNAME', 'eth,ib')

from spliceai.utils import get_cov, get_wid, one_hot_encode, normalise_chrom, get_delta_scores, get_available_cpu_count
from spliceai.batch.data_handlers import VCFReader, _get_attr, SequenceType_REF, SequenceType_ALT, VCFWriter

logger = logging.getLogger(__name__)

#: How long the parent waits for one inference worker to connect back before
#: giving up. Generous: a cold worker loads 5 checkpoints and (on GPU)
#: initialises CUDA. Override with SPLICEAI_WORKER_CONNECT_TIMEOUT (seconds).
WORKER_CONNECT_TIMEOUT_S = float(os.environ.get('SPLICEAI_WORKER_CONNECT_TIMEOUT', 600))
#: How often, while waiting, to check whether the worker process is still alive.
WORKER_ACCEPT_POLL_S = 2.0


def prepare_batches(reference, annotation, input_data, prediction_batch_size, tmpdir, 
                   prediction_queue, nr_workers, shared_dict,
                   distance=50, pytorch_batch_size=None, batch_workers=1):
    """
    Prepare batches for prediction.
    Creates Annotator instance in subprocess to avoid pickle errors with spawn.
    """
    logger.info("prepare_batches: Creating Annotator instance")
    from spliceai.utils import Annotator
    ann = Annotator(reference, annotation, cpu=True, load_models=False)
    
    if batch_workers > 1:
        logger.info(f"Multi-worker batch creation: {batch_workers} workers")
        _prepare_batches_multiprocess(
            reference=reference, annotation=annotation, input_data=input_data,
            prediction_batch_size=prediction_batch_size, tmpdir=tmpdir,
            prediction_queue=prediction_queue, nr_workers=nr_workers,
            shared_dict=shared_dict, 
            distance=distance, pytorch_batch_size=pytorch_batch_size,
            batch_workers=batch_workers,
        )
    else:
        logger.info("Single-worker batch creation")
        _prepare_batches_single(
            ann=ann, input_data=input_data, prediction_batch_size=prediction_batch_size,
            tmpdir=tmpdir, prediction_queue=prediction_queue, nr_workers=nr_workers,
            shared_dict=shared_dict, 
            distance=distance, pytorch_batch_size=pytorch_batch_size,
        )


def _prepare_batches_single(ann, input_data, prediction_batch_size, tmpdir, 
                           prediction_queue, nr_workers, shared_dict, 
                           distance, pytorch_batch_size):
    """Single-worker batch preparation"""
    vcf_reader = VCFReader(
        ann=ann, input_data=input_data, prediction_batch_size=prediction_batch_size,
        prediction_queue=prediction_queue, tmpdir=tmpdir, dist=distance,
        pytorch_batch_size=pytorch_batch_size or 128
    )
    vcf_reader.add_records()
    vcf_reader.finish(nr_workers, shared_dict)
    
    logger.info(f"Read {vcf_reader.total_vcf_records} records, queued {vcf_reader.total_predictions} predictions")

    # Copy shelf records to shared_dict
    for key in vcf_reader.shelf_records:
        shared_dict[key] = vcf_reader.shelf_records[key]
    vcf_reader.shelf_records.close()


def _prepare_batches_multiprocess(reference, annotation, input_data, prediction_batch_size,
                                  tmpdir, prediction_queue, nr_workers, shared_dict, 
                                  distance, pytorch_batch_size, batch_workers):
    """Multi-worker batch preparation for faster VCF processing"""
    ctx = mp.get_context('spawn')
    
    # queues for CPU workers
    work_queues = [ctx.Queue(maxsize=1000) for _ in range(batch_workers)]
    
    workers = []
    for worker_id in range(batch_workers):
        p = ctx.Process(
            target=_batch_worker_process,
            args=(worker_id, work_queues[worker_id], prediction_queue,
                  reference, annotation, tmpdir, prediction_batch_size,
                  pytorch_batch_size or 128, distance)
        )
        p.start()
        workers.append(p)
    
    import pysam
    try:
        vcf = pysam.VariantFile(input_data)
    except (IOError, ValueError) as e:
        logger.error(f"Failed to open VCF: {e}")
        for q in work_queues:
            q.put(None)
        for p in workers:
            p.join()
        raise
    
    total_vcf_records = 0
    for record in vcf:
        work_queues[total_vcf_records % batch_workers].put({
            'chrom': record.chrom, 'pos': record.pos, 'ref': record.ref,
            'alts': record.alts, 'vcf_idx': total_vcf_records + 1,
        })
        total_vcf_records += 1
        if total_vcf_records % 10000 == 0:
            logger.info(f"Distributed {total_vcf_records} VCF records...")
    vcf.close()
    
    # POISON PILL: Signal CPU workers to finish
    logger.info("Sending stop signals to batch workers...")
    for q in work_queues:
        q.put(None)
    
    # Wait for all CPU workers to complete.
    #
    # Exit codes are checked: previously this was a bare p.join() loop, so a
    # worker killed by the OOM killer (exitcode -9) or dying on an unhandled
    # exception (exitcode 1) was indistinguishable from one that finished. The
    # run continued, aggregated whatever shelf entries the survivors had
    # written, and produced a VCF silently missing every variant that worker
    # held -- with a zero exit status. Fail loudly instead.
    failed = []
    for p in workers:
        p.join()
        if p.exitcode not in (0, None):
            failed.append((p.name, p.exitcode))

    if failed:
        detail = ', '.join(
            '{} (exitcode {}{})'.format(
                name, code,
                ', killed by signal {}'.format(-code) if code < 0 else '')
            for name, code in failed)
        raise RuntimeError(
            '{} of {} batch worker(s) failed: {}. Output would be silently '
            'missing every variant assigned to them, so the run is aborted. '
            'A negative exitcode is a signal -- -9 (SIGKILL) is usually the '
            'OOM killer; retry with fewer workers (--cpu-processes / '
            '--batch-workers) or a smaller --prediction-batch-size.'.format(
                len(failed), len(workers), detail))

    logger.info(f"All batch workers finished")
    
    # =========================================================================
    # OPTIMIZED SHELF AGGREGATION (Streaming)
    # =========================================================================
    import shelve
    import gc
    
    shelf_path = os.path.join(tmpdir, 'shelf_records.db')
    logger.info(f"Streaming shelf records directly to {shelf_path}...")
    
    total_predictions = 0
    total_vcf_records = 0 # This should be counted during the pysam loop
    
    # Open the DB once and append each worker's data to it
    with shelve.open(shelf_path, 'n') as shelf:
        for worker_id in range(batch_workers):
            worker_shelf_path = os.path.join(tmpdir, f'shelf_worker_{worker_id}.pkl')
            
            if not os.path.exists(worker_shelf_path):
                logger.warning(f"Worker {worker_id} shelf missing.")
                continue
            
            try:
                with open(worker_shelf_path, 'rb') as f:
                    worker_data = pickle.load(f)
                
                # Add stats
                total_predictions += worker_data.get('total_model_calls', 0)
                records_processed = worker_data.get('records_processed', 0)
                
                # Write records to disk immediately (avoids Huge Dict in RAM)
                worker_records = worker_data['shelf_records']
                for key, value in worker_records.items():
                    shelf[key] = value
                    # CRITICAL FIX: Do NOT write to shared_dict[key] here if it's huge.
                    # shared_dict[key] = value  <-- REMOVED to prevent hang
                
                logger.info(f"Merged {len(worker_records)} records from worker {worker_id}")
                
                # cleanup memory immediately
                del worker_records
                del worker_data
                gc.collect()
                
                os.unlink(worker_shelf_path)
                
            except Exception as e:
                logger.error(f"Failed to merge worker {worker_id}: {e}")

    logger.info(f"Shelf database created. Total predictions: {total_predictions}")

    # Update summary stats in shared_dict (safe for small data)
    shared_dict['total_predictions'] = total_predictions
    
    # =========================================================================
    # SIGNAL GPU WORKERS
    # =========================================================================
    logger.info(f"Sending Finished signal to {nr_workers} GPU workers")
    for _ in range(nr_workers):
        prediction_queue.put('Finished')

def _batch_worker_process(worker_id, work_queue, prediction_queue,
                          reference, annotation, tmpdir, prediction_batch_size,
                          pytorch_batch_size, distance):
    """
    Worker process for batch creation.
    Groups genes by (strand, padding) to reduce redundant model inference.
    """
    logging.basicConfig(
        format=f'%(asctime)s batch_worker_{worker_id}: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO,
    )
    log = logging.getLogger(f"batch_worker_{worker_id}")
    log.info("Starting (optimized transcript batching)")
    
    from spliceai.utils import Annotator
    ann = Annotator(reference, annotation, cpu=True, load_models=False)
    
    batches = {}
    batch_counters = {}
    next_batch_index = {}
    shelf_records = {}
    total_transcripts = 0
    total_model_calls = 0
    records_processed = 0
    
    # Batch ids must be globally unique across batch workers, because every
    # worker's predictions are resolved from a flat {tensor_size}_{batch_ix}.npy
    # namespace. worker_id * BATCH_ID_STRIDE partitions the space, but silently
    # wraps into the NEXT worker's range once a worker emits BATCH_ID_STRIDE
    # batches -- at which point two different batches share a filename and the
    # writer emits one of them twice (wrong scores, no error). Guarded below in
    # flush_batch rather than left to chance.
    BATCH_ID_STRIDE = 100000
    batch_id_offset = worker_id * BATCH_ID_STRIDE
    
    cov = get_cov(distance)
    wid = get_wid(cov)
    
    def flush_batch(tensor_size):
        nonlocal total_model_calls
        if not batches.get(tensor_size):
            return
        
        data_np = np.concatenate(batches[tensor_size])
        if batch_counters[tensor_size] >= BATCH_ID_STRIDE:
            raise RuntimeError(
                f"Batch worker {worker_id} has emitted {batch_counters[tensor_size]} "
                f"batches for tensor_size={tensor_size}, exceeding BATCH_ID_STRIDE="
                f"{BATCH_ID_STRIDE}. The next id would collide with batch worker "
                f"{worker_id + 1}'s range and silently duplicate predictions. "
                f"Raise BATCH_ID_STRIDE or increase -T/--pytorch_batch_size."
            )
        global_batch_ix = batch_id_offset + batch_counters[tensor_size]
        
        queue_item = {
            'tensor_size': tensor_size, 'batch_ix': global_batch_ix,
            'data': data_np, 'length': len(data_np)
        }
        
        filename = f"bw{worker_id}_{tensor_size}--{global_batch_ix}.in.pickle"
        with open(os.path.join(tmpdir, filename), "wb") as p:
            pickle.dump(queue_item, p, protocol=5)
        prediction_queue.put(filename)
        
        total_model_calls += len(data_np)
        batches[tensor_size] = []
        batch_counters[tensor_size] += 1
        next_batch_index[tensor_size] = 0

    while True:
        # FIX: Use blocking get(). 
        # The main process guarantees sending None as a sentinel.
        # Removing timeout prevents crash on large VCF delays.
        item = work_queue.get()
        
        if item is None:
            # Poison Pill received - Flush and Exit
            break 
            
        records_processed += 1
        chrom, pos, ref = item['chrom'], item['pos'], item['ref']
        alts = item['alts'] or []
        vcf_idx = item['vcf_idx']
        
        gene_info = ann.get_name_and_strand(chrom, pos)
        
        if len(gene_info.genes) == 0:
            shelf_records[str(vcf_idx)] = {
                'vcf_idx': vcf_idx, 'gene_info': gene_info, 'locations': [],
            }
            continue
        
        total_transcripts += len(alts) * len(gene_info.genes)
        
        chrom_norm = normalise_chrom(chrom, list(ann.ref_fasta.keys())[0])
        try:
            seq = ann.ref_fasta[chrom_norm][pos - wid // 2 - 1: pos + wid // 2].seq
        except (IndexError, ValueError, KeyError):
            shelf_records[str(vcf_idx)] = {
                'vcf_idx': vcf_idx, 'gene_info': gene_info, 'locations': [],
            }
            continue
        
        if len(seq) != wid or seq[wid // 2: wid // 2 + len(ref)].upper() != ref:
            shelf_records[str(vcf_idx)] = {
                'vcf_idx': vcf_idx, 'gene_info': gene_info, 'locations': [],
            }
            continue
        
        batch_lookup_indexes = []
        
        for alt_ix, alt in enumerate(alts):
            if not alt or any(c in alt for c in '.-*<>'):
                continue
            
            groups = {}
            for gene_ix in range(len(gene_info.genes)):
                strand = gene_info.strands[gene_ix]
                dist_ann = ann.get_pos_data(gene_info.idxs[gene_ix], pos)
                key = (strand, max(wid // 2 + dist_ann[0], 0), max(wid // 2 - dist_ann[1], 0))
                groups.setdefault(key, []).append(gene_ix)
            
            for (strand, pad_left, pad_right), gene_indices in groups.items():
                x_ref = 'N' * pad_left + seq[pad_left:wid - pad_right] + 'N' * pad_right
                x_alt = x_ref[:wid // 2] + str(alt) + x_ref[wid // 2 + len(ref):]
                
                target_len = len(x_ref)
                
                if len(x_alt) > target_len:
                    x_alt = x_alt[:target_len]
                elif len(x_alt) < target_len:
                    x_alt = x_alt + 'N' * (target_len - len(x_alt))
                
                x_ref_enc = one_hot_encode(x_ref)[None, :]
                x_alt_enc = one_hot_encode(x_alt)[None, :]
                
                if strand == '-':
                    x_ref_enc = x_ref_enc[:, ::-1, ::-1].copy()
                    x_alt_enc = x_alt_enc[:, ::-1, ::-1].copy()
                
                tensor_size = x_ref_enc.shape[1]
                
                if tensor_size not in batches:
                    batches[tensor_size] = []
                    batch_counters[tensor_size] = 0
                    next_batch_index[tensor_size] = 0
                
                if next_batch_index[tensor_size] + 2 > prediction_batch_size:
                    flush_batch(tensor_size)
                
                global_batch_ix = batch_id_offset + batch_counters[tensor_size]
                
                ref_batch_index = next_batch_index[tensor_size]
                batches[tensor_size].append(x_ref_enc)
                next_batch_index[tensor_size] += 1
                
                alt_batch_index = next_batch_index[tensor_size]
                batches[tensor_size].append(x_alt_enc)
                next_batch_index[tensor_size] += 1
                
                for gene_ix in gene_indices:
                    batch_lookup_indexes.extend([
                        {'sequence_type': 0, 'tensor_size': tensor_size, 'batch_ix': global_batch_ix,
                         'batch_index': ref_batch_index, 'gene_ix': gene_ix, 'alt_ix': alt_ix},
                        {'sequence_type': 1, 'tensor_size': tensor_size, 'batch_ix': global_batch_ix,
                         'batch_index': alt_batch_index, 'gene_ix': gene_ix, 'alt_ix': alt_ix},
                    ])
        
        shelf_records[str(vcf_idx)] = {
            'vcf_idx': vcf_idx, 'gene_info': gene_info, 'locations': batch_lookup_indexes,
        }
        
        if records_processed % 5000 == 0:
            log.info(f"{records_processed} records: {total_transcripts} transcripts, {total_model_calls} model calls")

    for tensor_size in list(batches.keys()):
        flush_batch(tensor_size)
    
    savings_pct = ((total_transcripts - total_model_calls) / total_transcripts * 100) if total_transcripts > 0 else 0
    log.info(f"Done: {records_processed} records, {total_model_calls} model calls ({savings_pct:.1f}% reduction)")
    
    worker_shelf_path = os.path.join(tmpdir, f'shelf_worker_{worker_id}.pkl')
    with open(worker_shelf_path, 'wb') as f:
        pickle.dump({
            'shelf_records': shelf_records,
            'total_model_calls': total_model_calls,
            'records_processed': records_processed
        }, f, protocol=5)
    log.info(f"Saved {len(shelf_records)} shelf records to {worker_shelf_path}")


def start_workers(prediction_queue, tmpdir, args, devices, device_info, workers_per_gpu=1):
    """Start worker processes for GPU inference"""
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    port = 54677
    try:
        s.bind((socket.gethostname(), port))
    except Exception as e:
        logger.error(f"Cannot bind to port {port}: {e}")
        sys.exit(1)
    s.listen(len(devices) * workers_per_gpu + 5)

    clientThreads, serverThreads = [], []
    total_workers = len(devices) * workers_per_gpu
    logger.info(f"Starting {total_workers} workers ({workers_per_gpu}/GPU, {len(devices)} GPUs)")

    ctx = mp.get_context('spawn')
    worker_counter = 0
    
    for device_idx, device in enumerate(devices):
        device_str = str(device)
        is_cpu_device = device_str == 'cpu' or getattr(device, 'type', None) == 'cpu'
        if is_cpu_device:
            # torch.device('cpu') always has an `.index` attribute (it's just
            # None), so the `hasattr(device, 'index')` branch below used to
            # match CPU devices too and produce gpu_num="None" -- which then
            # got passed to the worker as "-d cuda:None", tripping the
            # `torch.cuda.is_available()` check in batch.py's main() and
            # causing the worker to log "CUDA not available!" and exit(1)
            # immediately, while the parent process hung waiting on a worker
            # that had already died. Handle CPU explicitly, before the
            # cuda/index branches.
            gpu_num = str(device_idx)
            device_arg = 'cpu'
        elif 'cuda:' in device_str:
            gpu_num = device_str.split('cuda:')[1].split()[0]
            device_arg = f"cuda:{gpu_num}"
        elif hasattr(device, 'index') and device.index is not None:
            gpu_num = str(device.index)
            device_arg = f"cuda:{gpu_num}"
        else:
            gpu_num = str(device_idx)
            device_arg = device_str

        for _ in range(workers_per_gpu):
            global_worker_id = worker_counter
            worker_counter += 1

            cmd = [
                "python3", os.path.join(os.path.dirname(os.path.realpath(__file__)), "batch.py"),
                "-t", tmpdir, "-d", device_arg, "-w", str(global_worker_id),
                '-R', args.reference, '-A', args.annotation,
                '-T', str(getattr(args, 'pytorch_batch_size', args.prediction_batch_size)),
                '--gpu-profile-interval', str(getattr(args, 'gpu_profile_interval', 100)),
                '--precision', str(getattr(args, 'precision', 'auto')),
                '--workers-per-gpu', str(workers_per_gpu),
                # so each CPU worker sizes its OMP/MKL pool to cores/N rather
                # than claiming all of them (see calculate_optimal_threads)
                '--cpu-processes', str(len(devices) if is_cpu_device else 1),
            ]
            if args.verbose:
                cmd.append('-V')
            if getattr(args, 'compile', False):
                cmd.append('--compile')
            if getattr(args, 'no_cuda_graphs', False):
                cmd.append('--no-cuda-graphs')

            env = os.environ.copy()
            if not is_cpu_device:
                env['CUDA_VISIBLE_DEVICES'] = gpu_num

            log_suffix = f"CPU_w{global_worker_id}" if is_cpu_device else f"GPU_{gpu_num}_w{global_worker_id}"
            fh_stdout = open(os.path.join(tmpdir, f'{log_suffix}.stdout'), 'w')
            fh_stderr = open(os.path.join(tmpdir, f'{log_suffix}.stderr'), 'w')

            p = subprocess.Popen(cmd, stdout=fh_stdout, stderr=fh_stderr, env=env)
            clientThreads.append(p)

            # Wait for the worker to connect, but poll so that a worker which
            # dies during startup is detected instead of hanging the parent
            # forever. s.accept() has no timeout by default, so any worker that
            # exited before connecting (bad -d string, missing checkpoint, OOM,
            # import error) left the parent blocked with no output and no error
            # -- the failure mode looked like a very slow run.
            client = None
            deadline = time.time() + WORKER_CONNECT_TIMEOUT_S
            s.settimeout(WORKER_ACCEPT_POLL_S)
            try:
                while True:
                    try:
                        client, _ = s.accept()
                        break
                    except socket.timeout:
                        rc = p.poll()
                        if rc is not None:
                            fh_stdout.flush(); fh_stderr.flush()
                            raise RuntimeError(
                                f"Inference worker {global_worker_id} (device {device}) "
                                f"exited with code {rc} before connecting. See "
                                f"{os.path.join(tmpdir, log_suffix + '.stderr')} for the "
                                f"traceback."
                            )
                        if time.time() > deadline:
                            raise TimeoutError(
                                f"Inference worker {global_worker_id} (device {device}) "
                                f"did not connect within {WORKER_CONNECT_TIMEOUT_S}s. "
                                f"It is still running (pid {p.pid}); check "
                                f"{os.path.join(tmpdir, log_suffix + '.stderr')}. Raise "
                                f"SPLICEAI_WORKER_CONNECT_TIMEOUT if model loading is "
                                f"genuinely this slow."
                            )
            except Exception as e:
                logger.error(f"Error waiting for worker {global_worker_id}: {e}")
                # Don't leave already-started workers orphaned on the way out.
                for proc in clientThreads:
                    if proc.poll() is None:
                        proc.terminate()
                raise
            finally:
                s.settimeout(None)
            logger.info(f"Worker {global_worker_id} connected")

            server_thread = ctx.Process(target=_process_server, args=(client, str(device), prediction_queue, global_worker_id))
            server_thread.start()
            serverThreads.append(server_thread)

    logger.info(f"All {total_workers} workers connected")
    return clientThreads, serverThreads, devices


def _process_server(clientsocket, device, input_queue, worker_id=0):
    """
    Server process that dispatches work batches to GPU worker
    """
    import logging
    import queue # Import queue for Empty exception
    
    logging.basicConfig(
        format='%(asctime)s SERVER %(name)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.DEBUG,
    )
    log = logging.getLogger(f"server_{worker_id}")
    
    batches_sent = 0
    
    try:
        clientsocket.send(b'Server is online\n')
        clientsocket.settimeout(10.0)
        log.info(f"Server {worker_id} started, waiting for worker messages")
        
        recv_buffer = b''
        while True:
            try:
                data = clientsocket.recv(4096)
                if not data:
                    log.info(f"Connection closed by worker")
                    break
                recv_buffer += data
            except socket.timeout:
                continue
            except (ConnectionResetError, BrokenPipeError) as e:
                log.info(f"Connection error: {e}")
                break
            
            while b'\n' in recv_buffer:
                line, recv_buffer = recv_buffer.split(b'\n', 1)
                msg = line.decode('utf-8').strip()
                
                if not msg:
                    continue
                
                if msg == 'Done':
                    log.info(f"Worker sent Done, exiting")
                    return
                
                if 'Ready for work' in msg or 'ready' in msg.lower():
                    # FIX: Don't count retries. Wait for queue or send Hold On.
                    try:
                        # Wait briefly for a batch (fast response)
                        item = input_queue.get(timeout=0.2)
                    except queue.Empty:
                        # Queue is empty, but we are NOT done. 
                        # Tell worker to wait (keep connection alive)
                        item = 'Hold On'
                    
                    try:
                        if item == 'Finished':
                            clientsocket.sendall(f"{item}\n".encode())
                            log.info(f"Sent Finished signal to worker. Total batches: {batches_sent}")
                            return # Exit strictly on Finished signal
                            
                        elif item == 'Hold On':
                            clientsocket.sendall(f"{item}\n".encode())
                            
                        else:
                            # It's a filename
                            clientsocket.sendall(f"{item}\n".encode())
                            batches_sent += 1
                            log.debug(f"Sent batch file {batches_sent}: {item}")
                            
                    except (BrokenPipeError, ConnectionResetError) as e:
                        log.info(f"Send error: {e}")
                        break
            
        log.info(f"Server {worker_id} exiting. Sent {batches_sent} batches")
                
    finally:
        try:
            clientsocket.close()
        except:
            pass

def initialize_one_device(args):
    """Initialize a single device for a worker process"""
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def initialize_devices(args):
    """Initialize all available devices"""
    device_info = {}
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"Detected {num_gpus} CUDA GPU(s)")
        
        if args.gpus.lower() == 'all':
            devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]
        elif args.gpus == '0':
            devices = [torch.device('cpu')]
        else:
            gpu_indices = [int(x) for x in args.gpus.split(',')]
            devices = [torch.device(f'cuda:{i}') for i in gpu_indices if i < num_gpus]
        
        for i, device in enumerate(devices):
            if device.type == 'cuda':
                props = torch.cuda.get_device_properties(device)
                device_info[str(device)] = {
                    'name': props.name,
                    'memory_gb': props.total_memory / 1024**3,
                    'compute_capability': f"{props.major}.{props.minor}"
                }
                logger.info(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)")
    else:
        # A single CPU "device" means a single worker process, and one process
        # cannot use a 16-core allocation: intra-op thread scaling on the Xeon
        # E5-2640 v3 reaches 4.43 preds/s at 16 threads, whereas 16 single-thread
        # processes over the same 16 cores reach 7.28 preds/s aggregate (1.64x)
        # -- MKLDNN's intra-op parallelism over 32-channel convs leaves cores
        # idle, while independent processes do not. Emit one CPU device per
        # `--cpu-processes` so start_workers() spawns that many workers; they
        # already shard work through the existing prediction socket, and
        # calculate_optimal_threads() divides the cores among them.
        cpu_procs = int(getattr(args, 'cpu_processes', 0))
        if cpu_procs == 0:
            cpu_procs = get_available_cpu_count()
        n_cpu_proc = max(1, cpu_procs)
        if n_cpu_proc > 1:
            logger.info(f"No CUDA GPUs available, using CPU with "
                        f"{n_cpu_proc} worker processes")
        else:
            logger.info("No CUDA GPUs available, using CPU")
        devices = [torch.device('cpu') for _ in range(n_cpu_proc)]
        device_info['cpu'] = {'name': 'CPU', 'memory_gb': 0,
                              'processes': n_cpu_proc}

    return devices, device_info