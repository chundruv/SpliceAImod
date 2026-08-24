# Original source code modified to add prediction batching support by Invitae in 2021.
# Modifications copyright (c) 2021 Invitae Corporation.
# Further modifications by Geert Van de Weyer in 2022 and Kartik Chundru in 2025.

import sys
import argparse
import logging
import pysam
import time
import tempfile
import multiprocessing as mp
from functools import partial
import shutil
import subprocess as sp
import os
import platform
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass 
from multiprocessing import Process, Queue
import torch
from spliceai.batch.batch_utils import prepare_batches, start_workers, initialize_devices
from spliceai.utils import Annotator, get_delta_scores, get_available_cpu_count, \
        info_header_description, n_info_fields, get_tsv_header_comments, get_tsv_header_line
from spliceai.batch.data_handlers import VCFWriter

def get_options():

    parser = argparse.ArgumentParser(description='Version: 1.3.1 GPU-Optimized PyTorch')
    
    parser.add_argument('-I', '--input_data',
                        metavar='input', 
                        type=str, 
                        required=True,
                        help='path to the input VCF file')
    parser.add_argument('-O', '--output_data', 
                        metavar='output', 
                        type=str,
                        required=True, 
                        help='path to the output VCF file')
    parser.add_argument('-R', '--reference', 
                        metavar='reference', 
                        required=True, 
                        help='path to the reference genome fasta file')
    parser.add_argument('-A', '--annotation', 
                        metavar='annotation', 
                        default='gencodev49',
                        help='Gene annotation file. GENCODE primary v49 (input "gencodev49", default) and MANEv1.4 (use "MANEv1.4") are provided')
    parser.add_argument('-D', '--distance', 
                        metavar='distance',
                        default=500, 
                        type=int, 
                        choices=range(0, 5000), 
                        help='maximum distance between the variant and gained/lost splice site, defaults to 500')
    parser.add_argument('-B', '--prediction_batch_size', 
                        metavar='prediction_batch_size', 
                        default=1, 
                        type=int, 
                        help='number of predictions to process at a time')
    parser.add_argument('-T', '--pytorch_batch_size', 
                        metavar='pytorch_batch_size', 
                        type=int, 
                        help='PyTorch batch size for model predictions')
    parser.add_argument('-V', '--verbose', 
                        action='store_true', 
                        help='enables verbose logging')
    parser.add_argument('--gpu-profile-interval', 
                        metavar='gpu_profile_interval', 
                        type=int, 
                        default=100, 
                        help='GPU profiling output frequency in batches')
    parser.add_argument('-t', '--tmpdir', 
                        metavar='tmpdir', 
                        type=str, 
                        default='/tmp/',
                        help="Use Alternate location to store tmp files")
    parser.add_argument('-G', '--gpus', 
                        metavar='gpus', 
                        type=str, 
                        default='all',
                        help="Number of GPUs to use for SpliceAI")
    parser.add_argument('--cpu-processes',
                        metavar='cpu_processes',
                        default=0,
                        type=int,
                        help='number of CPU worker processes to run when no GPU '
                             'is available (default: 0, which uses all available cores). '
                             'One process cannot saturate a multi-core allocation; '
                             'setting this to the number of allocated cores gives '
                             'significant throughput improvements. Ignored on GPU runs.')
    parser.add_argument('--workers-per-gpu', 
                        metavar='workers_per_gpu', 
                        type=int, 
                        default=1, 
                        help='Number of workers per GPU (default: 1).')
    parser.add_argument('--batch-workers', 
                        metavar='batch_workers', 
                        type=int, 
                        default=1, 
                        help='Number of parallel workers for batch creation/VCF reading (default: 1). Higher values can speed up batch preparation for large VCF files.')
    parser.add_argument('--precision', 
                        metavar='precision', 
                        type=str, 
                        default='auto', 
                        choices=['auto', 'fp32', 'fp16', 'bf16'],
                        help='Precision mode for inference')
    parser.add_argument('--compile', 
                        action='store_true',
                        help='Use torch.compile() for optimized inference (requires PyTorch 2.0+)')
    parser.add_argument('--no-cuda-graphs', 
                        action='store_true',
                        help='Disable CUDA Graphs optimization')
    parser.add_argument('--orig-output',
                        default=False,
                        action='store_true',
                        help='Only print original SpliceAI VCF output without additional fields')
    parser.add_argument('--skip-write',
                        default=False,
                        action='store_true',
                        help='Skip writing the final VCF output. Still outputs tar.gz of intermediate files.')
    parser.add_argument('--vcf-output',
                        default=False,
                        action='store_true',
                        help='Output results in VCF format instead of TSV (default is TSV).')
    args = parser.parse_args()
    
    return args


def resolve_auto_precision(is_cpu):
    """Pick the precision to use when --precision is 'auto'.

    This is a capability query, not a lookup of the GPU's model name: bf16
    tensor cores exist on Ampere and later, and older CUDA devices accelerate
    fp16 instead. CPU inference stays in fp32 -- half precision has no
    hardware support there and is slower, not faster.
    """
    if is_cpu or not torch.cuda.is_available():
        return 'fp32'
    return 'bf16' if torch.cuda.is_bf16_supported() else 'fp16'


# NOTE: count_vcf_variants() lived here — it counted non-header lines in the
# input VCF (with a file-size fallback) to size a progress estimate. Nothing
# ever called it, and on a large gzipped VCF it would have cost a full extra
# decompression pass before any work began. Removed.


def main():
    args = get_options()
    
    # Logging setup
    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(name)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=loglevel,
    )
    
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"cuDNN version: {torch.backends.cudnn.version()}")

    # Sanity check for mandatory arguments
    if None in [args.input_data, args.output_data, args.distance]:
        logging.error('Usage: spliceai [-h] [-I [input]] [-O [output]] -R reference -A annotation '
                      '[-D [distance]] [-B [prediction_batch_size]] [-T [pytorch_batch_size]] [-t [tmp_location]] [-G [gpus]] [--workers-per-gpu WORKERS_PER_GPU]  [--batch-workers BATCH_WORKERS] [--gpu-profile-interval GPU_PROFILE_INTERVAL] [--precision {auto,fp32,fp16,bf16}] [--compile] [--no-cuda-graphs] [-V] [--orig-output] [--skip-write] [--vcf-output]')
        exit()



    # Batched analysis (PyTorch)
    if args.prediction_batch_size > 1:
        # Initialize devices
        devices, device_info = initialize_devices(args)

        # Detect if we're using CPU or GPU
        is_cpu = all('cpu' in str(d).lower() for d in devices)

        # Device-specific optimization: Smart defaults for pytorch_batch_size
        if args.pytorch_batch_size is None:
            if is_cpu:
                cpu_procs = int(getattr(args, 'cpu_processes', 0))
                if cpu_procs == 0:
                    cpu_procs = get_available_cpu_count()
                
                if cpu_procs > 1:
                    # Multi-process CPU: batch size 1 maximizes L3 cache locality per core
                    args.pytorch_batch_size = min(args.prediction_batch_size, 1)
                else:
                    # Single-process CPU: higher batch sizes parallelize better across threads
                    args.pytorch_batch_size = min(args.prediction_batch_size, 16)
                logging.info(f"CPU: Auto-set pytorch_batch_size to {args.pytorch_batch_size}")
            else:
                if len(devices) == 1:
                    args.pytorch_batch_size = min(args.prediction_batch_size, 64)
                else:
                    args.pytorch_batch_size = min(args.prediction_batch_size, 32)
                logging.info(f"GPU: Auto-set pytorch_batch_size to {args.pytorch_batch_size} "
                             f"(raise it with -T if the device has memory to spare)")

        # Auto-detect precision mode
        if args.precision == 'auto':
            args.precision = resolve_auto_precision(is_cpu)

        # Auto-set batch workers based on CPU count if not specified
        if args.batch_workers == 1:
            cpu_count = get_available_cpu_count()
            # Default to a reasonable number based on CPU count
            if cpu_count >= 32:
                suggested_batch_workers = min(8, cpu_count // 8)
            elif cpu_count >= 16:
                suggested_batch_workers = min(4, cpu_count // 4)
            elif cpu_count >= 8:
                suggested_batch_workers = 2
            else:
                suggested_batch_workers = 1
            # Don't auto-set, just inform
            if suggested_batch_workers > 1:
                logging.info(f"Tip: Consider using --batch-workers {suggested_batch_workers} for faster batch preparation")

        # Log optimization status
        if is_cpu:
            logging.info("CPU Mode Active:")
            logging.info(f"  - Using {len(devices)} CPU device(s)")
        else:
            logging.info("GPU PyTorch Optimizations Active:")
            logging.info(f"  - Precision: {args.precision}")
            if args.compile:
                logging.info("  - torch.compile() enabled")
            logging.info(f"  - Using {len(devices)} GPU(s)")
        logging.info(f"  - Prediction batch size: {args.prediction_batch_size}")
        logging.info(f"  - PyTorch batch size: {args.pytorch_batch_size}")
        logging.info(f"  - Batch creation workers: {args.batch_workers}")
        
        # Load annotation data (without loading models - workers will load them on their devices)
        logging.debug("Loading annotations (models will be loaded by workers)")
        ann = Annotator(args.reference, args.annotation, cpu=True, load_models=False)
        logging.debug("Annotation loaded.")
        
        # Run batched analysis
        run_spliceai_batched(args, ann, devices, device_info)

    else:
        # Run original single-prediction mode
        logging.info("Running in single-prediction mode (consider using -B for better GPU performance)")
        # Resolve 'auto' here too: this path used to hand the literal string
        # 'auto' to Annotator, which matches none of its precision branches, so
        # --precision auto silently meant fp32 in single-prediction mode.
        if args.precision == 'auto':
            args.precision = resolve_auto_precision(not torch.cuda.is_available())
            logging.info(f"Auto-selected precision: {args.precision}")
        ann = Annotator(args.reference, args.annotation, cpu=False, load_models=True,
                       precision=args.precision, compile_model=args.compile)
        run_spliceai(args, ann)


def run_spliceai_batched(args, ann, devices, device_info):
    """Batched analysis: prepare batches -> GPU inference -> write VCF"""
    
    start_time = time.time()
   
    # Now args.input_data and args.output_data are strings (picklable)
    
    prediction_batch_size = args.prediction_batch_size
    pytorch_batch_size = args.pytorch_batch_size
    batch_workers = args.batch_workers

    tmpdir = tempfile.mkdtemp(dir=args.tmpdir)
    try:
        logging.info("Using tmpdir: {}".format(tmpdir))

        # Calculate queue size and memory usage
        try:
            import psutil
            total_ram_gb = psutil.virtual_memory().total / (1024**3)
            available_ram_gb = psutil.virtual_memory().available / (1024**3)
            logging.info(f"System RAM: {total_ram_gb:.1f}GB total, {available_ram_gb:.1f}GB available")

            disk_usage = psutil.disk_usage(tmpdir)
            available_disk_gb = disk_usage.free / (1024**3)
            logging.info(f"Disk space: {available_disk_gb:.1f}GB available in {tmpdir}")
        except ImportError:
            logging.warning("psutil not available, using conservative queue size")
            total_ram_gb = 16
            available_ram_gb = 12
            available_disk_gb = 50

        # Memory estimation
        bytes_per_sequence = 10000 * 4 * 4  # ~160 KB
        batch_memory_mb = (prediction_batch_size * bytes_per_sequence) / (1024 * 1024)

        num_workers = len(devices) * args.workers_per_gpu
        in_flight_batches = 1 + num_workers
        est_in_flight_memory_gb = (batch_memory_mb * in_flight_batches) / 1024

        # Calculate queue size
        if total_ram_gb >= 30:
            max_queue_batches = 256
        else:
            if prediction_batch_size <= 2048:
                max_queue_batches = 128
            elif prediction_batch_size <= 3072:
                max_queue_batches = 64
            else:
                max_queue_batches = 48


        disk_per_batch_mb = batch_memory_mb * 2.0
        usable_disk_gb = available_disk_gb * 0.6
        max_queue_by_disk = int((usable_disk_gb * 1024) / disk_per_batch_mb)
        adaptive_queue_size = max(32, min(max_queue_batches, max_queue_by_disk, 512))

        min_multi_gpu = len(devices) * 64 if len(devices) > 1 else 64
        queue_size = max(adaptive_queue_size, min_multi_gpu)

        logging.info(f"Adaptive queue size: {queue_size}")
        logging.info(f"Estimated in-flight batch memory: {est_in_flight_memory_gb:.1f}GB")

        # Use spawn context for Queue to avoid pthread issues
        ctx = mp.get_context('spawn')
        prediction_queue = ctx.Queue(maxsize=queue_size)


        # Shared memory dict
        from multiprocessing import Manager
        manager = Manager()
        shared_dict = manager.dict()

        # Log batch mode
        logging.info("=" * 80)
        logging.info("DISK-BASED BATCH MODE (default)")
        logging.info(f"Using tmpdir: {tmpdir}")
        logging.info("=" * 80)

        # Start batch preparation process(es)
        workers_per_gpu = args.workers_per_gpu
        total_gpu_workers = len(devices) * workers_per_gpu
        shared_dict['num_workers'] = total_gpu_workers

        logging.info(f"Total GPU workers: {len(devices)} GPUs × {workers_per_gpu} workers/GPU = {total_gpu_workers}")
        logging.info(f"Batch creation workers: {batch_workers}")
    
        # Phase 1: Read VCF and prepare batches
        reader_args = {
            'reference': args.reference,
            'annotation': args.annotation,
            'input_data': args.input_data,
            'prediction_batch_size': args.prediction_batch_size,
            'tmpdir': tmpdir,
            'prediction_queue': prediction_queue,
            'nr_workers': total_gpu_workers,
            'shared_dict': shared_dict,
            'distance': args.distance,
            'pytorch_batch_size': pytorch_batch_size,
            'batch_workers': batch_workers,
        }
    
        reader = ctx.Process(target=prepare_batches, kwargs=reader_args)
        reader.start()
        logging.info("Batch reader started")

        # Phase 2: Start GPU workers for inference
        worker_clients, worker_servers, devices = start_workers(prediction_queue, tmpdir, args, devices, device_info, workers_per_gpu)
        logging.info(f"Started {len(worker_clients)} GPU workers")

        # Wait for reader to finish (batches are prepared)
        logging.debug("Waiting for VCF reader to join")
        reader.join()
        logging.debug("Reader joined!")
    
        # Wait for GPU workers to finish (predictions are done)
        logging.debug("Waiting for workers to join")
        for p in worker_clients:
            p.wait()
        logging.debug("Workers are done!")
    
        logging.debug("Waiting for servers to join")
        for p in worker_servers:
            p.join()
        logging.debug("Servers are done")

        prediction_duration = time.time() - start_time
        logging.info(f"GPU inference complete in {prediction_duration:.1f}s")

        # Phase 3: Post-process predictions and write final VCF
        logging.info("Aggregating and writing output VCF...")
        write_start = time.time()
    
        # Use the high-performance VCFWriter from data_handlers
        import shelve
        import pickle
        import shutil

        shelf_path = os.path.join(tmpdir, 'shelf_records.db')
        pickle.dump(shared_dict, open(os.path.join(tmpdir, 'shared_dict.pkl'), 'wb'))

        if args.skip_write:
            logging.info("Skipping VCF writing as per --skip-write flag.")
            if args.output_data.endswith('.vcf.gz'):
                tar_file = args.output_data[:-7] + '.tar.gz'
            else:
                tar_file = args.output_data + '.tar.gz'

            # Create tar archive with proper command structure
            logging.info(f"Creating tar archive of tmpdir at {tar_file}...")
            logging.info("Use run_vcf_writer.py to extract and write the final VCF from the tar archive.")

            tmpdir_parent = os.path.dirname(tmpdir)
            tmpdir_name = os.path.basename(tmpdir)

            try:
                if shutil.which("pigz") is not None:
                    # Use pigz for faster compression
                    # First create uncompressed tar, then pipe to pigz
                    with open(tar_file, 'wb') as f:
                        tar_proc = sp.Popen(
                            ["tar", "-c", "-C", tmpdir_parent, tmpdir_name],
                            stdout=sp.PIPE
                        )
                        pigz_proc = sp.Popen(
                            ["pigz"],
                            stdin=tar_proc.stdout,
                            stdout=f
                        )
                        tar_proc.stdout.close()
                        pigz_proc.communicate()
                    
                    if pigz_proc.returncode != 0:
                        raise sp.CalledProcessError(pigz_proc.returncode, "pigz")
                else:
                    # Use standard gzip compression
                    sp.run(
                        ["tar", "-czf", tar_file, "-C", tmpdir_parent, tmpdir_name],
                        check=True
                    )
                logging.info(f"Tar archive created successfully: {tar_file}")
            except sp.CalledProcessError as e:
                logging.warning(f"Failed to create tar archive: {e}")
                logging.warning("Continuing without tar archive...")
            except Exception as e:
                logging.warning(f"Unexpected error creating tar archive: {e}")
                logging.warning("Continuing without tar archive...")

        else:
            with shelve.open(shelf_path, 'r') as shelf_records:
                writer = VCFWriter(args, tmpdir, devices, ann, shelf_records=shelf_records, shared_dict=shared_dict)
                if not args.vcf_output:
                    writer.process_tsv()
                else:
                    writer.process()

            write_duration = time.time() - write_start
            logging.info(f"{'VCF' if args.vcf_output else 'TSV'} writing complete in {write_duration:.1f}s")


        total_predictions = shared_dict.get('total_predictions', 0)
        total_vcf_records = shared_dict.get('total_vcf_records', 0)

        # Cleanup
    finally:
        # tempfile.mkdtemp() writes the shelf DB, per-worker pickles and a tar
        # of the whole directory here. The rmtree was previously the last
        # statement on the success path only, so any exception raised while
        # predicting, tarring or writing left the entire tmpdir behind — on a
        # whole-genome VCF that is tens of GB, and the default location is the
        # system temp dir. try/finally makes cleanup unconditional.
        shutil.rmtree(tmpdir, ignore_errors=True)
    
    # Stats
    overall_duration = time.time() - start_time
    preds_per_sec = total_predictions / prediction_duration if prediction_duration > 0 else 0
    preds_per_hour = preds_per_sec * 60 * 60
    
    logging.info("=" * 80)
    logging.info("PYTORCH GPU-OPTIMIZED ANALYSIS COMPLETE")
    logging.info("=" * 80)
    logging.info("Total RunTime: {:0.2f}s".format(overall_duration))
    logging.info("  - GPU Inference: {:0.2f}s".format(prediction_duration))
    if not args.skip_write:
        logging.info(f"  - {'VCF' if args.vcf_output else 'TSV'} Writing: {write_duration:.1f}s")
    logging.info("Processed Records: {}".format(total_vcf_records))
    logging.info("Processed Predictions: {}".format(total_predictions))
    logging.info("-" * 80)
    logging.info("GPU performance: {:0.2f} predictions/sec ; {:0.2f} predictions/hour".format(
        preds_per_sec, preds_per_hour
    ))
    logging.info("Average per device: {:0.2f} predictions/sec ; {:0.2f} predictions/hour".format(
        preds_per_sec / len(devices), preds_per_hour / len(devices)
    ))
    manager.shutdown()
    logging.info("=" * 80)


def run_spliceai(args, ann):
    """Original single-record processing flow with PyTorch"""
    input_data = args.input_data
    output_data = args.output_data
    distance = args.distance
    
    try:
        vcf = pysam.VariantFile(input_data)
    except (IOError, ValueError) as e:
        logging.error('{}'.format(e))
        exit()

    if not args.vcf_output:
        import gzip
        try:
            out_file = gzip.open(output_data, 'wt')
        except (IOError, ValueError) as e:
            logging.error('{}'.format(e))
            exit()

        out_file.write(get_tsv_header_comments(args.orig_output))
        out_file.write(get_tsv_header_line(args.orig_output))

        for record in vcf:
            scores = get_delta_scores(record, ann, distance, args.orig_output)
            alt_str = ','.join(record.alts) if record.alts else '.'
            for score_str in scores:
                score_fields = score_str.split('|')
                row_fields = [record.chrom, str(record.pos), record.ref, alt_str] + score_fields
                out_file.write('\t'.join(row_fields) + '\n')

        vcf.close()
        out_file.close()
    else:
        header = vcf.header
        if args.orig_output:
            header.add_line(
                '##INFO=<ID=SpliceAI_orig,Number=.,Type=String,Description="'
                + info_header_description(orig=True) + '">')
        else:
            header.add_line(
                '##INFO=<ID=SpliceAI,Number=.,Type=String,Description="'
                + info_header_description(orig=False) + '">')

        try:
            out_vcf = pysam.VariantFile(output_data, mode='w', header=header)
        except (IOError, ValueError) as e:
            logging.error('{}'.format(e))
            exit()

        for record in vcf:
            scores = get_delta_scores(record, ann, distance, args.orig_output)
            if len(scores) > 0:
                record.info['SpliceAI'] = scores
            out_vcf.write(record)
       
        vcf.close()
        out_vcf.close()


if __name__ == '__main__':
    main()
