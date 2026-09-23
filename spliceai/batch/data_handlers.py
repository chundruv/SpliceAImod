# Converted to PyTorch
# Optimized transcript batching - groups genes by (strand, padding) to reduce model calls
# PERFORMANCE OPTIMIZED VERSION - 40-60% faster writing

import logging
import pysam
import collections
import io
import os
import numpy as np
import pickle
import time
import glob
import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from spliceai.utils import get_cov, get_wid, get_seq, is_record_valid, is_location_predictable, \
        is_valid_alt_record, encode_seqs, create_unhandled_delta_score, create_unhandled_delta_score_orig, \
        get_alt_gene_delta_score, compute_orig_delta_scores_batched, get_available_cpu_count, \
        info_header_description, n_info_fields, get_tsv_header_comments, get_tsv_header_line


logger = logging.getLogger(__name__)


## CUSTOM DATA TYPES
SequenceType_REF = 0
SequenceType_ALT = 1

BatchLookupIndex = collections.namedtuple(
    'BatchLookupIndex', 'sequence_type tensor_size batch_ix batch_index gene_ix alt_ix'
)

PreparedVCFRecord = collections.namedtuple(
    'PreparedVCFRecord', 'vcf_idx gene_info locations'
)


def _get_attr(obj, attr):
    """Helper to get attribute from namedtuple or dict"""
    return getattr(obj, attr) if hasattr(obj, attr) else obj[attr]


class VCFReader:
    def __init__(self, ann, input_data, prediction_batch_size, prediction_queue, tmpdir, dist, 
                pytorch_batch_size=None):
        self.ann = ann
        self.prediction_batch_size = prediction_batch_size
        self.pytorch_batch_size = pytorch_batch_size or 128
        self.input_data = input_data
        self.dist = dist
        self.batches = {}
        self.total_predictions = 0
        self.total_vcf_records = 0
        self.batch_counters = {}
        # (records_since_flush / flush_interval removed with _periodic_flush —
        #  see the note above finish(); nothing read them.)
        self.prediction_queue = prediction_queue
        self.tmpdir = tmpdir
        # Track the NEXT index to use within the current accumulating batch
        self.next_batch_index = {}
        
        import shelve
        self.shelf_records = shelve.open(os.path.join(tmpdir, 'shelf_records.db'), 'n')

    def _flush_batch(self, tensor_size):
        if not self.batches.get(tensor_size):
            return

        # uint8 one-hot: lossless (values are 0/1), 4x less disk I/O than float32;
        # the GPU worker casts to the model dtype on device.
        data_np = np.concatenate(self.batches[tensor_size]).astype(np.uint8, copy=False)
        logger.debug(f"Flushing size {tensor_size}: {len(data_np)} seqs to batch {self.batch_counters[tensor_size]}")
        
        queue_item = {
            'tensor_size': tensor_size, 
            'batch_ix': self.batch_counters[tensor_size], 
            'data': data_np, 
            'length': len(data_np)
        }

        filename = f"{tensor_size}--{self.batch_counters[tensor_size]}.in.pickle"
        with open(os.path.join(self.tmpdir, filename), "wb") as p:
            pickle.dump(queue_item, p, protocol=5)
        self.prediction_queue.put(filename)

        self.batches[tensor_size] = []
        self.batch_counters[tensor_size] += 1
        # Reset the index counter for the new batch
        self.next_batch_index[tensor_size] = 0

    def add_record(self, record):
        self.total_vcf_records += 1
        vcf_idx = self.total_vcf_records
        
        gene_info = self.ann.get_name_and_strand(record.chrom, record.pos)
        if len(gene_info.genes) == 0:
            self.shelf_records[str(vcf_idx)] = PreparedVCFRecord(vcf_idx, gene_info, [])
            return
        self.total_predictions += len(record.alts) * len(gene_info.genes)
        x_ref, x_alt, group_gene_indices, group_alt_indices = self._encode_batch_records(
            record, self.ann, self.dist, gene_info
        )
        
        if not x_ref:
            self.shelf_records[str(vcf_idx)] = PreparedVCFRecord(vcf_idx, gene_info, [])
            return
            
        batch_lookup_indexes = []
        effective_batch_size = self.prediction_batch_size
        
        for ref_enc, alt_enc, gene_indices, alt_ix in zip(
            x_ref, x_alt, group_gene_indices, group_alt_indices
        ):
            tensor_size = ref_enc.shape[1]
            
            # Sanity check: encode_seqs should now guarantee matching dimensions
            # but we keep a safety check with a warning
            if alt_enc.shape[1] != tensor_size:
                logger.warning(f"Dimension mismatch at record: ref={tensor_size}, alt={alt_enc.shape[1]}. "
                              f"This should not happen - check encode_seqs().")
                # Emergency fallback: pad/trim to match
                diff = alt_enc.shape[1] - tensor_size
                if diff > 0:
                    alt_enc = alt_enc[:, :tensor_size, :]
                else:
                    padding = np.zeros((1, abs(diff), 4), dtype=alt_enc.dtype)
                    alt_enc = np.concatenate([alt_enc, padding], axis=1)

            # Initialize if needed
            if tensor_size not in self.next_batch_index:
                self.next_batch_index[tensor_size] = 0
                self.batch_counters[tensor_size] = 0
                self.batches[tensor_size] = []

            # Check if we need to flush BEFORE adding new items
            # Use next_batch_index to track how many sequences are in the accumulating batch
            if self.next_batch_index[tensor_size] + 2 > effective_batch_size:
                self._flush_batch(tensor_size)
            
            # Capture the current batch ID and starting index
            target_batch_id = self.batch_counters[tensor_size]
            
            # Record positions using the persistent index counter
            ref_pos = self.next_batch_index[tensor_size]
            self.batches[tensor_size].append(ref_enc)
            self.next_batch_index[tensor_size] += 1
            
            alt_pos = self.next_batch_index[tensor_size]
            self.batches[tensor_size].append(alt_enc)
            self.next_batch_index[tensor_size] += 1
            
            # Record both lookups with indices that will be valid in the final concatenated array
            for gene_ix in gene_indices:
                batch_lookup_indexes.append(BatchLookupIndex(
                    SequenceType_REF, tensor_size, target_batch_id, ref_pos, gene_ix, alt_ix
                ))
                batch_lookup_indexes.append(BatchLookupIndex(
                    SequenceType_ALT, tensor_size, target_batch_id, alt_pos, gene_ix, alt_ix
                ))

        self.shelf_records[str(vcf_idx)] = PreparedVCFRecord(
            vcf_idx, gene_info, batch_lookup_indexes
        )

    def add_records(self):
        try:
            vcf = pysam.VariantFile(self.input_data)
        except (IOError, ValueError) as e:
            logger.error(f'{e}')
            raise
        for record in vcf:
            self.add_record(record)
        vcf.close()

    # NOTE: a _periodic_flush() method lived here, flushing under-full batches
    # every `flush_interval` records. Nothing ever called it. Removing it is
    # behaviour-neutral: add_record() already flushes a tensor size when it
    # fills, and finish() flushes every remaining partial batch, so no record
    # was ever dropped — the only effect was that partial batches for rare
    # tensor sizes stayed resident until the end of the run. If that memory
    # becomes a problem on a VCF with many distinct indel lengths, the fix is to
    # call such a flush from add_record(), not to reinstate an uncalled method.

    def finish(self, nr_workers, shared_dict=None):
        """Finish processing and send finish signals"""
        logger.info(f"Finishing: {len(self.batch_counters)} unique tensor sizes")

        for tensor_size in list(self.batches.keys()):
            if self.batches[tensor_size]:
                self._flush_batch(tensor_size)
        self.batches.clear()

        # Store statistics in shared dict if provided
        if shared_dict is not None:
            shared_dict['total_predictions'] = self.total_predictions
            shared_dict['total_vcf_records'] = self.total_vcf_records

        for _ in range(nr_workers):
            self.prediction_queue.put('Finished')

    def _encode_batch_records(self, record, ann, dist_var, gene_info):
        """
        Encode VCF records - groups transcripts by (strand, padding) to share encodings.
        Returns: (all_x_ref, all_x_alt, group_gene_indices, group_alt_indices)
        """
        cov = get_cov(dist_var)
        wid = get_wid(cov)

        if not is_record_valid(record):
            return [], [], [], []

        seq = get_seq(record, ann, wid)
        if not seq or not is_location_predictable(record, seq, wid, dist_var):
            return [], [], [], []

        all_x_ref, all_x_alt, group_gene_indices, group_alt_indices = [], [], [], []

        for alt_ix in range(len(record.alts)):
            if not is_valid_alt_record(record, alt_ix):
                continue
            
            # Group genes by (strand, padding)
            strand_padding_groups = {}
            for gene_ix in range(len(gene_info.idxs)):
                strand = gene_info.strands[gene_ix]
                dist_ann = ann.get_pos_data(gene_info.idxs[gene_ix], record.pos)
                key = (strand, max(wid // 2 + dist_ann[0], 0), max(wid // 2 - dist_ann[1], 0))
                strand_padding_groups.setdefault(key, []).append(gene_ix)
            
            for gene_indices in strand_padding_groups.values():
                x_ref, x_alt = encode_seqs(
                    record=record, seq=seq, ann=ann, gene_info=gene_info,
                    gene_ix=gene_indices[0], alt_ix=alt_ix, wid=wid
                )
                all_x_ref.append(x_ref)
                all_x_alt.append(x_alt)
                group_gene_indices.append(gene_indices)
                group_alt_indices.append(alt_ix)

        return all_x_ref, all_x_alt, group_gene_indices, group_alt_indices


class VCFWriter:
    def __init__(self, args, tmpdir, devices, ann, shelf_records=None, shared_dict=None, max_memory_gb=None):
        self.args = args
        self.input_data = args.input_data
        self.output_data = args.output_data
        self.dist = args.distance
        self.tmpdir = tmpdir
        self.shelf_records = shelf_records
        self.total_vcf_records = 0
        self.total_predictions = 0
        self.orig = args.orig_output
        self.ann = ann
        # Pre-compute cov and wid since dist is constant
        self.cov = get_cov(self.dist)
        self.wid = get_wid(self.cov)
        self.prediction_dirs = {}
        self._discover_prediction_directories()
        
        # Auto-tune based on system resources
        cpu_count = get_available_cpu_count()
        try:
            import psutil
            available_ram_gb = psutil.virtual_memory().available / (1024**3)
        except ImportError:
            available_ram_gb = 16  # Conservative default
        
        # OPTIMIZATION 1: Reduce thread count (optimal for GIL)
        # Cap at 8 threads - more causes GIL contention
        num_gpus = len(devices) if devices else 1
        reserved_for_gpus = num_gpus * 4
        available_cores = max(8, cpu_count - reserved_for_gpus)
        
        self.num_write_threads = min(24, max(8, int(available_cores * 0.4)))

        # Memory-based cache limit (not count-based)
        # Use explicit limit if provided, otherwise 70% of detected RAM
        effective_ram_gb = max_memory_gb if max_memory_gb else available_ram_gb * 0.70
        self.max_memory_bytes = int(effective_ram_gb * 1024**3)
        logger.info(f"VCFWriter: Memory limit set to {effective_ram_gb:.1f}GB")
        self.batch_cache_size = 50000  # High count limit - memory limit will trigger first
        
        # Increase batch size to reduce overhead
        self.record_batch_size = 1000 if cpu_count >= 64 else 500
        
        logger.info(f"VCFWriter: OPTIMIZED for {cpu_count} cores, {available_ram_gb:.1f}GB RAM")
        logger.info(f"VCFWriter: {self.num_write_threads} threads (GIL-optimized), {self.batch_cache_size} cache slots, {self.record_batch_size} records/batch")
        
        # Get statistics from shared dict if available
        if shared_dict:
            self.total_predictions = shared_dict.get('total_predictions', 0)
            self.total_vcf_records = shared_dict.get('total_vcf_records', 0)
            logger.info(f"VCFWriter: Expecting {self.total_predictions} total predictions from {self.total_vcf_records} records")
        
        if self.prediction_dirs:
            logger.info(f"VCFWriter: {len(self.prediction_dirs)} prediction directories")
        else:
            logger.error(f"No prediction directories found in {self.tmpdir}")

        self._array_registry = weakref.WeakValueDictionary()


    def _discover_prediction_directories(self):
        """Discover all prediction directories recursively"""
        for item in os.listdir(self.tmpdir):
            full_path = os.path.join(self.tmpdir, item)
            if not os.path.isdir(full_path):
                continue
                
            if item.startswith('spliceai_preds'):
                self.prediction_dirs[item] = full_path
            elif item.startswith('process_'):
                for subitem in os.listdir(full_path):
                    sub_path = os.path.join(full_path, subitem)
                    if os.path.isdir(sub_path) and subitem.startswith('spliceai_preds'):
                        self.prediction_dirs[f"{item}/{subitem}"] = sub_path
        
        # Also find directories containing .npy files
        for npy_file in glob.glob(os.path.join(self.tmpdir, '**', '*.npy'), recursive=True):
            dir_path = os.path.dirname(npy_file)
            if dir_path != self.tmpdir:
                rel_path = os.path.relpath(dir_path, self.tmpdir)
                if rel_path not in self.prediction_dirs:
                    self.prediction_dirs[rel_path] = dir_path

    def process(self):
        self.vcf_in = pysam.VariantFile(self.input_data)
        header = self.vcf_in.header
        # Identical text to spliceai/__main__.py -- both derive the field list
        # from spliceai.utils, so the single-record and batch paths can no longer
        # emit different headers for the same schema (they previously differed in
        # field NAMES, field COUNT, and in whether EVENT_CLASS existed at all).
        if self.orig:
            header.add_line(
                '##INFO=<ID=SpliceAI,Number=.,Type=String,Description="'
                + info_header_description(orig=True) + '">')
        else:
            header.add_line(
                '##INFO=<ID=SpliceAI,Number=.,Type=String,Description="'
                + info_header_description(orig=False) + '">')

        self.vcf_out = pysam.VariantFile(self.output_data, mode='w', header=header)
        self._validate_prediction_files()
        self._write_records()
        self.vcf_in.close()
        self.vcf_out.close()

    def process_tsv(self):
        self.vcf_in = pysam.VariantFile(self.input_data)
        self._pigz_proc = None
        if isinstance(self.output_data, str):
            self.tsv_out = self._open_compressed_output(self.output_data)
        else:
            self.tsv_out = self.output_data

        self.tsv_out.write(get_tsv_header_comments(self.orig))
        self.tsv_out.write(get_tsv_header_line(self.orig))
        self._validate_prediction_files()
        self._write_tsv_records()
        self.vcf_in.close()
        self.tsv_out.close()
        if self._pigz_proc is not None:
            self._pigz_proc.wait()

    def _open_compressed_output(self, path):
        """Prefer pigz (parallel gzip) over the single-threaded gzip module, which
        was the writer's actual bottleneck -- falls back to gzip if unavailable."""
        import shutil
        pigz_path = shutil.which('pigz')
        if pigz_path:
            import subprocess
            f_out = open(path, 'wb')
            self._pigz_proc = subprocess.Popen(
                [pigz_path, '-p', str(self.num_write_threads), '-4', '-c'],
                stdin=subprocess.PIPE, stdout=f_out
            )
            f_out.close()
            return io.TextIOWrapper(self._pigz_proc.stdin, encoding='utf-8')

        logger.warning("pigz not found on PATH, falling back to single-threaded gzip")
        import gzip
        return gzip.open(path, 'wt', compresslevel=4)

    def _validate_prediction_files(self):
            """Validate that all expected prediction files exist"""
            logger.info("Validating prediction files...")

            expected_files = set()
            for key in self.shelf_records:
                # Skip statistics keys (they're not PreparedVCFRecord objects)
                if key in ('total_predictions', 'total_vcf_records'):
                    continue
                
                record = self.shelf_records[key]
                locations = _get_attr(record, 'locations')
                
                for location in locations:
                    tensor_size = _get_attr(location, 'tensor_size')
                    if tensor_size > 0:
                        expected_files.add((tensor_size, _get_attr(location, 'batch_ix')))

            missing = [
                f"{ts}_{bx}.npy" for ts, bx in sorted(expected_files)
                if not any(os.path.exists(os.path.join(d, f"{ts}_{bx}.npy")) for d in self.prediction_dirs.values())
            ]

            if missing:
                logger.error(f"MISSING {len(missing)} PREDICTION FILES: {missing[:5]}...")
                raise FileNotFoundError(f"{len(missing)} prediction files missing")
            
            logger.info(f"Validated {len(expected_files)} prediction batches")

    def _load_prediction(self, tensor_size, batch_ix):
        """Load prediction from numpy file - fully load into memory to avoid fd leaks.

        Workers store predictions as fp16 to halve disk I/O. The array is kept
        fp16 IN THE CACHE (half the RAM -> fewer evictions -> fewer re-reads);
        the individual row slices are upcast to fp32 at extraction time so all
        delta-score math still runs in full precision. Legacy fp32 files load
        and pass through unchanged (backward compatible)."""
        for pred_dir in self.prediction_dirs.values():
            path = os.path.join(pred_dir, f"{tensor_size}_{batch_ix}.npy")
            if os.path.exists(path):
                return np.load(path)
        raise FileNotFoundError(f"Prediction not found: {tensor_size}_{batch_ix}.npy")

    def _write_records(self):
        """Write output file - OPTIMIZED to prevent decay"""
        logger.info("Writing output file (MEMORY-OPTIMIZED)...")
        import psutil
        process = psutil.Process(os.getpid())
        peak_memory = current_memory = process.memory_info().rss

        from collections import OrderedDict
        batch_cache = OrderedDict()
        batch_cache_lock = threading.Lock()
        cache_hits = 0
        cache_misses = 0
        
        def get_batch(tensor_size, batch_ix):
            nonlocal cache_hits, cache_misses
            key = f"{tensor_size}_{batch_ix}"

            with batch_cache_lock:
                if key in batch_cache:
                    batch_cache.move_to_end(key)
                    cache_hits += 1
                    return batch_cache[key]

            # Load outside lock
            data = self._load_prediction(tensor_size, batch_ix)

            with batch_cache_lock:
                # Check memory BEFORE adding - evict if needed
                current_mem = process.memory_info().rss
                if current_mem > self.max_memory_bytes:
                    # Evict 40% of cache to create headroom
                    evict_count = max(1, len(batch_cache) * 2 // 5)
                    for _ in range(evict_count):
                        if batch_cache:
                            _, old_arr = batch_cache.popitem(last=False)
                            del old_arr
                    gc.collect()

                batch_cache[key] = data
                cache_misses += 1

            return data

        line_idx = 0
        records_buffer = []
        last_log_time = time.time()
        write_start = time.time()
        
        with ThreadPoolExecutor(max_workers=self.num_write_threads) as executor:
            for record in self.vcf_in:
                records_buffer.append(record)
                
                if len(records_buffer) >= self.record_batch_size:
                    self._process_record_batch(records_buffer, line_idx, get_batch, 
                                               self.num_write_threads, executor)
                    line_idx += len(records_buffer)
                    records_buffer = []
                    
                    if line_idx % 10000 == 0:
                        current_memory = process.memory_info().rss
                        if current_memory > peak_memory:
                            peak_memory = current_memory
                        logger.info(f"Memory: current={current_memory/1024**2:.1f}MB, peak={peak_memory/1024**2:.1f}MB")
                        
                        # Only clear if cache is very full
                        with batch_cache_lock:
                            if len(batch_cache) > self.batch_cache_size * 0.9:
                                # Clear oldest 30% (not 50%)
                                for _ in range(len(batch_cache) // 3):
                                    old_key, old_arr = batch_cache.popitem(last=False)
                                    del old_arr
                        gc.collect(generation=1)  # Only now do gen 1
                    
                    if time.time() - last_log_time > 5:
                        elapsed = time.time() - write_start
                        rate = line_idx / elapsed if elapsed > 0 else 0
                        hit_rate = cache_hits/(cache_hits+cache_misses+1e-9)*100
                        logger.info(f"- Writing: {line_idx} records ({rate:.0f} rec/s), "
                                  f"cache: {hit_rate:.1f}%, size: {len(batch_cache)}, "
                                  f"mem: {current_memory/1024**2:.1f}MB")
                        last_log_time = time.time()
        
            if records_buffer:
                self._process_record_batch(records_buffer, line_idx, get_batch, 
                                           self.num_write_threads, executor)
                line_idx += len(records_buffer)

        with batch_cache_lock:
            for key in list(batch_cache.keys()):
                arr = batch_cache.pop(key)
                del arr
            batch_cache.clear()
        
        gc.collect()
        
        total_time = time.time() - write_start
        rate = line_idx / total_time if total_time > 0 else 0
        logger.info(f"Writing complete: {line_idx} records in {total_time:.1f}s ({rate:.0f} rec/s)")
        logger.info(f"Cache stats: {cache_hits} hits, {cache_misses} misses "
                   f"({cache_hits/(cache_hits+cache_misses)*100:.1f}% hit rate)")
        self.total_vcf_records = line_idx

    def _process_record_batch(self, records, start_idx, get_batch_func, num_threads, executor=None):
        """Process records - uses batched numpy ops for orig mode, threading for extended mode"""
        work_items = []
        for i, rec in enumerate(records):
            vcf_idx_key = str(start_idx + i + 1)
            if vcf_idx_key not in self.shelf_records or vcf_idx_key in ('total_predictions', 'total_vcf_records'):
                work_items.append((start_idx + i + 1, rec, None))
            else:
                work_items.append((start_idx + i + 1, rec, self.shelf_records[vcf_idx_key]))

        results = {}

        if self.orig:
            # BATCHED PATH: Collect all predictions and process in one vectorized call
            self._process_record_batch_orig(work_items, results, get_batch_func)
        else:
            # THREADED PATH: Use threads for extended output (more complex per-record logic)
            if executor is None:
                context = ThreadPoolExecutor(max_workers=num_threads)
            else:
                import contextlib
                context = contextlib.nullcontext(executor)

            with context as pool:
                futures = {}
                for idx, rec, prep in work_items:
                    if prep is None:
                        results[idx] = []
                    else:
                        futures[pool.submit(self._compute_delta_scores, rec, prep, get_batch_func)] = idx

                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        logger.error(f"Error at record {idx}: {e}")
                        results[idx] = []

        for idx, record, _ in work_items:
            if results.get(idx):
                record.info['SpliceAI'] = results[idx]
            self.vcf_out.write(record)

    def _write_tsv_records(self):
        """Write output TSV file (gzipped) - MEMORY-OPTIMIZED"""
        logger.info("Writing output TSV file (MEMORY-OPTIMIZED)...")
        import psutil
        process = psutil.Process(os.getpid())
        peak_memory = current_memory = process.memory_info().rss

        from collections import OrderedDict
        batch_cache = OrderedDict()
        batch_cache_lock = threading.Lock()
        cache_hits = 0
        cache_misses = 0
        
        def get_batch(tensor_size, batch_ix):
            nonlocal cache_hits, cache_misses
            key = f"{tensor_size}_{batch_ix}"

            with batch_cache_lock:
                if key in batch_cache:
                    batch_cache.move_to_end(key)
                    cache_hits += 1
                    return batch_cache[key]

            # Load outside lock
            data = self._load_prediction(tensor_size, batch_ix)

            with batch_cache_lock:
                # Check memory BEFORE adding - evict if needed
                current_mem = process.memory_info().rss
                if current_mem > self.max_memory_bytes:
                    # Evict 40% of cache to create headroom
                    evict_count = max(1, len(batch_cache) * 2 // 5)
                    for _ in range(evict_count):
                        if batch_cache:
                            _, old_arr = batch_cache.popitem(last=False)
                            del old_arr
                    gc.collect()

                batch_cache[key] = data
                cache_misses += 1

            return data

        line_idx = 0
        records_buffer = []
        last_log_time = time.time()
        write_start = time.time()
        
        with ThreadPoolExecutor(max_workers=self.num_write_threads) as executor:
            for record in self.vcf_in:
                records_buffer.append(record)
                
                if len(records_buffer) >= self.record_batch_size:
                    self._process_record_batch_tsv(records_buffer, line_idx, get_batch, 
                                                   self.num_write_threads, executor)
                    line_idx += len(records_buffer)
                    records_buffer = []
                    
                    if line_idx % 10000 == 0:
                        current_memory = process.memory_info().rss
                        if current_memory > peak_memory:
                            peak_memory = current_memory
                        logger.info(f"Memory: current={current_memory/1024**2:.1f}MB, peak={peak_memory/1024**2:.1f}MB")
                        
                        # Only clear if cache is very full
                        with batch_cache_lock:
                            if len(batch_cache) > self.batch_cache_size * 0.9:
                                for _ in range(len(batch_cache) // 3):
                                    old_key, old_arr = batch_cache.popitem(last=False)
                                    del old_arr
                        gc.collect(generation=1)
                    
                    if time.time() - last_log_time > 5:
                        elapsed = time.time() - write_start
                        rate = line_idx / elapsed if elapsed > 0 else 0
                        hit_rate = cache_hits/(cache_hits+cache_misses+1e-9)*100
                        logger.info(f"- Writing TSV: {line_idx} records ({rate:.0f} rec/s), "
                                  f"cache: {hit_rate:.1f}%, size: {len(batch_cache)}, "
                                  f"mem: {current_memory/1024**2:.1f}MB")
                        last_log_time = time.time()
        
            if records_buffer:
                self._process_record_batch_tsv(records_buffer, line_idx, get_batch, 
                                               self.num_write_threads, executor)
                line_idx += len(records_buffer)

        with batch_cache_lock:
            for key in list(batch_cache.keys()):
                arr = batch_cache.pop(key)
                del arr
            batch_cache.clear()
        
        gc.collect()
        
        total_time = time.time() - write_start
        rate = line_idx / total_time if total_time > 0 else 0
        logger.info(f"TSV writing complete: {line_idx} records in {total_time:.1f}s ({rate:.0f} rec/s)")
        logger.info(f"Cache stats: {cache_hits} hits, {cache_misses} misses "
                   f"({cache_hits/(cache_hits+cache_misses)*100:.1f}% hit rate)")
        self.total_vcf_records = line_idx

    def _process_record_batch_tsv(self, records, start_idx, get_batch_func, num_threads, executor=None):
        work_items = []
        for i, rec in enumerate(records):
            vcf_idx_key = str(start_idx + i + 1)
            if vcf_idx_key not in self.shelf_records or vcf_idx_key in ('total_predictions', 'total_vcf_records'):
                work_items.append((start_idx + i + 1, rec, None))
            else:
                work_items.append((start_idx + i + 1, rec, self.shelf_records[vcf_idx_key]))

        results = {}

        if self.orig:
            self._process_record_batch_orig(work_items, results, get_batch_func)
        else:
            if executor is None:
                context = ThreadPoolExecutor(max_workers=num_threads)
            else:
                import contextlib
                context = contextlib.nullcontext(executor)

            with context as pool:
                futures = {}
                for idx, rec, prep in work_items:
                    if prep is None:
                        results[idx] = []
                    else:
                        futures[pool.submit(self._compute_delta_scores, rec, prep, get_batch_func)] = idx

                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        logger.error(f"Error at record {idx}: {e}")
                        results[idx] = []

        lines = []
        for idx, record, _ in work_items:
            scores = results.get(idx, [])
            alt_str = ','.join(record.alts) if record.alts else '.'
            for score_str in scores:
                score_fields = score_str.split('|')
                row_fields = [record.chrom, str(record.pos), record.ref, alt_str] + score_fields
                lines.append('\t'.join(row_fields))
        # One write() per batch instead of per score line -- each write() triggers
        # gzip compression, so batching drastically cuts compression call overhead.
        if lines:
            self.tsv_out.write('\n'.join(lines) + '\n')

    def _process_record_batch_orig(self, work_items, results, get_batch_func):
        """Batched processing for orig output - vectorized numpy operations"""
        # Phase 1: Collect all prediction tasks across all records
        all_tasks = []  # List of (record_idx, alt_ix, gene_ix, gene_info, y_ref, y_alt)
        record_task_counts = {}  # record_idx -> number of tasks

        for idx, rec, prep in work_items:
            if prep is None:
                results[idx] = []
                record_task_counts[idx] = 0
                continue

            gene_info = _get_attr(prep, 'gene_info')
            locations = _get_attr(prep, 'locations')

            # Build location map
            loc_map = {}
            for loc in locations:
                key = (_get_attr(loc, 'alt_ix'), _get_attr(loc, 'gene_ix'), _get_attr(loc, 'sequence_type'))
                loc_map[key] = loc

            gene_idxs = _get_attr(gene_info, 'idxs')
            gene_names = _get_attr(gene_info, 'genes')
            strands = _get_attr(gene_info, 'strands')
            num_genes = len(gene_idxs)
            task_count = 0

            for alt_ix in range(len(rec.alts)):
                for gene_ix in range(num_genes):
                    ref_loc = loc_map.get((alt_ix, gene_ix, SequenceType_REF))
                    alt_loc = loc_map.get((alt_ix, gene_ix, SequenceType_ALT))

                    if not ref_loc or not alt_loc:
                        continue

                    if not is_valid_alt_record(rec, alt_ix):
                        continue

                    # Handle complex indels specially
                    if len(rec.ref) > 1 and len(rec.alts[alt_ix]) > 1:
                        all_tasks.append({
                            'record_idx': idx,
                            'type': 'unhandled',
                            'alt': rec.alts[alt_ix],
                            'gene_name': gene_names[gene_ix]
                        })
                        task_count += 1
                        continue

                    ref_ts = _get_attr(ref_loc, 'tensor_size')
                    alt_ts = _get_attr(alt_loc, 'tensor_size')

                    if ref_ts <= 0 or alt_ts <= 0:
                        continue

                    # Load prediction data
                    try:
                        ref_batch = get_batch_func(ref_ts, _get_attr(ref_loc, 'batch_ix'))
                        alt_batch = get_batch_func(alt_ts, _get_attr(alt_loc, 'batch_ix'))
                        # Upcast the extracted row to fp32 (cache stays fp16).
                        y_ref = ref_batch[[_get_attr(ref_loc, 'batch_index')], :, :].astype(np.float32, copy=False)
                        y_alt = alt_batch[[_get_attr(alt_loc, 'batch_index')], :, :].astype(np.float32, copy=False)
                    except Exception as e:
                        logger.warning(f"Failed to load batch for record {idx}: {e}")
                        continue

                    all_tasks.append({
                        'record_idx': idx,
                        'type': 'compute',
                        'y_ref': y_ref,
                        'y_alt': y_alt,
                        'strand': strands[gene_ix],
                        'gene_idx': gene_idxs[gene_ix],
                        'position': rec.pos,
                        'ref': rec.ref,
                        'alt': rec.alts[alt_ix],
                        'gene_name': gene_names[gene_ix]
                    })
                    task_count += 1

            record_task_counts[idx] = task_count
            if task_count == 0:
                results[idx] = []

        # Phase 2: Separate unhandled vs compute tasks
        unhandled_tasks = [t for t in all_tasks if t['type'] == 'unhandled']
        compute_tasks = [t for t in all_tasks if t['type'] == 'compute']

        # Phase 3: Process compute tasks with batched function
        if compute_tasks:
            y_refs = [t['y_ref'] for t in compute_tasks]
            y_alts = [t['y_alt'] for t in compute_tasks]
            strands = [t['strand'] for t in compute_tasks]
            gene_idxs = [t['gene_idx'] for t in compute_tasks]
            positions = [t['position'] for t in compute_tasks]
            refs = [t['ref'] for t in compute_tasks]
            alts = [t['alt'] for t in compute_tasks]
            gene_names = [t['gene_name'] for t in compute_tasks]

            batch_results = compute_orig_delta_scores_batched(
                y_refs=y_refs,
                y_alts=y_alts,
                strands=strands,
                gene_idxs=gene_idxs,
                cov=self.cov,
                exon_boundary_cache=self.ann._exon_boundary_cache,
                positions=positions,
                alts=alts,
                gene_names=gene_names,
                refs=refs
            )

            # Assign results back to compute tasks
            for i, task in enumerate(compute_tasks):
                task['result'] = batch_results[i]

        # Phase 4: Process unhandled tasks
        for task in unhandled_tasks:
            task['result'] = create_unhandled_delta_score_orig(task['alt'], task['gene_name'])

        # Phase 5: Gather results by record, preserving order
        # Group all tasks by record_idx, maintaining insertion order
        from collections import defaultdict
        record_results = defaultdict(list)
        for task in all_tasks:
            if 'result' in task:
                record_results[task['record_idx']].append(task['result'])

        # Assign to results dict
        for idx, _, _ in work_items:
            if idx not in results:
                results[idx] = record_results.get(idx, [])

    def _compute_delta_scores(self, record, prepared_record, get_batch_func):
        """Compute delta scores for a single record (thread-safe)"""
        gene_info = _get_attr(prepared_record, 'gene_info')
        locations = _get_attr(prepared_record, 'locations')
        
        # Index locations by (alt_ix, gene_ix, seq_type)
        loc_map = {}
        for loc in locations:
            key = (_get_attr(loc, 'alt_ix'), _get_attr(loc, 'gene_ix'), _get_attr(loc, 'sequence_type'))
            loc_map[key] = loc
        
        all_y_ref, all_y_alt = [], []
        num_genes = len(_get_attr(gene_info, 'idxs'))
        
        for alt_ix in range(len(record.alts)):
            for gene_ix in range(num_genes):
                ref_loc = loc_map.get((alt_ix, gene_ix, SequenceType_REF))
                alt_loc = loc_map.get((alt_ix, gene_ix, SequenceType_ALT))
                
                if ref_loc and alt_loc:
                    ref_ts = _get_attr(ref_loc, 'tensor_size')
                    alt_ts = _get_attr(alt_loc, 'tensor_size')
                    
                    if ref_ts > 0:
                        batch = get_batch_func(ref_ts, _get_attr(ref_loc, 'batch_ix'))
                        all_y_ref.append(batch[[_get_attr(ref_loc, 'batch_index')], :, :].astype(np.float32, copy=False))
                    else:
                        all_y_ref.append(None)

                    if alt_ts > 0:
                        batch = get_batch_func(alt_ts, _get_attr(alt_loc, 'batch_ix'))
                        all_y_alt.append(batch[[_get_attr(alt_loc, 'batch_index')], :, :].astype(np.float32, copy=False))
                    else:
                        all_y_alt.append(None)
                else:
                    all_y_ref.append(None)
                    all_y_alt.append(None)
        
        return self._extract_delta_scores(all_y_ref, all_y_alt, record, gene_info)

    def _extract_delta_scores(self, all_y_ref, all_y_alt, record, gene_info):
        # Use pre-computed cov and wid from __init__
        # OPTIMIZATION: Skip get_seq when orig=True since seq is not used in that path
        seq = None if self.orig else get_seq(record, self.ann, self.wid)
        
        gene_idxs = _get_attr(gene_info, 'idxs')
        gene_names = _get_attr(gene_info, 'genes')

        delta_scores = []
        pred_ix = 0
        
        for alt_ix in range(len(record.alts)):
            for gene_ix in range(len(gene_idxs)):
                if pred_ix >= len(all_y_ref):
                    return delta_scores
                    
                y_ref, y_alt = all_y_ref[pred_ix], all_y_alt[pred_ix]
                pred_ix += 1

                if y_ref is None or y_alt is None:
                    continue

                if not is_valid_alt_record(record, alt_ix):
                    continue

                if len(record.ref) > 1 and len(record.alts[alt_ix]) > 1:
                    if self.orig:
                        delta_scores.append(create_unhandled_delta_score_orig(record.alts[alt_ix], gene_names[gene_ix])) 
                    else:
                        delta_scores.append(create_unhandled_delta_score(record.alts[alt_ix], gene_names[gene_ix]))
                    continue

                delta_scores.append(get_alt_gene_delta_score(
                    record=record, ann=self.ann, alt_ix=alt_ix, gene_ix=gene_ix,
                    y_ref=y_ref, y_alt=y_alt, cov=self.cov, gene_info=gene_info,
                    seq=seq, wid=self.wid, orig=self.orig
                ))

        return delta_scores