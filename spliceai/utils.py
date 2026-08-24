# Original source code modified to add prediction batching support by Invitae in 2021.
# Modifications copyright (c) 2021 Invitae Corporation.

# Converted to PyTorch with FP16/BF16 support

import os
import collections
from importlib.resources import files
import pandas as pd
import numpy as np
from pyfaidx import Fasta
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import gc
from intervaltree import IntervalTree
import time
from collections import defaultdict
from functools import lru_cache

GeneInfo = collections.namedtuple('GeneInfo', 'genes strands idxs')


def get_available_cpu_count():
    """Return the number of CPUs this process can actually use.

    ``multiprocessing.cpu_count()`` / ``os.cpu_count()`` report the *machine's*
    total core count, ignoring any cgroup/cpuset restriction the process is
    confined to (e.g. a SLURM allocation with ``--cpus-per-task=1`` on a
    16-core node, or a Docker container started with ``--cpuset-cpus``). On
    such systems this mismatch causes worker/thread counts to be sized for
    the whole machine when only a handful of cores (or just one) are actually
    available, which doesn't just fail to help -- it makes throughput *flat*
    regardless of thread/batch settings, since every extra thread is time-sliced
    across the same tiny cpuset with pure scheduling overhead and no added
    parallelism.

    ``os.sched_getaffinity(0)`` reports the actual set of CPUs the calling
    process is allowed to run on, which correctly reflects cpuset/affinity
    restrictions. It's Linux-only, so fall back to the machine-wide count on
    platforms (e.g. macOS) where it doesn't exist.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 8



# --- Splice-event classification thresholds ---
# These gate the new classify_splice_event logic. Tuned conservatively: a
# "canonical" MANE site must have a real model score, not model noise.
MIN_CANONICAL_SS_SCORE = 0.50   # canonical donor/acceptor must have ref score >= this
MIN_NEW_SS_SCORE       = 0.50   # a new/shifted donor/acceptor must have alt score >= this
COMPETING_MIN_ABS      = 0.50   # competing site must have an absolute score >= this
COMPETING_REL_TOL      = 0.10   # ... and be within this fraction of the top score
# NMD_ESCAPE_WINDOW_NT is applied to the EVENT position, not to a predicted
# premature stop codon: no translation happens anywhere in this codebase, so
# the true PTC position is unknown. A 'Frameshift_NMD' label therefore means
# "the splice event lies more than 55 nt upstream of the last junction", which
# is a proxy for NMD, not a prediction of it. The bias is one-directional: a
# frameshift whose actual PTC falls inside the escape window is still labelled
# Frameshift_NMD. See the INFO description for EVENT_CLASS and the README.
NMD_ESCAPE_WINDOW_NT   = 55     # event >this far upstream of last junction -> NMD proxy

# Cache-miss sentinel, distinct from None (which is a legitimate cached value
# meaning "this transcript has no exon-exon junction").
_CACHE_MISS = object()
# (MAX_NMD_SCAN_NT removed: it capped "how much mRNA we translate looking for a
#  PTC" — a translation step that does not exist. It was never read.)

# --- INFO field schemas (single source of truth) ---
#
# The header Description and the record format string are BOTH derived from
# these tuples, so they cannot drift apart. Previously three different field
# lists existed: __main__.py declared 26 tokens, data_handlers.py declared 29,
# and the records carried 23 -- so any consumer that split the Format string
# onto the data columns mis-assigned every field past SYMBOL.
#
# No field name may contain '|': that is the record separator. The EVENT_CLASS
# enumeration therefore uses '/' internally, where it previously used '|' and
# produced seven spurious trailing field names.

#: Extended ("print everything") schema -- 23 fields.
INFO_FIELDS_EXTENDED = (
    'ALLELE', 'SYMBOL',
    'DSM_AG', 'DSM_AL', 'DSM_DG', 'DSM_DL',
    'DS_AG', 'DS_AL', 'DS_DG', 'DS_DL',
    'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL',
    'RS_AG', 'RS_AL', 'RS_DG', 'RS_DL',
    'MANEselect Donor splice sites within context (DP,RS_REF,RS_ALT)',
    'MANEselect Acceptor splice sites within context (DP,RS_REF,RS_ALT)',
    'Donor Sites with Raw Score > 0.5 (DP,RS_REF,RS_ALT)',
    'Acceptor Sites with Raw Score > 0.5 (DP,RS_REF,RS_ALT)',
    'EVENT_CLASS(NoChange/InFrame/Frameshift_NMD/Frameshift_NMDescape/'
    'Pseudoexon/Ambiguous)',
    'N_EXONS',
    'PREDICTED_EVENT'
)

#: --orig_output schema -- 14 fields, matching what the orig fast path emits
#: (masked DS x4, raw DS x4, DP x4, plus ALLELE and SYMBOL). The old header
#: declared only the 10 v1.3.1 names and omitted the masked scores entirely.
INFO_FIELDS_ORIG = (
    'ALLELE', 'SYMBOL',
    'DSM_AG', 'DSM_AL', 'DSM_DG', 'DSM_DL',
    'DS_AG', 'DS_AL', 'DS_DG', 'DS_DL',
    'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL',
)

for _schema_name, _schema in (('INFO_FIELDS_EXTENDED', INFO_FIELDS_EXTENDED),
                              ('INFO_FIELDS_ORIG', INFO_FIELDS_ORIG)):
    _bad = [f for f in _schema if '|' in f]
    if _bad:
        raise AssertionError(
            f"{_schema_name} contains '|' inside field name(s) {_bad}; '|' is the "
            f"INFO record separator and must not appear in a field name."
        )
del _schema_name, _schema, _bad


def info_format_string(orig=False):
    """Return the 'Format: a|b|c' fragment for the selected output schema."""
    fields = INFO_FIELDS_ORIG if orig else INFO_FIELDS_EXTENDED
    return 'Format: ' + '|'.join(fields)


def n_info_fields(orig=False):
    """Number of '|'-separated fields the selected schema emits."""
    return len(INFO_FIELDS_ORIG if orig else INFO_FIELDS_EXTENDED)


#: Description text for the ##INFO header line. Defined here so the batched
#: writer (batch/data_handlers.py) and the single-process writer (__main__.py)
#: cannot drift apart -- they previously carried two independent copies of this
#: paragraph.
#:
#: The EVENT_CLASS sentence states what the label is derived from. The previous
#: wording, "an NMD-aware splice-event classification", implied the caller had
#: predicted a premature termination codon and asked whether NMD would act on
#: it. No translation is performed anywhere in this codebase; the escape-window
#: test is applied to the position of the splice EVENT. Downstream filtering on
#: Frameshift_NMD is only as sound as that proxy, so the header says so.
_EVENT_CLASS_NOTE = (
    'EVENT_CLASS gives a frame-consequence call for the predicted event. '
    'Frameshift_NMD/Frameshift_NMDescape are assigned by testing whether the '
    'EVENT lies more than {} nt upstream of the last exon-exon junction; no '
    'premature stop codon is predicted, so these labels are a positional proxy '
    'for NMD and not a prediction of transcript decay. '
)


def info_header_description(orig=False):
    """Return the ##INFO Description string for the selected output schema."""
    if orig:
        return (
            'SpliceAIv1.3.1-compatible variant annotation. Delta scores (DS), masked '
            'delta scores (DSM) and delta positions (DP) for acceptor gain (AG), '
            'acceptor loss (AL), donor gain (DG) and donor loss (DL). '
            + info_format_string(orig=True))
    return (
        'SpliceAImod variant annotation. These include delta '
        'scores (DS), masked delta scores (DSM), delta positions (DP) and raw '
        'scores (RS) for acceptor gain (AG), acceptor loss (AL), donor gain (DG) '
        'and donor loss (DL). Additional fields include MANEselect splice sites '
        'within the context window and all sites with raw score >0.5. '
        + _EVENT_CLASS_NOTE.format(NMD_ESCAPE_WINDOW_NT)
        + info_format_string(orig=False))


def get_tsv_header_comments(orig=False):
    """Return commented lines describing output columns for TSV mode."""
    comments = [
        "# " + info_header_description(orig=orig),
        "# Description of columns:",
        "# CHROM: Chromosome",
        "# POS: 1-based position",
        "# REF: Reference allele",
        "# ALT: Alternative allele(s) in VCF record",
        "# ALLELE: Alternative allele evaluated",
        "# SYMBOL: Gene symbol",
    ]
    if orig:
        comments.extend([
            "# DSM_AG: Masked delta score for acceptor gain",
            "# DSM_AL: Masked delta score for acceptor loss",
            "# DSM_DG: Masked delta score for donor gain",
            "# DSM_DL: Masked delta score for donor loss",
            "# DS_AG: Delta score for acceptor gain",
            "# DS_AL: Delta score for acceptor loss",
            "# DS_DG: Delta score for donor gain",
            "# DS_DL: Delta score for donor loss",
            "# DP_AG: Delta position for acceptor gain",
            "# DP_AL: Delta position for acceptor loss",
            "# DP_DG: Delta position for donor gain",
            "# DP_DL: Delta position for donor loss",
        ])
    else:
        comments.extend([
            "# DSM_AG: Masked delta score for acceptor gain",
            "# DSM_AL: Masked delta score for acceptor loss",
            "# DSM_DG: Masked delta score for donor gain",
            "# DSM_DL: Masked delta score for donor loss",
            "# DS_AG: Delta score for acceptor gain",
            "# DS_AL: Delta score for acceptor loss",
            "# DS_DG: Delta score for donor gain",
            "# DS_DL: Delta score for donor loss",
            "# DP_AG: Delta position for acceptor gain",
            "# DP_AL: Delta position for acceptor loss",
            "# DP_DG: Delta position for donor gain",
            "# DP_DL: Delta position for donor loss",
            "# RS_AG: Raw score for acceptor gain",
            "# RS_AL: Raw score for acceptor loss",
            "# RS_DG: Raw score for donor gain",
            "# RS_DL: Raw score for donor loss",
            "# MANEselectDonors: MANEselect donor splice sites within context window (DP,RS_REF,RS_ALT)",
            "# MANEselectAcceptors: MANEselect acceptor splice sites within context window (DP,RS_REF,RS_ALT)",
            "# DonorsRSgt0.5: Donor sites with raw score > 0.5 (DP,RS_REF,RS_ALT)",
            "# AcceptorsRSgt0.5: Acceptor sites with raw score > 0.5 (DP,RS_REF,RS_ALT)",
            "# EVENT_CLASS: Splice-event consequence classification",
            "# N_EXONS: Number of exons in the transcript",
            "# PREDICTED_EVENT: Predicted structural event (e.g. exon_skip, intron_retention, pseudoexon)",
        ])
    return "\n".join(comments) + "\n"


def get_tsv_header_line(orig=False):
    """Return the column names header line for TSV mode."""
    if orig:
        cols = [
            '#CHROM', 'POS', 'REF', 'ALT', 'ALLELE', 'SYMBOL',
            'DSM_AG', 'DSM_AL', 'DSM_DG', 'DSM_DL',
            'DS_AG', 'DS_AL', 'DS_DG', 'DS_DL',
            'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL'
        ]
    else:
        cols = [
            '#CHROM', 'POS', 'REF', 'ALT', 'ALLELE', 'SYMBOL',
            'DSM_AG', 'DSM_AL', 'DSM_DG', 'DSM_DL',
            'DS_AG', 'DS_AL', 'DS_DG', 'DS_DL',
            'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL',
            'RS_AG', 'RS_AL', 'RS_DG', 'RS_DL',
            'MANEselectDonors', 'MANEselectAcceptors',
            'DonorsRSgt0.5', 'AcceptorsRSgt0.5',
            'EVENT_CLASS', 'N_EXONS', 'PREDICTED_EVENT'
        ]
    return "\t".join(cols) + "\n"



# --- Profiling Classes ---

class ProfilingStats:
    """Aggregates profiling data and prints summaries periodically."""
    _stats = defaultdict(float)
    _counts = defaultdict(int)
    _iteration_count = 0
    _print_interval = 10000
    
    @classmethod
    def record(cls, name, duration_ms):
        cls._stats[name] += duration_ms
        cls._counts[name] += 1
        
    @classmethod
    def increment_iteration(cls):
        cls._iteration_count += 1
        if cls._iteration_count % cls._print_interval == 0:
            cls.print_stats()
            cls.reset()
            
    @classmethod
    def print_stats(cls):
        logging.info(f"=== Profiling Stats (Last {cls._print_interval} iterations) ===")
        sorted_stats = sorted(cls._stats.items(), key=lambda x: x[1], reverse=True)
        for name, total_ms in sorted_stats:
            count = cls._counts[name]
            avg_ms = total_ms / count if count > 0 else 0
            logging.info(f"{name:<25}: Avg {avg_ms:.4f} ms | Total {total_ms:.2f} ms | Count {count}")
        logging.info("="*60)

    @classmethod
    def reset(cls):
        cls._stats.clear()
        cls._counts.clear()

class BlockTimer:
    """Context manager to time code blocks and record to ProfilingStats."""
    def __init__(self, name, enabled=True):
        self.name = name
        self.enabled = enabled
    
    def __enter__(self):
        if self.enabled:
            self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            elapsed = (time.perf_counter() - self.start) * 1000  # ms
            ProfilingStats.record(self.name, elapsed)


# --- Helper Functions ---

def has_stop_codon(sequence):
    if not sequence or len(sequence) < 3:
        return False
    stop_codons = frozenset(['TAA', 'TAG', 'TGA'])
    for i in range(0, len(sequence) - 2, 3):
        codon = sequence[i:i+3].upper()
        if codon in stop_codons:
            return True
    return False

@lru_cache(maxsize=1024)
def reverse_complement(sequence):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(complement.get(base.upper(), 'N') for base in reversed(sequence))

def find_stop_in_frame(sequence, frame_offset):
    if not sequence or len(sequence) < 3:
        return None
    stop_codons = frozenset(['TAA', 'TAG', 'TGA'])
    for i in range(frame_offset, len(sequence) - 2, 3):
        codon = sequence[i:i+3].upper()
        if codon in stop_codons:
            return i
    return None

# ============================================
# PyTorch SpliceAI Model Definition
# ============================================

# ============================================
# Model architecture
# ============================================
#
# The model classes and the Keras converter previously existed here AND in
# spliceai/models/. The two SpliceAI definitions were state-dict compatible
# (238/238 tensors, bitwise-identical outputs on the shipped checkpoints) but
# only the models/ copy validates its input, so the production path — which ran
# this file's copy — raised opaque errors on a malformed batch:
#
#   2D input           -> "running_mean should contain 11001 elements not 32"
#   5 channels         -> "Given groups=1, weight of size [32, 4, 1], expected..."
#   length 10000       -> "Calculated padded input size per channel: (0)."
#
# against the models/ copy's "SpliceAI-5000 requires input sequence length >=
# 10001, got 10000". Importing from one definition removes the drift risk and
# gives the production path that validation. Cropping1D and _transfer_weights
# were dead in both copies and are gone.
from spliceai.models.pytorch_model import (      # noqa: F401  (re-exported)
    ResidualUnit,
    Skip,
    SpliceAI,
    create_spliceai_model,
)


class Annotator:
    """Annotator class with PyTorch model support"""

    def __init__(self, ref_fasta, annotations, cpu=True, load_models=True,
                 precision='fp32', compile_model=False, device=None):

        if annotations == 'grch37' or annotations == 'GRCh37' or annotations == 'hg19':
            annotations = str(files(__name__).joinpath('annotations/grch37.txt'))
        elif annotations == 'grch38' or annotations == 'GRCh38' or annotations == 'hg38':
            annotations = str(files(__name__).joinpath('annotations/grch38.txt'))
        elif annotations == 'gencodev49' or annotations == 'gencode' or annotations == 'Gencode' or annotations == 'GENCODE':
            annotations = str(files(__name__).joinpath('annotations/gencode.v49.annotation.txt'))
        elif annotations == 'MANEv1.4' or annotations == 'MANE' or annotations == 'mane':
            annotations = str(files(__name__).joinpath('annotations/MANE.GRCh38.v1.4.ensembl_genomic.txt'))

        try:
            df = pd.read_csv(annotations, sep='\t', dtype={'CHROM': object})
            self.genes = df['#NAME'].to_numpy()
            self.chroms = df['CHROM'].to_numpy()
            self.strands = df['STRAND'].to_numpy()
            self.tx_starts = df['TX_START'].to_numpy()+1
            self.tx_ends = df['TX_END'].to_numpy()
            self.exon_starts = [np.asarray([int(i) for i in c.split(',') if i])+1
                                for c in df['EXON_START'].to_numpy()]
            self.exon_ends = [np.asarray([int(i) for i in c.split(',') if i])
                              for c in df['EXON_END'].to_numpy()]
                              
            if 'CDS_START' in df.columns and 'CDS_END' in df.columns:
                self.cds_starts = df['CDS_START'].fillna(-1).astype(int).to_numpy()
                # +1 because CDS_START is 0-based in the file, just like TX_START
                self.cds_starts = np.where(self.cds_starts != -1, self.cds_starts + 1, -1)
                self.cds_ends = df['CDS_END'].fillna(-1).astype(int).to_numpy()
            else:
                self.cds_starts = np.full(len(self.genes), -1, dtype=int)
                self.cds_ends = np.full(len(self.genes), -1, dtype=int)
            # Pre-compute exon boundary unions (avoids np.union1d per-call in get_pos_data)
            self._exon_boundary_cache = [
                np.union1d(self.exon_starts[i], self.exon_ends[i])
                for i in range(len(self.exon_starts))
            ]
            # Lazy cache: transcript-relative position of the last exon-exon
            # junction per gene index (used by the NMD predictor).
            self._last_junction_cache = {}
        except IOError as e:
            logging.error('{}'.format(e))
            exit()
        except (KeyError, pd.errors.ParserError) as e:
            logging.error('Gene annotation file {} not formatted properly: {}'.format(annotations, e))
            exit()

        try:
            self.ref_fasta = Fasta(ref_fasta, rebuild=False)
        except IOError as e:
            logging.error('{}'.format(e))
            exit()

        # Build interval trees for O(log n) gene lookups
        logging.info("Building interval trees for fast gene lookups...")
        self.gene_trees = {}
        for idx in range(len(self.genes)):
            chrom = self.chroms[idx]
            start = self.tx_starts[idx]
            end = self.tx_ends[idx]

            if chrom not in self.gene_trees:
                self.gene_trees[chrom] = IntervalTree()

            self.gene_trees[chrom][start:end+1] = idx

        logging.info(f"Built interval trees for {len(self.gene_trees)} chromosomes, "
                    f"{len(self.genes)} genes total")

        # Set device
        if device is not None:
            self.device = device
        elif cpu:
            self.device = torch.device('cpu')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.precision = precision
        self.compile_model = compile_model

        # Load models if requested
        if load_models:
            self._load_models()
        else:
            self.models = None

    def get_coding_length(self, gene_idx, pos):
        """
        Calculates the cumulative coding length from the start codon 
        up to the given genomic position `pos` (inclusive).
        Returns -1 for 5'UTR, -2 for 3'UTR, and None if CDS info is missing.
        """
        cds_s = self.cds_starts[gene_idx]
        cds_e = self.cds_ends[gene_idx]
        if cds_s == -1 or cds_e == -1:
            return None
            
        strand = self.strands[gene_idx]
        
        if strand == '+':
            if pos < cds_s: return -1 # 5' UTR
            if pos > cds_e: return -2 # 3' UTR
        else:
            if pos > cds_e: return -1 # 5' UTR (remember, for -, 5' is high coord)
            if pos < cds_s: return -2 # 3' UTR
            
        ex_starts = self.exon_starts[gene_idx]
        ex_ends = self.exon_ends[gene_idx]
        
        coding_len = 0
        if strand == '+':
            for s, e in zip(ex_starts, ex_ends):
                if e < cds_s: continue
                start_match = max(s, cds_s)
                end_match = min(e, pos)
                if start_match <= end_match:
                    coding_len += (end_match - start_match + 1)
                if e >= pos: break
        else:
            for s, e in reversed(list(zip(ex_starts, ex_ends))):
                if s > cds_e: continue
                end_match = min(e, cds_e)
                start_match = max(s, pos)
                if start_match <= end_match:
                    coding_len += (end_match - start_match + 1)
                if s <= pos: break
        return coding_len

    def get_total_coding_length(self, gene_idx):
        cds_s = self.cds_starts[gene_idx]
        cds_e = self.cds_ends[gene_idx]
        if cds_s == -1 or cds_e == -1:
            return None
            
        ex_starts = self.exon_starts[gene_idx]
        ex_ends = self.exon_ends[gene_idx]
        
        total_len = 0
        for s, e in zip(ex_starts, ex_ends):
            start_match = max(s, cds_s)
            end_match = min(e, cds_e)
            if start_match <= end_match:
                total_len += (end_match - start_match + 1)
        return total_len

    def get_cds_sequence(self, gene_idx):
        chrom = self.chroms[gene_idx]
        strand = self.strands[gene_idx]
        ex_starts = self.exon_starts[gene_idx]
        ex_ends = self.exon_ends[gene_idx]
        cds_s = self.cds_starts[gene_idx]
        cds_e = self.cds_ends[gene_idx]
        
        if cds_s == -1 or cds_e == -1:
            return ""
            
        seq_parts = []
        if strand == '+':
            for s, e in zip(ex_starts, ex_ends):
                start_match = max(s, cds_s)
                end_match = min(e, cds_e)
                if start_match <= end_match:
                    seq = self.ref_fasta[chrom][start_match - 1 : end_match]
                    if hasattr(seq, "seq"): seq = seq.seq
                    seq_parts.append(seq)
            return "".join(seq_parts)
        else:
            for s, e in reversed(list(zip(ex_starts, ex_ends))):
                start_match = max(s, cds_s)
                end_match = min(e, cds_e)
                if start_match <= end_match:
                    seq = self.ref_fasta[chrom][start_match - 1 : end_match]
                    if hasattr(seq, "seq"): seq = seq.seq
                    seq_parts.append(seq)
            full_seq = "".join(seq_parts)
            from spliceai.utils import reverse_complement
            return reverse_complement(full_seq)

    def is_in_long_exon_or_nmd_escape(self, gene_idx, pos, long_exon_threshold=400):
        """
        Check if the genomic position falls within an NMD escape region.
        This is defined as either being anywhere within a long exon, OR 
        being within the standard NMD escape window (<= 55nt upstream of the last junction).
        
        Args:
            gene_idx: Index of the gene
            pos: 1-based genomic position
            long_exon_threshold: Minimum length (in bp) for an exon to be considered 'long'
        """
        starts = self.exon_starts[gene_idx]
        ends = self.exon_ends[gene_idx]
        
        # Find the exon containing the position
        exon_idx = -1
        for i in range(len(starts)):
            if starts[i] <= pos <= ends[i]:
                exon_idx = i
                break
                
        if exon_idx != -1:
            exon_len = ends[exon_idx] - starts[exon_idx] + 1
            if exon_len >= long_exon_threshold:
                return True
            
        # Check standard NMD escape region (using standard 55nt rule from last junction)
        tx_pos = _genomic_to_transcript_pos(self, gene_idx, pos)
        if tx_pos is None:
            return False
            
        last_junc = self.get_last_junction_transcript_pos(gene_idx)
        if last_junc is None:
            return True # Single exon genes escape NMD
            
        if tx_pos < (last_junc - NMD_ESCAPE_WINDOW_NT):
            return False
            
        return True

    def _load_models(self):
        """Load SpliceAI models with optimizations for inference"""
        logging.info(f"Loading SpliceAI models on {self.device}...")
        
        # OPTIMIZATION: enable cudnn autotuning and TF32. TF32 only exists on
        # Ampere and later; on older cards the flag is simply inert.
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logging.info("cudnn.benchmark=True, TF32 enabled")
        
        # Check for PyTorch models first
        pytorch_model_paths = [
            str(files(__name__).joinpath(f'models/spliceai{x}.pt')) for x in range(1, 6)
        ]
        h5_model_paths = [
            str(files(__name__).joinpath(f'models/spliceai{x}.h5')) for x in range(1, 6)
        ]
        
        self.models = []
        
        for i in range(5):
            model = None
            
            # Try PyTorch model first
            if os.path.exists(pytorch_model_paths[i]):
                logging.info(f"Loading PyTorch model {i+1}: {pytorch_model_paths[i]}")
                model = self._load_pytorch_model(pytorch_model_paths[i])
            elif os.path.exists(h5_model_paths[i]):
                logging.info(f"Converting Keras H5 model {i+1}: {h5_model_paths[i]}")
                model = self._convert_keras_model(h5_model_paths[i])
            else:
                # Do NOT fall back to random weights. An untrained model emits a
                # full, well-formed, entirely meaningless VCF, and a
                # logging.warning is easy to lose in a batch run's log.
                raise FileNotFoundError(
                    f"No trained weights found for model {i + 1}. Expected either "
                    f"{pytorch_model_paths[i]} or {h5_model_paths[i]}. Refusing to "
                    f"run with randomly initialised weights -- scores would look "
                    f"valid and be meaningless. Convert the Keras checkpoints with "
                    f"spliceai.convert_keras_to_pytorch first."
                )
            
            # CRITICAL: Ensure eval mode BEFORE precision conversion
            model.eval()
            
            # Apply precision - USE IN-PLACE CONVERSION
            if self.precision == 'bf16':
                model = model.to(torch.bfloat16)
                logging.info(f"Model {i+1}: Converted to BF16")
            elif self.precision == 'fp16' and self.device.type == 'cuda':
                # IN-PLACE conversion to save memory
                model.half()  # This is in-place
                logging.info(f"Model {i+1}: Converted to FP16")
            
            # Compile if requested (PyTorch 2.0+)
            if self.compile_model and hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model, mode='reduce-overhead')
                    logging.info(f"Model {i+1} compiled with torch.compile()")
                except Exception as e:
                    logging.warning(f"torch.compile() failed for model {i+1}: {e}")
            
            self.models.append(model)
            logging.info(f"Model {i+1} loaded successfully")
        
        # Log model dtype
        if self.models:
            sample_param = next(self.models[0].parameters())
            logging.info(f"Model dtype: {sample_param.dtype}, device: {sample_param.device}")
        
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def _load_pytorch_model(self, path):
        """Load a native PyTorch model"""
        model = create_spliceai_model(device=self.device)
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _convert_keras_model(self, path):
        """Refuse to 'convert' a Keras model by returning random weights.

        This method never implemented conversion: it logged two warnings and
        returned create_spliceai_model(), i.e. randomly initialised weights.
        A deployment with the shipped .h5 files but no .pt files therefore
        produced a complete, plausible-looking, meaningless VCF.
        """
        raise NotImplementedError(
            f"In-process Keras conversion is not implemented (attempted for {path}). "
            f"Run spliceai.convert_keras_to_pytorch to produce spliceai{{1..5}}.pt "
            f"and re-run. Previously this returned an UNTRAINED model with only a "
            f"logging.warning."
        )

    def _apply_precision(self, model):
        """Apply precision settings to model"""
        if self.precision == 'fp16':
            model = model.half()
        elif self.precision == 'bf16':
            model = model.to(torch.bfloat16)
        return model

    def get_name_and_strand(self, chrom, pos):
        """O(log n) gene lookup using interval trees"""
        chrom = normalise_chrom(chrom, list(self.chroms)[0])

        if chrom not in self.gene_trees:
            return GeneInfo(genes=[], strands=[], idxs=[])

        overlaps = self.gene_trees[chrom][pos]

        if len(overlaps) == 0:
            return GeneInfo(genes=[], strands=[], idxs=[])

        idxs = np.array([interval.data for interval in overlaps])

        return GeneInfo(genes=self.genes[idxs], strands=self.strands[idxs], idxs=idxs)

    def get_pos_data(self, idx, pos):
        dist_tx_start = self.tx_starts[idx]-pos
        dist_tx_end = self.tx_ends[idx]-pos
        # Use pre-computed union from cache (avoids sorting/merging per call)
        dist_exon_bdry = self._exon_boundary_cache[idx] - pos
        dist_ann = (dist_tx_start, dist_tx_end, dist_exon_bdry)
        return dist_ann

    def get_last_junction_transcript_pos(self, idx):
        """
        Return the transcript-relative (0-based) position of the last
        exon-exon junction for the gene at ``idx``. Used by the NMD
        predictor: a PTC more than NMD_ESCAPE_WINDOW_NT nt upstream of this
        position is presumed to trigger NMD.

        For a single-exon transcript there is no junction; returns None.
        """
        # `None` is a legitimate cached VALUE (single-exon transcript: no
        # junction exists), so it cannot double as the cache-miss sentinel.
        # Testing `if cached is not None` treated every single-exon gene as a
        # permanent miss and recomputed it on every variant. Use a distinct
        # sentinel so a negative result is cached like any other.
        cached = self._last_junction_cache.get(idx, _CACHE_MISS)
        if cached is not _CACHE_MISS:
            return cached

        starts = self.exon_starts[idx]   # genomic, 1-based (after +1 in __init__)
        ends   = self.exon_ends[idx]     # genomic, 0-based-exclusive
        if len(starts) < 2:
            self._last_junction_cache[idx] = None
            return None

        strand = self.strands[idx]
        # Exon lengths in genomic order, then walked in transcript order.
        exon_lengths = (ends - (starts - 1)).astype(int)

        if strand == '+':
            # Transcript order == genomic order. Last junction = start of the
            # last exon. Its transcript position is the sum of all preceding
            # exon lengths.
            transcript_pos = int(exon_lengths[:-1].sum())
        else:
            # Transcript order == reverse genomic order. Last junction = end
            # of the genomically-first exon (= 3'-most exon boundary in
            # transcript orientation). Its transcript position is the sum
            # of all exon lengths after index 0 (walked in transcript order,
            # i.e. from the last genomic exon down to exon 1).
            transcript_pos = int(exon_lengths[1:].sum())

        self._last_junction_cache[idx] = transcript_pos
        return transcript_pos



# ============================================
# One-hot Encoding
# ============================================

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    logging.warning("Numba not available - one-hot encoding will be slower")


if HAS_NUMBA:
    @jit(nopython=True, cache=True)
    def _one_hot_encode_numba(seq_bytes):
        """Numba-optimized one-hot encoding"""
        n = len(seq_bytes)
        result = np.zeros((n, 4), dtype=np.float32)

        for i in range(n):
            c = seq_bytes[i]
            if c == 65 or c == 97:  # A/a
                result[i, 0] = 1.0
            elif c == 67 or c == 99:  # C/c
                result[i, 1] = 1.0
            elif c == 71 or c == 103:  # G/g
                result[i, 2] = 1.0
            elif c == 84 or c == 116:  # T/t
                result[i, 3] = 1.0

        return result

    def one_hot_encode(seq):
        """Fast one-hot encoding with Numba"""
        seq_bytes = np.frombuffer(seq.encode('latin-1'), dtype=np.uint8)
        return _one_hot_encode_numba(seq_bytes)
else:
    # 256-row lookup indexed by raw byte value. Only A/a C/c G/g T/t get a 1;
    # every other byte -- N, IUPAC ambiguity codes (RYSWKMBDHV), and anything
    # else -- stays all-zero. This is the same convention as the numba encoder
    # above.
    #
    # The previous implementation substituted \x01..\x04 for ACGT and then
    # indexed map[byte % 5] on the RAW ASCII of whatever was left, so IUPAC
    # codes were silently encoded as concrete wrong bases (R -> C, Y -> T,
    # S -> G, ...) instead of all-zero. GRCh38 primary assembly does contain
    # IUPAC codes, and numba is not installed by default, so that fallback was
    # the production path and gave different scores from a machine with numba.
    _ONE_HOT_LUT = np.zeros((256, 4), dtype=np.float32)
    for _b, _col in ((b'A', 0), (b'C', 1), (b'G', 2), (b'T', 3)):
        _ONE_HOT_LUT[_b[0], _col] = 1.0          # upper case
        _ONE_HOT_LUT[_b.lower()[0], _col] = 1.0  # lower case

    def one_hot_encode(seq):
        """One-hot encode a sequence; non-ACGT is all-zero (matches numba path)."""
        return _ONE_HOT_LUT[np.frombuffer(seq.encode('latin-1'), np.uint8)]


def normalise_chrom(source, target):
    """Match `source`'s 'chr' prefix convention to `target`'s.

    Uses slicing rather than str.strip('chr'), which removes any leading OR
    trailing run of the CHARACTERS c/h/r rather than the prefix: it maps
    'chrScaffold_hcr' -> 'Scaffold_' and 'chrArch' -> 'A'. No GRCh37/38 contig
    name is affected (verified across primary, alt, decoy and EBV contigs), so
    this was latent rather than active, but it would silently truncate any
    contig ending in c, h or r on a non-human assembly.
    """
    def has_prefix(x):
        return x.startswith('chr')

    if has_prefix(source) and not has_prefix(target):
        return source[3:]
    elif not has_prefix(source) and has_prefix(target):
        return 'chr' + source

    return source


def get_cov(dist_var):
    return 2 * dist_var + 1


def get_wid(cov):
    return 10000 + cov


def _position_corrected_diff(y_alt_2d, y_ref_2d, cov_half, ins_len, del_len):
    """
    Compute position-corrected delta (alt - ref) for indels, indexed in
    REFERENCE coordinates.

    When an indel is encoded, downstream positions in x_alt are shifted
    relative to x_ref, so a splice site that merely *moved* produces gain=1 and
    loss=1 simultaneously under plain subtraction. This function compares each
    reference position against the alt position holding the same base, so a
    preserved-but-shifted site gives delta ≈ 0.

    COORDINATE FRAME. The returned array is indexed by REFERENCE position:
    output index i corresponds to reference offset (i - cov_half) from
    record.pos, on the same axis as y_ref_2d. Callers therefore report
    (argmax - cov_half) as a genomic offset and match it against annotated
    exon boundaries (which are reference offsets) without further adjustment.
    Returning alt-frame indices instead would misreport every downstream site
    of an indel by `shift` bp and break the canonical-site mask.

    Regions of the reference axis, for a left-anchored VCF indel where
    encode_seqs builds x_alt = x_ref[:wid//2] + alt + x_ref[wid//2 + ref_len:]:

      shared prefix   [0, cov_half]                  x_alt[i] == x_ref[i];
                                                     compare directly.
      deleted span    [cov_half+1, cov_half+del_len] these reference bases are
                                                     removed by the variant, so
                                                     alt := 0 and a site here
                                                     reports its full loss.
      shifted tail    [cov_half+del_len+1, L)        reference base i sits at
                                                     alt index i - shift;
                                                     compare against that.
      unmappable tail  where i - shift >= L          the alt window ends before
                                                     this reference position;
                                                     alt := ref (delta 0) rather
                                                     than 0, which would
                                                     fabricate a maximal LOSS at
                                                     a position whose only
                                                     defect is that the window
                                                     ran out.

    Sequence novel to the alt allele (the inserted bases themselves) has no
    reference position and so cannot appear on this axis. A site the model
    calls inside an insertion is attributed to the variant position: the
    maximum alt score over the inserted bases is compared against the
    reference score at cov_half, which is where such a gain is reported.

    Args:
        y_alt_2d : ndarray (L, 2) — model output for alt, columns = [acceptor, donor]
        y_ref_2d : ndarray (L, 2) — model output for ref
        cov_half : int            — output index of the variant position
        ins_len  : int            — number of inserted bases (alt_len - ref_len, ≥ 0)
        del_len  : int            — number of deleted bases (ref_len - alt_len, ≥ 0)

    Returns:
        ndarray (L, 2) of position-corrected deltas, in reference coordinates.
    """
    if del_len == 0 and ins_len == 0:
        return y_alt_2d - y_ref_2d

    L = y_alt_2d.shape[0]
    # shift > 0  → deletion  (downstream ref indices are larger)
    # shift < 0  → insertion (downstream ref indices are smaller)
    shift = del_len - ins_len

    # Build an alt-score array aligned to the REFERENCE axis. See the docstring
    # for the derivation of each region.
    #
    # The previous implementation remapped the reference onto the ALT axis and
    # started the remap at dst = cov_half, which (a) compared p_alt(anchor)
    # against p_ref(anchor + shift), injecting a spurious delta of up to 1.0 at
    # the variant position itself whenever the variant sat `shift` bp from a
    # strong site — exactly the case this function exists to handle; (b) left
    # the result in alt coordinates while callers reported the index as a
    # reference offset; and (c) zero-filled the unmappable tail, fabricating a
    # maximal gain there.
    y_alt_corr = y_alt_2d.copy()                     # shared prefix: as-is

    first_shifted = cov_half + del_len + 1           # start of the shifted tail

    if del_len:
        # Deleted reference bases: absent from alt, so no site can be called
        # there. Yields delta = -p_ref, the full loss.
        y_alt_corr[cov_half + 1:min(first_shifted, L), :] = 0.0

    if first_shifted < L:
        ref_idx = np.arange(first_shifted, L)
        alt_idx = ref_idx - shift                    # shift = del_len - ins_len
        ok = (alt_idx >= 0) & (alt_idx < L)
        y_alt_corr[ref_idx[ok], :] = y_alt_2d[alt_idx[ok], :]
        # Unmappable tail → delta 0 (see docstring).
        y_alt_corr[ref_idx[~ok], :] = y_ref_2d[ref_idx[~ok], :]

    if ins_len:
        # Inserted bases have no reference position. Attribute the strongest
        # score among them to the variant position, where an insertion-internal
        # gain is reported.
        ins_block = y_alt_2d[cov_half + 1:min(cov_half + 1 + ins_len, L), :]
        if ins_block.size:
            y_alt_corr[cov_half, :] = np.maximum(y_alt_corr[cov_half, :],
                                                 ins_block.max(axis=0))

    return y_alt_corr - y_ref_2d


def encode_seqs(record, seq, ann, gene_info, gene_ix, alt_ix, wid):
    """
    Encode ref and alt sequences for SpliceAI prediction.
    
    For indels, both ref and alt sequences are encoded to the SAME length (wid)
    to ensure model outputs have matching dimensions for delta score calculation.
    
    For insertions: The extra bases are placed centered at the variant position,
                   then the sequence is truncated symmetrically from both ends.
    For deletions:  N's are added to pad the sequence back to wid length,
                   distributed around the deletion site.
    """
    dist_ann = ann.get_pos_data(gene_info.idxs[gene_ix], record.pos)
    pad_size = [max(wid // 2 + dist_ann[0], 0), max(wid // 2 - dist_ann[1], 0)]
    ref_len = len(record.ref)
    alt_len = len(record.alts[alt_ix])
    
    # Construct x_ref (always wid length)
    x_ref = 'N' * pad_size[0] + seq[pad_size[0]:wid - pad_size[1]] + 'N' * pad_size[1]
    
    # Construct x_alt - initially may be different length due to indel
    x_alt = x_ref[:wid // 2] + str(record.alts[alt_ix]) + x_ref[wid // 2 + ref_len:]
    
    # --- INDEL LENGTH NORMALIZATION ---
    # Force x_alt to match wid so that model outputs have matching dimensions.
    # IMPORTANT: use end-only truncation/padding to keep the upstream context
    # identical between x_ref and x_alt.  Symmetric trimming would shift upstream
    # splice sites, creating additional spurious gain/loss signals.
    len_diff = len(x_alt) - wid

    if len_diff > 0:
        # Insertion: truncate from the 3' end only.
        # Upstream context is preserved identically; only the downstream tail is lost.
        x_alt = x_alt[:wid]
    elif len_diff < 0:
        # Deletion: pad with N's at the 3' end only.
        # Downstream context is correctly shifted; the tail is filled with N padding.
        x_alt = x_alt + 'N' * (-len_diff)
    # ---------------------------------

    x_ref = one_hot_encode(x_ref)[None, :]
    x_alt = one_hot_encode(x_alt)[None, :]

    if gene_info.strands[gene_ix] == '-':
        x_ref = x_ref[:, ::-1, ::-1]
        x_alt = x_alt[:, ::-1, ::-1]

    return x_ref, x_alt

def is_record_valid(record):
    try:
        record.chrom, record.pos, record.ref, len(record.alts)
    except TypeError:
        logging.warning('Skipping record (bad input): {}'.format(record))
        return False
    return True


def get_seq(record, ann, wid):
    chrom = normalise_chrom(record.chrom, list(ann.ref_fasta.keys())[0])
    try:
        seq = ann.ref_fasta[chrom][
            record.pos - wid // 2 - 1: record.pos + wid // 2
        ].seq
    except (IndexError, ValueError):
        logging.warning('Skipping record (fasta issue): {}'.format(record))
        return ""
    return seq


def is_valid_alt_record(record, alt_ix):
    if '.' in record.alts[alt_ix] or '-' in record.alts[alt_ix] or '*' in record.alts[alt_ix]:
        return False
    if '<' in record.alts[alt_ix] or '>' in record.alts[alt_ix]:
        return False
    return True


def is_location_predictable(record, seq, wid, dist_var):
    # Window-length check FIRST. pyfaidx clamps a slice at the contig
    # boundaries, so a variant within wid//2 (5500 bp at the default -D 500) of
    # a contig start or end comes back with a SHORT sequence. seq[wid//2] is
    # then not the variant base at all, and the reference comparison below
    # fails for a reason that has nothing to do with the reference. Ordered the
    # other way round, every such variant was reported as
    #     'Skipping record (ref issue) should be N - chr1:100 G'
    # which reads as a reference-build or contig-naming mismatch and sends the
    # user to check their FASTA. Both checks still skip the record; only the
    # diagnostic changes, and it is the diagnostic that costs debugging time.
    if len(seq) != wid:
        logging.warning(
            'Skipping record (too near a chromosome end for a {} bp window): '
            '{}'.format(wid, record)
        )
        return False

    if len(record.ref) > 2 * dist_var:
        logging.warning('Skipping record (ref too long): {}'.format(record))
        return False

    var_ref_seq = seq[wid // 2: wid // 2 + len(record.ref)].upper()
    if var_ref_seq != record.ref:
        logging.warning(
            'Skipping record (REF does not match the reference FASTA; '
            'expected {}) - {}'.format(var_ref_seq, record)
        )
        return False

    return True


def _unhandled(alt, gene, orig):
    """'.'-filled placeholder record of exactly the schema's field count."""
    n_trailing = n_info_fields(orig=orig) - 2  # ALLELE and SYMBOL are populated
    return '|'.join([str(alt), str(gene)] + ['.'] * n_trailing)


def create_unhandled_delta_score_orig(alt, gene):
    # 14 fields: ALLELE, SYMBOL, DSM(4), DS(4), DP(4). Derived from
    # INFO_FIELDS_ORIG so it cannot drift from the header.
    return _unhandled(alt, gene, orig=True)


def create_unhandled_delta_score(alt, gene):
    # 23 fields: ALLELE, SYMBOL, DSM(4), DS(4), DP(4), RS(4), mane_donor,
    # mane_acceptor, sites_gt05_donor, sites_gt05_acceptor, event_class.
    # Derived from INFO_FIELDS_EXTENDED so it cannot drift from the header.
    return _unhandled(alt, gene, orig=False)


def get_delta_scores(record, ann, dist_var, orig=False):
    """Get delta scores using PyTorch models"""
    cov = get_cov(dist_var)
    wid = get_wid(cov)
    delta_scores = []

    if not is_record_valid(record):
        return delta_scores

    gene_info = ann.get_name_and_strand(record.chrom, record.pos)
    if len(gene_info.idxs) == 0:
        return delta_scores

    seq = get_seq(record, ann, wid)
    if not seq:
        return delta_scores

    if not is_location_predictable(record, seq, wid, dist_var):
        return delta_scores

    for alt_ix in range(len(record.alts)):
        for gene_ix in range(len(gene_info.idxs)):

            if not is_valid_alt_record(record, alt_ix):
                continue

            if len(record.ref) > 1 and len(record.alts[alt_ix]) > 1:
                if orig:
                    delta_score = create_unhandled_delta_score_orig(record.alts[alt_ix], gene_info.genes[gene_ix])
                else:
                    delta_score = create_unhandled_delta_score(record.alts[alt_ix], gene_info.genes[gene_ix])
                delta_scores.append(delta_score)
                continue

            x_ref, x_alt = encode_seqs(record=record,
                                       seq=seq,
                                       ann=ann,
                                       gene_info=gene_info,
                                       gene_ix=gene_ix,
                                       alt_ix=alt_ix,
                                       wid=wid)

            # Convert to PyTorch tensors: (N, L, 4) -> (N, 4, L)
            x_ref_tensor = torch.from_numpy(np.transpose(x_ref, (0, 2, 1)).copy()).float().to(ann.device)
            x_alt_tensor = torch.from_numpy(np.transpose(x_alt, (0, 2, 1)).copy()).float().to(ann.device)

            with torch.no_grad():
                y_refs = []
                y_alts = []
                for m in range(5):
                    y_ref = ann.models[m](x_ref_tensor)
                    y_alt = ann.models[m](x_alt_tensor)
                    # Output: (N, 3, L) -> (N, L, 3)
                    y_refs.append(y_ref.cpu().numpy().transpose(0, 2, 1))
                    y_alts.append(y_alt.cpu().numpy().transpose(0, 2, 1))
                
                y_ref = np.mean(y_refs, axis=0)
                y_alt = np.mean(y_alts, axis=0)

            delta_score = get_alt_gene_delta_score(record=record,
                                                   ann=ann,
                                                   alt_ix=alt_ix,
                                                   gene_ix=gene_ix,
                                                   y_ref=y_ref,
                                                   y_alt=y_alt,
                                                   cov=cov,
                                                   gene_info=gene_info,
                                                   seq=seq,
                                                   wid=wid,
                                                   # Was hardcoded orig=False, which
                                                   # discarded the caller's flag, so
                                                   # --orig_output selected the 10-field
                                                   # v1.3.1 header but emitted extended
                                                   # 23-field records.
                                                   orig=orig)
            delta_scores.append(delta_score)
    
    gc.collect()
    return delta_scores


def compute_orig_delta_scores_batched(y_refs, y_alts, strands, gene_idxs, cov, exon_boundary_cache, positions, alts, gene_names, refs=None):
    """
    Vectorized batch computation for orig output format.

    Args:
        y_refs: list of (1, L, 3) arrays
        y_alts: list of (1, L, 3) arrays
        strands: list of strand chars ('+' or '-')
        gene_idxs: list of gene indices for exon boundary lookup
        cov: coverage value
        exon_boundary_cache: pre-computed exon boundaries from Annotator
        positions: list of record positions
        alts: list of alt alleles (strings)
        gene_names: list of gene names

    Returns:
        list of delta score strings
    """
    n = len(y_refs)
    if n == 0:
        return []

    cov_half = cov // 2
    results = []

    # Process in chunks to balance vectorization vs memory
    CHUNK_SIZE = 256

    for chunk_start in range(0, n, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, n)
        chunk_size = chunk_end - chunk_start

        # Collect chunk data
        chunk_y_refs = []
        chunk_y_alts = []
        chunk_strands = strands[chunk_start:chunk_end]
        chunk_gene_idxs = gene_idxs[chunk_start:chunk_end]
        chunk_positions = positions[chunk_start:chunk_end]
        chunk_alts = alts[chunk_start:chunk_end]
        chunk_gene_names = gene_names[chunk_start:chunk_end]
        chunk_refs = refs[chunk_start:chunk_end] if refs is not None else None

        # Apply strand reversal and collect arrays
        for i in range(chunk_size):
            y_ref = y_refs[chunk_start + i]
            y_alt = y_alts[chunk_start + i]

            if chunk_strands[i] == '-':
                y_ref = y_ref[:, ::-1]
                y_alt = y_alt[:, ::-1]

            # Ensure shapes match
            if y_ref.shape[1] != y_alt.shape[1]:
                min_len = min(y_ref.shape[1], y_alt.shape[1])
                y_ref = y_ref[:, :min_len, :]
                y_alt = y_alt[:, :min_len, :]

            chunk_y_refs.append(y_ref[0])  # Remove batch dim: (L, 3)
            chunk_y_alts.append(y_alt[0])

        # Stack into batch arrays - handle variable lengths by processing individually
        # but still avoid Python loops for the math
        for i in range(chunk_size):
            y_ref = chunk_y_refs[i]
            y_alt = chunk_y_alts[i]

            # Position-corrected diff: prevents spurious all-1 scores for indels
            # near strong splice sites by accounting for the downstream sequence shift.
            if chunk_refs is not None:
                ref_seq = chunk_refs[i]
                alt_seq = chunk_alts[i]
                ins_len_i = max(len(alt_seq) - len(ref_seq), 0)
                del_len_i = max(len(ref_seq) - len(alt_seq), 0)
            else:
                ins_len_i = del_len_i = 0

            diff_subset = _position_corrected_diff(
                y_alt[:, 1:], y_ref[:, 1:], cov_half, ins_len_i, del_len_i
            )  # (L, 2)

            # Get indices
            idx_pa = np.argmax(diff_subset[:, 0])
            idx_pd = np.argmax(diff_subset[:, 1])
            idx_na = np.argmin(diff_subset[:, 0])
            idx_nd = np.argmin(diff_subset[:, 1])

            # Get exon boundary set
            gene_idx = chunk_gene_idxs[i]
            pos = chunk_positions[i]
            dist_exon_bdry = exon_boundary_cache[gene_idx] - pos
            dist_ann_set = set(dist_exon_bdry)

            # Compute masks
            mask_pa = 1 if (idx_pa - cov_half) in dist_ann_set else 0
            mask_na = 1 if (idx_na - cov_half) not in dist_ann_set else 0
            mask_pd = 1 if (idx_pd - cov_half) in dist_ann_set else 0
            mask_nd = 1 if (idx_nd - cov_half) not in dist_ann_set else 0

            # Use corrected delta values for the score output
            score_pa =  diff_subset[idx_pa, 0]
            score_na = -diff_subset[idx_na, 0]
            score_pd =  diff_subset[idx_pd, 1]
            score_nd = -diff_subset[idx_nd, 1]

            # Format result
            result = "{}|{}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{}|{}|{}|{}".format(
                chunk_alts[i],
                chunk_gene_names[i],
                score_pa * (1 - mask_pa),
                score_na * (1 - mask_na),
                score_pd * (1 - mask_pd),
                score_nd * (1 - mask_nd),
                score_pa,
                score_na,
                score_pd,
                score_nd,
                idx_pa - cov_half,
                idx_na - cov_half,
                idx_pd - cov_half,
                idx_nd - cov_half
            )
            results.append(result)

    return results


# ============================================
# Splice-event classification (new, rewritten)
# ============================================


def predict_nmd(event_transcript_pos, last_junction_transcript_pos, new_junction_transcript_pos=None):
    """
    Conservative NMD predictor for a frameshift/truncating splice event.
    Returns (is_nmd, is_nmd_by_new_exon).
    """
    if event_transcript_pos is None:
        return False, False
        
    old_nmd = False
    if last_junction_transcript_pos is not None:
        old_nmd = event_transcript_pos < (last_junction_transcript_pos - NMD_ESCAPE_WINDOW_NT)
        
    new_nmd = old_nmd
    if new_junction_transcript_pos is not None:
        new_last = new_junction_transcript_pos
        if last_junction_transcript_pos is not None:
            new_last = max(last_junction_transcript_pos, new_junction_transcript_pos)
        new_nmd = event_transcript_pos < (new_last - NMD_ESCAPE_WINDOW_NT)
        
    return new_nmd, (new_nmd and not old_nmd)


def _genomic_to_transcript_pos(ann, idx, genomic_pos):
    """
    Map a genomic position to a 0-based transcript-relative coordinate in
    the gene ``idx``. For intronic positions, returns the transcript
    coordinate of the most-recent-in-transcript exon boundary (i.e. the
    upstream side of the intron). Returns None if the gene has no exons.
    """
    starts = ann.exon_starts[idx]
    ends   = ann.exon_ends[idx]
    if len(starts) == 0:
        return None
    strand = ann.strands[idx]
    exon_lengths = (ends - (starts - 1)).astype(int)

    if strand == '+':
        cumlen = 0
        for i in range(len(starts)):
            s = int(starts[i])
            e = int(ends[i])
            if genomic_pos < s:
                return cumlen
            if genomic_pos <= e:
                return cumlen + (genomic_pos - s)
            cumlen += int(exon_lengths[i])
        return cumlen
    else:
        cumlen = 0
        for i in range(len(starts) - 1, -1, -1):
            s = int(starts[i])
            e = int(ends[i])
            if genomic_pos > e:
                return cumlen
            if genomic_pos >= s:
                return cumlen + (e - genomic_pos)
            cumlen += int(exon_lengths[i])
        return cumlen


def _tagged_mane_sites(dist_ann_all, y_ref, cov_half, strand):
    """
    Return the list of MANE splice sites near the variant, each tagged
    with its type ('D' or 'A'), sorted in transcript order.

    Each element is (rel_pos, type, ref_score).
    Sites with both channels below MIN_CANONICAL_SS_SCORE are dropped —
    these are annotation positions that the model does not confirm as
    real splice sites (e.g. TSS / polyA boundaries).
    """
    L = y_ref.shape[1]
    tagged = []
    for raw in dist_ann_all[2]:
        r = int(raw)
        idx = cov_half + r
        if idx < 0 or idx >= L:
            continue
        acc = float(y_ref[0, idx, 1])
        don = float(y_ref[0, idx, 2])
        if don >= acc and don >= MIN_CANONICAL_SS_SCORE:
            tagged.append((r, 'D', don))
        elif acc > don and acc >= MIN_CANONICAL_SS_SCORE:
            tagged.append((r, 'A', acc))
    if strand == '-':
        tagged.sort(key=lambda t: -t[0])
    else:
        tagged.sort(key=lambda t: t[0])
    return tagged


def _has_competing_alt_sites(y_alt, ch_ix, cov_half, dist_ann_all, center_rel):
    """
    Detect >1 strong sites of the same type within the canonical
    neighborhood (bounded by the two nearest flanking MANE sites) of
    ``center_rel``. Used to emit Ambiguous when the model can't choose.
    """
    L = y_alt.shape[1]
    rels = [int(r) for r in dist_ann_all[2]]
    left_rel  = max([r for r in rels if r < center_rel], default=-cov_half)
    right_rel = min([r for r in rels if r > center_rel], default=cov_half)
    left  = max(0, min(cov_half + left_rel + 1, L - 1))
    right = max(left + 1, min(cov_half + right_rel,   L))
    window = y_alt[0, left:right, ch_ix]
    strong = window[window >= COMPETING_MIN_ABS]
    if strong.size <= 1:
        return False
    top = float(strong.max())
    if top <= 0:
        return False
    second = float(np.partition(strong, -2)[-2])
    return (top - second) / top < COMPETING_REL_TOL


def _lookup_loss_neighbors(loss_rel, site_channel, tagged, y_alt, cov_half):
    """
    Given a lost canonical site at ``loss_rel`` (of type 'D' or 'A'),
    identify the Intron-Retention pair (next opposite-type site in
    transcript direction) and the Exon-Skip pair (the opposite-type site
    *after* the next same-type site, i.e. the acceptor of the exon after
    the one being skipped).

    Returns (ir_info, es_info), each either None or a dict:
        {'site_rel': int, 'size': int, 'alt_score': float}

    ``size`` is the length in bp of the retained intron (IR) or of the
    skipped exon (ES).
    """
    want_partner = 'A' if site_channel == 'D' else 'D'
    partner_ch = 1 if want_partner == 'A' else 2

    # Locate the canonical site in the transcript-ordered list
    canonical_ix = -1
    for i, (r, s, _score) in enumerate(tagged):
        if r == loss_rel and s == site_channel:
            canonical_ix = i
            break
    if canonical_ix == -1:
        return None, None

    # IR pair: next partner-type site after the canonical one
    ir_pair = None
    next_same = None
    es_pair = None
    for i in range(canonical_ix + 1, len(tagged)):
        r, s, _ = tagged[i]
        if s == want_partner and ir_pair is None:
            ir_pair = (r, i)
        elif s == site_channel and ir_pair is not None and next_same is None:
            next_same = (r, i)
        elif s == want_partner and next_same is not None and es_pair is None:
            es_pair = (r, i)
            break

    L = y_alt.shape[1]

    def alt_score_at(rel):
        idx = cov_half + rel
        if idx < 0 or idx >= L:
            return 0.0
        return float(y_alt[0, idx, partner_ch])

    ir_info = None
    es_info = None
    if ir_pair is not None:
        # Retained intron = the bases STRICTLY BETWEEN the two exons.
        # The lost site and its partner are both exonic bases (the annotation
        # marks the last exonic base as the donor and the first exonic base as
        # the acceptor — verified against canonical GT/AG dinucleotides), so the
        # intron spans |partner - loss| - 1 bases. Using abs() alone over-counted
        # by 1 and put every retained-intron frame call in the wrong residue
        # class.
        ir_size = abs(ir_pair[0] - loss_rel) - 1
        ir_info = {
            'site_rel': ir_pair[0],
            'size': ir_size,
            'alt_score': alt_score_at(ir_pair[0]),
        }

    if ir_pair is not None and next_same is not None:
        # Skipped exon length = distance (in bp) from the immediate-partner
        # acceptor to the next-same-type donor (i.e. the exon between IR
        # pair and next_same). This is the correct "exon size" to use for
        # the ES frame shift — NOT the donor→donor distance (which would
        # include the intron).
        #
        # Both endpoints are exonic bases (acceptor = first exonic base, donor =
        # last exonic base — verified against canonical GT/AG dinucleotides), so
        # the exon is |donor - acceptor| + 1 bases INCLUSIVE. Using abs() alone
        # under-counted by 1 and put every exon-skip frame call in the wrong
        # residue class — in the opposite direction to the IR error above, so
        # the two did not cancel.
        skipped_size = abs(next_same[0] - ir_pair[0]) + 1
        es_site_rel = es_pair[0] if es_pair is not None else next_same[0]
        es_info = {
            'site_rel': es_site_rel,
            'size': skipped_size,
            'alt_score': alt_score_at(es_site_rel),
        }

    return ir_info, es_info


def _pick_ir_vs_es(ir_info, es_info):
    """
    Use alt-model partner scores to choose between Intron Retention and
    Exon Skipping. Returns 'IR', 'ES', or 'ambiguous'.
    """
    if ir_info is None and es_info is None:
        return 'ambiguous'
    ir_s = ir_info['alt_score'] if ir_info else 0.0
    es_s = es_info['alt_score'] if es_info else 0.0
    if ir_s >= MIN_NEW_SS_SCORE and es_s < 0.3:
        return 'IR'
    if es_s >= MIN_NEW_SS_SCORE and ir_s < 0.3:
        return 'ES'
    return 'ambiguous'


def _finalize_frame_call(shift_bp, event_genomic_pos, ann, gene_idx, event_type='insertion', pex_seq=None, new_junction_tx=None, divergence_pos=None):
    """
    Convert a shift_bp (total change in transcript length) + event location 
    into an event_class label. For frameshifts, apply NMD.
    """
    div_pos = divergence_pos if divergence_pos is not None else event_genomic_pos
    coding_len = ann.get_coding_length(gene_idx, div_pos)
    
    if coding_len == -1:
        return '5UTR_Event'
    elif coding_len == -2:
        return '3UTR_Event'
    elif coding_len is None:
        return 'NonCoding'
        
    frame_shift = shift_bp % 3
    
    if frame_shift == 0:
        proportion_str = ""
        total_len = ann.get_total_coding_length(gene_idx)
        if total_len and total_len > 0:
            proportion = shift_bp / total_len
            proportion_str = f'_{proportion:.2f}'

        # Check for PTC in insertions
        if pex_seq is not None:
            frame_offset = (3 - (coding_len % 3)) % 3
            if find_stop_in_frame(pex_seq, frame_offset) is not None:
                return f'InFrameInsertion_PTC({shift_bp // 3}aa{proportion_str})'
            
        if event_type in ['insertion', 'pseudoexon', 'intron_retention']:
            return f'InFrameInsertion({shift_bp // 3}aa{proportion_str})'
        else:
            return f'InFrameDeletion({shift_bp // 3}aa{proportion_str})'

    event_tx = _genomic_to_transcript_pos(ann, gene_idx, div_pos)
    last_junction = ann.get_last_junction_transcript_pos(gene_idx)
    
    prefix = 'Frameshift'
    total_len = ann.get_total_coding_length(gene_idx)
    
    aa_count = coding_len // 3
    dist_to_stop = "?"
    cds_seq = ann.get_cds_sequence(gene_idx)
    if cds_seq:
        if event_type in ['insertion', 'pseudoexon', 'intron_retention']:
            new_cds = cds_seq[:coding_len] + (pex_seq or "") + cds_seq[coding_len:]
        else:
            new_cds = cds_seq[:coding_len] + cds_seq[coding_len + shift_bp:]
            
        leftover = coding_len % 3
        seq_to_scan = new_cds[coding_len - leftover :]
        stop_idx = find_stop_in_frame(seq_to_scan, 0)
        if stop_idx is not None:
            dist_to_stop = stop_idx // 3

    if total_len and total_len > 0:
        proportion = coding_len / total_len
        prefix = f'Frameshift({proportion:.2f}_{aa_count}aa_{dist_to_stop}aa)'
            
    if event_tx is None:
        return f'{prefix}_Ambiguous'
        
    is_nmd, is_nmd_by_new_exon = predict_nmd(event_tx, last_junction, new_junction_tx)
    if is_nmd:
        if is_nmd_by_new_exon:
            return f'{prefix}_NMD_by_NewExon'
        return f'{prefix}_NMD'
    return f'{prefix}_NMDescape'


def _classify_splice_event_inner(y_ref, y_alt, diff_subset,
                           idx_pa, idx_pd, idx_na, idx_nd,
                           ann, gene_ix, gene_info, record,
                           dist_ann_all, cov_half, get_subsequence):
    """
    Inner logic for classify_splice_event.
    Classify the variant's effect on the nearest splicing event.
    """
    strand    = gene_info.strands[gene_ix]
    gene_idx  = gene_info.idxs[gene_ix]
    L = y_ref.shape[1]

    dpa_delta =  float(diff_subset[idx_pa, 0])   # acceptor gain (positive)
    dpd_delta =  float(diff_subset[idx_pd, 1])   # donor gain
    dna_delta = -float(diff_subset[idx_na, 0])   # acceptor loss (stored as +ve)
    dnd_delta = -float(diff_subset[idx_nd, 1])   # donor loss

    pa_rel = int(idx_pa - cov_half)
    pd_rel = int(idx_pd - cov_half)
    na_rel = int(idx_na - cov_half)
    nd_rel = int(idx_nd - cov_half)

    dist_ann_set = set(int(x) for x in dist_ann_all[2])

    def y_at(y, idx, ch):
        if idx < 0 or idx >= L:
            return 0.0
        return float(y[0, idx, ch])

    alt_pa = y_at(y_alt, idx_pa, 1)
    alt_pd = y_at(y_alt, idx_pd, 2)
    ref_na = y_at(y_ref, idx_na, 1)
    ref_nd = y_at(y_ref, idx_nd, 2)

    tagged = _tagged_mane_sites(dist_ann_all, y_ref, cov_half, strand)
    
    def get_nearest_canonical_alt_score(rel_pos, channel):
        min_dist = float('inf')
        nearest_score = None
        for r, ch, _ in tagged:
            if ch == channel:
                dist = abs(r - rel_pos)
                if dist < min_dist:
                    min_dist = dist
                    idx = cov_half + r
                    nearest_score = float(y_alt[0, idx, 1 if ch == 'A' else 2])
        return nearest_score

    donor_gain    = (dpd_delta >= MIN_NEW_SS_SCORE
                     and alt_pd  >= MIN_NEW_SS_SCORE
                     and pd_rel not in dist_ann_set)
    donor_loss    = (dnd_delta >= MIN_CANONICAL_SS_SCORE
                     and ref_nd  >= MIN_CANONICAL_SS_SCORE
                     and nd_rel in dist_ann_set)
    acceptor_gain = (dpa_delta >= MIN_NEW_SS_SCORE
                     and alt_pa  >= MIN_NEW_SS_SCORE
                     and pa_rel not in dist_ann_set)
    acceptor_loss = (dna_delta >= MIN_CANONICAL_SS_SCORE
                     and ref_na  >= MIN_CANONICAL_SS_SCORE
                     and na_rel in dist_ann_set)

    new_junction_tx = None
    if donor_gain:
        new_junction_tx = _genomic_to_transcript_pos(ann, gene_idx, record.pos + pd_rel)

    donor_event    = donor_gain or donor_loss
    acceptor_event = acceptor_gain or acceptor_loss

    if not donor_event and not acceptor_event:
        return 'NoChange', 'no_change'

    # --- Pseudoexon: donor-gain AND acceptor-gain at non-canonical positions,
    # with no canonical loss (otherwise it's better explained as a shift).
    if (donor_gain and acceptor_gain and not donor_loss and not acceptor_loss):
        # A pseudoexon is an inserted exon: its acceptor is at the 5' end and
        # its donor at the 3' end, in TRANSCRIPT orientation. In genomic
        # coordinates that means acceptor < donor on the + strand and
        # donor < acceptor on the - strand. pa_rel/pd_rel are genomic offsets
        # from record.pos (y is flipped back to genomic orientation before the
        # argmax), so:
        #
        # The strand arms were previously inverted — start_g was taken from the
        # donor on + and from the acceptor on - — which put start_g downstream
        # of end_g for every real pseudoexon on either strand. The `end_g <=
        # start_g` guard below then fired unconditionally, making this entire
        # branch unreachable: 'Pseudoexon' and 'Pseudoexon_PoisonExon' could
        # never be emitted, and every genuine pseudoexon was labelled
        # 'Ambiguous'.
        if strand == '+':
            start_g = record.pos + pa_rel      # acceptor: 5' (genomic left)
            end_g   = record.pos + pd_rel      # donor:    3' (genomic right)
        else:
            start_g = record.pos + pd_rel      # donor:    3' (genomic left)
            end_g   = record.pos + pa_rel      # acceptor: 5' (genomic right)
        if end_g <= start_g:
            return 'Ambiguous', 'ambiguous'
        pex_seq = get_subsequence(start_g, end_g)
        if pex_seq and strand == '-':
            pex_seq = reverse_complement(pex_seq)
        if not pex_seq:
            return 'Ambiguous', 'ambiguous'
        # No has_stop_codon() check: it scanned frame 0 of a genomic fragment
        # whose transcript phase is unknown, so 'Pseudoexon_PoisonExon' was
        # correct only when the phase happened to be 0. See _finalize_frame_call.
        #
        # The frameshift call below IS frame-independent -- an inserted exon
        # whose length is not a multiple of 3 shifts the downstream frame
        # regardless of phase -- so those labels are retained.
        pex_size = len(pex_seq)
        event_genomic = start_g if strand == '+' else end_g
        
        is_minor = False
        nearest_can_a = get_nearest_canonical_alt_score(pa_rel, 'A')
        nearest_can_d = get_nearest_canonical_alt_score(pd_rel, 'D')
        if (nearest_can_a is not None and nearest_can_a > alt_pa) or \
           (nearest_can_d is not None and nearest_can_d > alt_pd):
            is_minor = True
            
        ret = _finalize_frame_call(pex_size, event_genomic, ann, gene_idx, event_type='pseudoexon', pex_seq=pex_seq, new_junction_tx=new_junction_tx)
        if is_minor:
            ret += '_Minor'
        return ret, 'pseudoexon'

    # --- Double Canonical Loss (Exon Skip / Intron Retention) ---
    if donor_loss and acceptor_loss and not donor_gain and not acceptor_gain:
        is_exon = (na_rel < nd_rel) if strand == '+' else (nd_rel < na_rel)
        if is_exon:
            shift_bp = abs(nd_rel - na_rel) + 1
            event_genomic = record.pos + nd_rel
            divergence_pos = record.pos + na_rel
            ret = _finalize_frame_call(shift_bp, event_genomic, ann, gene_idx, event_type='deletion', new_junction_tx=new_junction_tx, divergence_pos=divergence_pos)
            return ret, 'exon_skip'
        else:
            shift_bp = max(0, abs(nd_rel - na_rel) - 1)
            event_genomic = record.pos + nd_rel
            start_g = record.pos + min(na_rel, nd_rel)
            end_g = record.pos + max(na_rel, nd_rel)
            ir_seq = get_subsequence(start_g, end_g)
            if ir_seq and strand == '-':
                ir_seq = reverse_complement(ir_seq)
            ret = _finalize_frame_call(shift_bp, event_genomic, ann, gene_idx, event_type='intron_retention', pex_seq=ir_seq, new_junction_tx=new_junction_tx, divergence_pos=event_genomic)
            return ret, 'intron_retention'

    # --- Any other combined donor+acceptor event → Ambiguous ---
    if donor_event and acceptor_event:
        return 'Ambiguous', 'ambiguous'

    # --- Single-channel event: donor XOR acceptor ---
    if donor_event:
        site_channel = 'D'
        ch_ix = 2
        gain_on = donor_gain
        loss_on = donor_loss
        gain_rel = pd_rel
        loss_rel = nd_rel
    else:
        site_channel = 'A'
        ch_ix = 1
        gain_on = acceptor_gain
        loss_on = acceptor_loss
        gain_rel = pa_rel
        loss_rel = na_rel

    center_rel = loss_rel if loss_on else gain_rel
    if _has_competing_alt_sites(y_alt, ch_ix, cov_half, dist_ann_all, center_rel):
        return 'Ambiguous', 'ambiguous'

    # --- Case A: Splice shift (loss of canonical AND gain near it) ---
    if gain_on and loss_on:
        shift_bp = abs(gain_rel - loss_rel)
        if shift_bp == 0:
            return 'NoChange', 'no_change'
            
        is_longer = False
        if site_channel == 'D':
            is_longer = (gain_rel > loss_rel) if strand == '+' else (gain_rel < loss_rel)
        else:
            is_longer = (gain_rel < loss_rel) if strand == '+' else (gain_rel > loss_rel)
            
        event_type = 'insertion' if is_longer else 'deletion'
        struct_ev = f"cryptic_shift_{event_type}"
        
        pex_seq = None
        if is_longer:
            start_g = record.pos + min(gain_rel, loss_rel)
            end_g = record.pos + max(gain_rel, loss_rel)
            pex_seq = get_subsequence(start_g, end_g)
            if pex_seq and strand == '-':
                pex_seq = reverse_complement(pex_seq)
                
        is_minor = False
        alt_cryptic = float(y_alt[0, cov_half + gain_rel, ch_ix])
        alt_canonical = float(y_alt[0, cov_half + loss_rel, ch_ix])
        if alt_canonical > alt_cryptic:
            is_minor = True
            
        event_genomic = record.pos + loss_rel
        if is_longer or site_channel == 'A':
            divergence_pos = record.pos + loss_rel
        else:
            divergence_pos = record.pos + gain_rel
            
        ret = _finalize_frame_call(shift_bp, event_genomic, ann, gene_idx, event_type=event_type, pex_seq=pex_seq, new_junction_tx=new_junction_tx, divergence_pos=divergence_pos)
        if is_minor:
            ret += '_Minor'
        return ret, struct_ev



    # --- Case B: Pure canonical loss (IR vs ES) ---
    if loss_on and not gain_on:
        ir_info, es_info = _lookup_loss_neighbors(
            loss_rel, site_channel, tagged, y_alt, cov_half)
        if ir_info is None and es_info is None:
            return 'Ambiguous', 'ambiguous'
        chosen = _pick_ir_vs_es(ir_info, es_info)
        if chosen == 'ambiguous':
            return 'Ambiguous', 'ambiguous'
        if chosen == 'IR':
            shift_bp = ir_info['size']
            event_type = 'intron_retention'
            struct_ev = 'intron_retention'
            start_g = record.pos + min(loss_rel, ir_info['site_rel'])
            end_g = record.pos + max(loss_rel, ir_info['site_rel'])
            pex_seq = get_subsequence(start_g, end_g)
            if pex_seq and strand == '-':
                pex_seq = reverse_complement(pex_seq)
            divergence_pos = record.pos + loss_rel
        else:
            shift_bp = es_info['size']
            event_type = 'deletion'
            struct_ev = 'exon_skip'
            pex_seq = None
            if site_channel == 'D':
                divergence_pos = record.pos + es_info['site_rel']
            else:
                divergence_pos = None
                
        event_genomic = record.pos + loss_rel
        ret = _finalize_frame_call(shift_bp, event_genomic, ann, gene_idx, event_type=event_type, pex_seq=pex_seq, new_junction_tx=new_junction_tx, divergence_pos=divergence_pos)
        return ret, struct_ev

    # --- Case C: Pure gain (cryptic site without a clear canonical partner loss) ---
    # This acts as an exon extension or truncation, competing with the nearest 
    # canonical site of the same type.
    if gain_on and not loss_on:
        left_site = None
        right_site = None
        for t in tagged:
            if t[0] <= gain_rel:
                left_site = t
            elif right_site is None:
                right_site = t
                break
                
        partner_rel = None
        if left_site and left_site[1] == site_channel and (not right_site or right_site[1] != site_channel):
            partner_rel = left_site[0]
        elif right_site and right_site[1] == site_channel and (not left_site or left_site[1] != site_channel):
            partner_rel = right_site[0]
            
        if partner_rel is not None:
            shift_bp = abs(gain_rel - partner_rel)
            if shift_bp > 0:
                is_minor = False
                alt_cryptic = float(y_alt[0, cov_half + gain_rel, ch_ix])
                alt_canonical = float(y_alt[0, cov_half + partner_rel, ch_ix])
                if alt_canonical > alt_cryptic:
                    is_minor = True
                    
                is_longer = False
                if site_channel == 'D':
                    is_longer = (gain_rel > partner_rel) if strand == '+' else (gain_rel < partner_rel)
                else:
                    is_longer = (gain_rel < partner_rel) if strand == '+' else (gain_rel > partner_rel)
                    
                event_type = 'insertion' if is_longer else 'deletion'
                struct_ev = f"cryptic_extension_{event_type}"
                
                pex_seq = None
                if is_longer:
                    start_g = record.pos + min(gain_rel, partner_rel)
                    end_g = record.pos + max(gain_rel, partner_rel)
                    pex_seq = get_subsequence(start_g, end_g)
                    if pex_seq and strand == '-':
                        pex_seq = reverse_complement(pex_seq)
                        
                event_genomic = record.pos + partner_rel
                
                if is_longer or site_channel == 'A':
                    divergence_pos = record.pos + partner_rel
                else:
                    divergence_pos = record.pos + gain_rel
                    
                ret = _finalize_frame_call(shift_bp, event_genomic, ann, gene_idx, event_type=event_type, pex_seq=pex_seq, new_junction_tx=new_junction_tx, divergence_pos=divergence_pos)
                if is_minor:
                    ret += '_Minor'
                return ret, struct_ev
                
    return 'Ambiguous', 'ambiguous'


def classify_splice_event(y_ref, y_alt, diff_subset,
                           idx_pa, idx_pd, idx_na, idx_nd,
                           ann, gene_ix, gene_info, record,
                           dist_ann_all, cov_half, get_subsequence):
    """
    Wrapper around _classify_splice_event_inner that resolves UTR ambiguities.
    """
    res, struct_ev = _classify_splice_event_inner(
        y_ref, y_alt, diff_subset, idx_pa, idx_pd, idx_na, idx_nd,
        ann, gene_ix, gene_info, record, dist_ann_all, cov_half, get_subsequence
    )
    
    if res == 'Ambiguous':
        gene_idx = gene_info.idxs[gene_ix]
        coding_len = ann.get_coding_length(gene_idx, record.pos)
        if coding_len == -1:
            return '5UTR_Event', 'utr_event'
        elif coding_len == -2:
            return '3UTR_Event', 'utr_event'
            
    return res, struct_ev


def get_alt_gene_delta_score(record, ann, alt_ix, gene_ix, y_ref, y_alt, cov, gene_info, seq, wid, orig=False):
    PROFILING = False
    cov_half = cov // 2

    # Strand handling - needed for both paths
    if gene_info.strands[gene_ix] == '-':
        y_ref = y_ref[:, ::-1]
        y_alt = y_alt[:, ::-1]

    # Safety check: ensure shapes match (they should after encode_seqs fix)
    if y_ref.shape[1] != y_alt.shape[1]:
        min_len = min(y_ref.shape[1], y_alt.shape[1])
        y_ref = y_ref[:, :min_len, :]
        y_alt = y_alt[:, :min_len, :]

    # Shared setup for both orig and extended paths.
    dist_ann_all = ann.get_pos_data(gene_info.idxs[gene_ix], record.pos)
    dist_ann_set = set(dist_ann_all[2])

    # Position-corrected diff: accounts for the downstream sequence shift
    # introduced by indels so that a shifted splice site scores near 0 rather
    # than producing spurious gain=1 + loss=1 simultaneously. Applied to BOTH
    # paths — the extended path previously used a naive subtraction and was
    # producing ghost gain/loss pairs for every indel.
    ref_len_rec = len(record.ref)
    alt_len_rec = len(record.alts[alt_ix])
    ins_len_rec = max(alt_len_rec - ref_len_rec, 0)
    del_len_rec = max(ref_len_rec - alt_len_rec, 0)

    diff_subset = _position_corrected_diff(
        y_alt[0, :, 1:], y_ref[0, :, 1:], cov_half, ins_len_rec, del_len_rec
    )
    max_indices = np.argmax(diff_subset, axis=0)
    min_indices = np.argmin(diff_subset, axis=0)

    idx_pa = max_indices[0]
    idx_pd = max_indices[1]
    idx_na = min_indices[0]
    idx_nd = min_indices[1]

    mask_pa = ((idx_pa - cov_half) in dist_ann_set)
    mask_na = ((idx_na - cov_half) not in dist_ann_set)
    mask_pd = ((idx_pd - cov_half) in dist_ann_set)
    mask_nd = ((idx_nd - cov_half) not in dist_ann_set)

    # FAST PATH for orig=True: minimal computation
    if orig:
        ProfilingStats.increment_iteration()
        # Use corrected delta values for the score output
        score_pa =  diff_subset[idx_pa, 0]
        score_na = -diff_subset[idx_na, 0]
        score_pd =  diff_subset[idx_pd, 1]
        score_nd = -diff_subset[idx_nd, 1]

        return "{}|{}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{}|{}|{}|{}".format(
            record.alts[alt_ix],
            gene_info.genes[gene_ix],
            score_pa * (1 - mask_pa),
            score_na * (1 - mask_na),
            score_pd * (1 - mask_pd),
            score_nd * (1 - mask_nd),
            score_pa,
            score_na,
            score_pd,
            score_nd,
            idx_pa - cov_half,
            idx_na - cov_half,
            idx_pd - cov_half,
            idx_nd - cov_half)

    # EXTENDED PATH: Full computation for detailed output.
    # dist_ann_all / diff_subset / idx_* / mask_* are already populated above.
    with BlockTimer("Init & Array Slicing", PROFILING):
        # Sequence extraction helper (used by the classifier and by fallbacks).
        seq_offset = record.pos - wid // 2 - 1
        def get_subsequence(start_genomic, end_genomic):
            """Return seq[start_genomic .. end_genomic] INCLUSIVE of both ends.

            get_seq() slices the FASTA as [pos - wid//2 - 1 : pos + wid//2], so
            seq[i] has 1-based genomic coordinate seq_offset + i + 1 and a
            genomic coordinate g lives at index g - seq_offset - 1. An earlier
            version omitted that -1 and returned every fragment shifted 1 nt 3'.

            Both endpoints are inclusive because the only caller passes the
            first and last base of a feature (a pseudoexon's acceptor and donor
            positions, each of which is itself exonic). A half-open slice
            returned end - start bases and so under-counted every pseudoexon by
            one, making `pex_size % 3` — and hence the frameshift call — wrong
            for every pseudoexon.
            """
            start_idx = start_genomic - seq_offset - 1
            end_idx = end_genomic - seq_offset          # inclusive of end_genomic
            if start_idx < 0 or end_idx > len(seq):
                return None
            return seq[start_idx:end_idx]

    with BlockTimer("MANE Sites Loop", PROFILING):
        mane_parts = {'Acceptor': [], 'Donor': []}

        mane_indices = [i for i in dist_ann_all[2] if abs(i) < cov_half]

        for i in mane_indices:
            idx = cov_half + i
            donor_ref = y_ref[0, idx, 2]
            acceptor_ref = y_ref[0, idx, 1]
            donor_alt = y_alt[0, idx, 2]
            acceptor_alt = y_alt[0, idx, 1]

        sites_gt05_parts = {'Acceptor': [], 'Donor': []}

        # Check Acceptor (col 1)
        acceptor_mask = (y_ref[0, :, 1] > 0.5) | (y_alt[0, :, 1] > 0.5)
        acceptor_indices = np.flatnonzero(acceptor_mask)

        if len(acceptor_indices) > 0:
            acceptor_ref_scores = y_ref[0, acceptor_indices, 1]
            acceptor_alt_scores = y_alt[0, acceptor_indices, 1]
            sites_gt05_parts['Acceptor'] = [
                f"({pos - cov_half},{ref:.2f},{alt:.2f})"
                for pos, ref, alt in zip(acceptor_indices, acceptor_ref_scores, acceptor_alt_scores)
            ]

        # Check Donor (col 2)
        donor_mask = (y_ref[0, :, 2] > 0.5) | (y_alt[0, :, 2] > 0.5)
        donor_indices = np.flatnonzero(donor_mask)

        if len(donor_indices) > 0:
            donor_ref_scores = y_ref[0, donor_indices, 2]
            donor_alt_scores = y_alt[0, donor_indices, 2]
            sites_gt05_parts['Donor'] = [
                f"({pos - cov_half},{ref:.2f},{alt:.2f})"
                for pos, ref, alt in zip(donor_indices, donor_ref_scores, donor_alt_scores)
            ]

    with BlockTimer("Classify Splice Event", PROFILING):
        event_class, predicted_event = classify_splice_event(
            y_ref, y_alt, diff_subset,
            idx_pa, idx_pd, idx_na, idx_nd,
            ann, gene_ix, gene_info, record,
            dist_ann_all, cov_half, get_subsequence,
        )

    with BlockTimer("String Formatting", PROFILING):
        mane_donor_str = ''.join(f"({s})" for s in mane_parts['Donor']) if mane_parts['Donor'] else '.'
        mane_acceptor_str = ''.join(f"({s})" for s in mane_parts['Acceptor']) if mane_parts['Acceptor'] else '.'
        sites_gt05_donor_str = ''.join(sites_gt05_parts['Donor']) if sites_gt05_parts['Donor'] else '.'
        sites_gt05_acceptor_str = ''.join(sites_gt05_parts['Acceptor']) if sites_gt05_parts['Acceptor'] else '.'

        y_alt_pa = y_alt[0, idx_pa, 1]
        y_ref_pa = y_ref[0, idx_pa, 1]
        y_alt_na = y_alt[0, idx_na, 1]
        y_ref_na = y_ref[0, idx_na, 1]
        y_alt_pd = y_alt[0, idx_pd, 2]
        y_ref_pd = y_ref[0, idx_pd, 2]
        y_alt_nd = y_alt[0, idx_nd, 2]
        y_ref_nd = y_ref[0, idx_nd, 2]

        delta_score = "{}|{}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{}|{}|{}|{}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{}|{}|{}|{}|{}|{}|{}".format(
            record.alts[alt_ix],
            gene_info.genes[gene_ix],
            (y_alt_pa - y_ref_pa) * (1 - mask_pa),
            (y_ref_na - y_alt_na) * (1 - mask_na),
            (y_alt_pd - y_ref_pd) * (1 - mask_pd),
            (y_ref_nd - y_alt_nd) * (1 - mask_nd),
            (y_alt_pa - y_ref_pa),
            (y_ref_na - y_alt_na),
            (y_alt_pd - y_ref_pd),
            (y_ref_nd - y_alt_nd),
            idx_pa - cov_half,
            idx_na - cov_half,
            idx_pd - cov_half,
            idx_nd - cov_half,
            y_alt_pa,
            y_alt_na,
            y_alt_pd,
            y_alt_nd,
            mane_donor_str,
            mane_acceptor_str,
            sites_gt05_donor_str,
            sites_gt05_acceptor_str,
            event_class,
            len(ann.exon_starts[gene_info.idxs[gene_ix]]),
            predicted_event)
    if PROFILING:
        ProfilingStats.increment_iteration()
    return delta_score