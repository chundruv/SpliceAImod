"""
Standalone verification of the in-frame / out-of-frame (frameshift) splice
event classification logic in spliceai.utils (classify_splice_event and its
helpers: _finalize_frame_call, predict_nmd, _genomic_to_transcript_pos,
has_stop_codon).

This imports the REAL production code (not a reimplementation). Heavy
dependencies that classify_splice_event's code path never actually needs
(torch, pandas, pyfaidx, intervaltree) are stubbed out in sys.modules before
import, since they're only exercised by unrelated parts of utils.py (model
definition, VCF/FASTA I/O) that we don't call here.

Run with: python3 tests/test_frame_classification.py
"""
import sys
import os
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# ---- Stub heavy deps not needed for classify_splice_event's code path ----
if 'torch' not in sys.modules:
    torch_mod = types.ModuleType('torch')
    nn_mod = types.ModuleType('torch.nn')
    functional_mod = types.ModuleType('torch.nn.functional')

    class _Module:
        def __init__(self, *a, **kw):
            pass

    nn_mod.Module = _Module
    nn_mod.BatchNorm1d = _Module
    nn_mod.Conv1d = _Module
    nn_mod.functional = functional_mod
    torch_mod.nn = nn_mod
    sys.modules['torch'] = torch_mod
    sys.modules['torch.nn'] = nn_mod
    sys.modules['torch.nn.functional'] = functional_mod

if 'pandas' not in sys.modules:
    pandas_mod = types.ModuleType('pandas')
    errors_mod = types.ModuleType('pandas.errors')
    errors_mod.ParserError = type('ParserError', (Exception,), {})
    pandas_mod.errors = errors_mod
    sys.modules['pandas'] = pandas_mod
    sys.modules['pandas.errors'] = errors_mod

if 'pyfaidx' not in sys.modules:
    pyfaidx_mod = types.ModuleType('pyfaidx')
    pyfaidx_mod.Fasta = type('Fasta', (), {})
    sys.modules['pyfaidx'] = pyfaidx_mod

if 'intervaltree' not in sys.modules:
    intervaltree_mod = types.ModuleType('intervaltree')
    intervaltree_mod.IntervalTree = type('IntervalTree', (), {})
    sys.modules['intervaltree'] = intervaltree_mod

import numpy as np  # noqa: E402
import importlib.util  # noqa: E402

# Load spliceai/utils.py directly as a standalone module, bypassing
# spliceai/__init__.py (which calls importlib.metadata.version('spliceai') and
# raises PackageNotFoundError when the package isn't pip-installed).
_spec = importlib.util.spec_from_file_location(
    "spliceai_utils_standalone", os.path.join(REPO_ROOT, "spliceai", "utils.py"))
_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils)

Annotator = _utils.Annotator
GeneInfo = _utils.GeneInfo
classify_splice_event = _utils.classify_splice_event
has_stop_codon = _utils.has_stop_codon
find_stop_in_frame = _utils.find_stop_in_frame
predict_nmd = _utils.predict_nmd
_genomic_to_transcript_pos = _utils._genomic_to_transcript_pos
reverse_complement = _utils.reverse_complement

FAILURES = []


def check(label, actual, expected):
    ok = actual == expected
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {label}: got {actual!r}, expected {expected!r}")
    if not ok:
        FAILURES.append(label)


class FakeAnn:
    """Duck-typed stand-in for Annotator, exposing only what
    classify_splice_event's helpers actually touch: exon_starts, exon_ends,
    strands (for _genomic_to_transcript_pos) and get_last_junction_transcript_pos
    (reusing the REAL Annotator method bound to this fake instance)."""

    def __init__(self, exon_starts, exon_ends, strands):
        self.exon_starts = exon_starts
        self.exon_ends = exon_ends
        self.strands = strands
        self._last_junction_cache = {}

    def get_last_junction_transcript_pos(self, idx):
        return Annotator.get_last_junction_transcript_pos(self, idx)


class FakeRecord:
    def __init__(self, pos):
        self.pos = pos


def make_donor_shift_arrays(cov_half, loss_rel, gain_rel):
    L = 2 * cov_half + 1
    y_ref = np.zeros((1, L, 3), dtype=np.float64)
    y_alt = np.zeros((1, L, 3), dtype=np.float64)
    diff_subset = np.zeros((L, 2), dtype=np.float64)

    y_ref[0, cov_half + loss_rel, 2] = 0.9   # canonical donor, ref
    y_alt[0, cov_half + gain_rel, 2] = 0.9   # new donor, alt

    diff_subset[cov_half + loss_rel, 1] = -0.9  # donor loss (argmin)
    diff_subset[cov_half + gain_rel, 1] = 0.9   # donor gain (argmax)

    idx_pa = int(np.argmax(diff_subset[:, 0]))
    idx_pd = int(np.argmax(diff_subset[:, 1]))
    idx_na = int(np.argmin(diff_subset[:, 0]))
    idx_nd = int(np.argmin(diff_subset[:, 1]))
    return y_ref, y_alt, diff_subset, idx_pa, idx_pd, idx_na, idx_nd


def run_donor_shift_case(label, ann, record, cov_half, loss_rel, gain_rel,
                          gained_seq, expected):
    y_ref, y_alt, diff_subset, idx_pa, idx_pd, idx_na, idx_nd = \
        make_donor_shift_arrays(cov_half, loss_rel, gain_rel)
    dist_ann_all = (None, None, [loss_rel])
    gene_info = GeneInfo(genes=np.array(['GENE1']), strands=np.array(['+']),
                          idxs=np.array([0]))

    def get_subsequence(start, end):
        assert end - start == len(gained_seq), (start, end, gained_seq)
        return gained_seq

    result = classify_splice_event(
        y_ref, y_alt, diff_subset, idx_pa, idx_pd, idx_na, idx_nd,
        ann, 0, gene_info, record, dist_ann_all, cov_half, get_subsequence,
    )
    check(label, result, expected)


def run_pseudoexon_case(label, ann, record, cov_half, pd_rel, pa_rel,
                         pex_seq, expected, strand='+'):
    L = 2 * cov_half + 1
    y_ref = np.zeros((1, L, 3), dtype=np.float64)
    y_alt = np.zeros((1, L, 3), dtype=np.float64)
    diff_subset = np.zeros((L, 2), dtype=np.float64)

    y_alt[0, cov_half + pd_rel, 2] = 0.9  # new donor, alt
    y_alt[0, cov_half + pa_rel, 1] = 0.9  # new acceptor, alt
    diff_subset[cov_half + pd_rel, 1] = 0.9   # donor gain (argmax)
    diff_subset[cov_half + pa_rel, 0] = 0.9   # acceptor gain (argmax)

    idx_pa = int(np.argmax(diff_subset[:, 0]))
    idx_pd = int(np.argmax(diff_subset[:, 1]))
    idx_na = int(np.argmin(diff_subset[:, 0]))
    idx_nd = int(np.argmin(diff_subset[:, 1]))

    dist_ann_all = (None, None, [])
    gene_info = GeneInfo(genes=np.array(['GENE1']), strands=np.array([strand]),
                          idxs=np.array([0]))

    def get_subsequence(start, end):
        assert end - start == len(pex_seq), (start, end, pex_seq)
        # On '-' the classifier reverse-complements what it gets back, so hand
        # it the genomic-orientation sequence and let it do that.
        return reverse_complement(pex_seq) if strand == '-' else pex_seq

    result = classify_splice_event(
        y_ref, y_alt, diff_subset, idx_pa, idx_pd, idx_na, idx_nd,
        ann, 0, gene_info, record, dist_ann_all, cov_half, get_subsequence,
    )
    check(label, result, expected)


def main():
    # 2-exon '+' strand gene: exon1 = genomic [1,100], exon2 = genomic [200,300].
    # last junction (start of last exon, transcript-relative) = len(exon1) = 100.
    ann = FakeAnn(
        exon_starts=[np.array([1, 200])],
        exon_ends=[np.array([100, 300])],
        strands=['+'],
    )
    last_junction = ann.get_last_junction_transcript_pos(0)
    check("last_junction_transcript_pos (sanity check)", last_junction, 100)

    print("\n--- Pure helper functions ---")
    check("has_stop_codon('AAA')", has_stop_codon("AAA"), False)
    check("has_stop_codon('TAA')", has_stop_codon("TAA"), True)
    check("has_stop_codon('AATAAA')", has_stop_codon("AATAAA"), False)  # stop not on frame-0 codon boundary
    check("find_stop_in_frame('AATAAA', 0)", find_stop_in_frame("AATAAA", 0), None)
    check("find_stop_in_frame('AATAAA', 2)", find_stop_in_frame("AATAAA", 2), 2)
    check("predict_nmd(0, 100) [55nt window]", predict_nmd(0, 100), True)
    check("predict_nmd(89, 100) [11nt before junction]", predict_nmd(89, 100), False)
    check("predict_nmd(44, 100) [boundary-1]", predict_nmd(44, 100), True)
    check("predict_nmd(45, 100) [boundary]", predict_nmd(45, 100), False)
    check("_genomic_to_transcript_pos(exon1 start)", _genomic_to_transcript_pos(ann, 0, 1), 0)
    check("_genomic_to_transcript_pos(exon1 end)", _genomic_to_transcript_pos(ann, 0, 100), 99)

    print("\n--- Case A: donor-site splice shift (extend/truncate exon) ---")
    # 3nt shift (multiple of 3) -> in-frame, no stop in the shifted bases -> InFrame
    run_donor_shift_case(
        "shift=+3nt, no stop -> InFrame",
        ann, FakeRecord(pos=50), cov_half=50, loss_rel=0, gain_rel=3,
        gained_seq="AAA", expected="InFrame")

    # 3nt shift whose gained bases happen to read TAA in frame 0 of the genomic
    # fragment. This is now InFrame, NOT Truncating_NoFS: the transcript phase of
    # the fragment is unknown, so "TAA is a stop here" cannot be asserted. The
    # frame-shift call (3 % 3 == 0) is phase-independent and stands.
    run_donor_shift_case(
        "shift=+3nt, stop-like bases -> InFrame (no phase anchor)",
        ann, FakeRecord(pos=50), cov_half=50, loss_rel=0, gain_rel=3,
        gained_seq="TAA", expected="InFrame")

    # 2nt shift (not a multiple of 3) near the last junction -> frameshift, NMD escape
    run_donor_shift_case(
        "shift=+2nt near last junction -> Frameshift_NMDescape",
        ann, FakeRecord(pos=90), cov_half=50, loss_rel=0, gain_rel=2,
        gained_seq="AA", expected="Frameshift_NMDescape")

    # 2nt shift far from the last junction -> frameshift, NMD triggered
    run_donor_shift_case(
        "shift=+2nt far from last junction -> Frameshift_NMD",
        ann, FakeRecord(pos=1), cov_half=50, loss_rel=0, gain_rel=2,
        gained_seq="AA", expected="Frameshift_NMD")

    print("\n--- Pseudoexon insertion ---")
    # GEOMETRY: on the '+' strand the transcript runs left-to-right, so an
    # inserted exon has its ACCEPTOR at the genomic-left edge and its DONOR at
    # the genomic-right edge: pa_rel < pd_rel. These three cases originally had
    # the two offsets the other way round, matching the inverted strand arms
    # that classify_splice_event used to carry. With that inversion fixed, the
    # old offsets make end_g <= start_g and every case returns 'Ambiguous' --
    # the tests were encoding the bug. Swapping pd_rel/pa_rel leaves each span
    # (and therefore pex_seq) unchanged.
    #
    # Length-preserving pseudoexons report 'Pseudoexon' rather than the generic
    # 'InFrame', so the event type survives in the output.
    run_pseudoexon_case(
        "3nt pseudoexon -> Pseudoexon (length-preserving)",
        ann, FakeRecord(pos=50), cov_half=50, pa_rel=10, pd_rel=13,
        pex_seq="CCC", expected="Pseudoexon")

    # 6nt is also a multiple of 3, so this is length-preserving too. It was
    # Pseudoexon_PoisonExon only because AAATAG contains TAG in frame 0 of an
    # arbitrary genomic fragment -- not a defensible PTC call.
    run_pseudoexon_case(
        "6nt pseudoexon, stop-like bases -> Pseudoexon (no phase anchor)",
        ann, FakeRecord(pos=50), cov_half=50, pa_rel=10, pd_rel=16,
        pex_seq="AAATAG", expected="Pseudoexon")

    run_pseudoexon_case(
        "5nt pseudoexon, no stop -> frameshift (NMD or NMDescape, not InFrame)",
        ann, FakeRecord(pos=1), cov_half=50, pa_rel=10, pd_rel=15,
        pex_seq="AAAAA", expected="Frameshift_NMD")

    # '-' strand: transcript runs right-to-left, so the arms mirror -- the DONOR
    # is at the genomic-left edge and the ACCEPTOR at the right (pd_rel <
    # pa_rel). This arm had no coverage, which is why the inverted strand logic
    # survived; the pre-fix code would return 'Ambiguous' here.
    ann_minus = FakeAnn(
        exon_starts=[np.array([1, 200])],
        exon_ends=[np.array([100, 300])],
        strands=['-'],
    )
    run_pseudoexon_case(
        "3nt pseudoexon on '-' strand -> Pseudoexon",
        ann_minus, FakeRecord(pos=50), cov_half=50, pd_rel=10, pa_rel=13,
        pex_seq="CCC", expected="Pseudoexon", strand='-')

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()
