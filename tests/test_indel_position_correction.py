"""
Regression tests for _position_corrected_diff (indel delta scoring).

Truth model. For a left-anchored VCF indel, encode_seqs builds
    x_alt = x_ref[:wid//2] + alt + x_ref[wid//2 + ref_len:]
so with cov_half the output index of the variant position and
shift = del_len - ins_len:

  * reference index i <= cov_half            -> alt index i          (shared)
  * cov_half < i <= cov_half + del_len       -> deleted, no alt base
  * i > cov_half + del_len                   -> alt index i - shift
  * inserted bases occupy alt indices cov_half+1 .. cov_half+ins_len and have
    no reference position at all

The returned delta array is indexed in REFERENCE coordinates: index i means
reference offset (i - cov_half) from record.pos, which is the frame callers
report and the frame annotated exon boundaries are in.

Run: python tests/test_indel_position_correction.py
"""
import numpy as np

from spliceai.utils import _position_corrected_diff

L = 101
COV_HALF = 50
TOL = 1e-9


def build(ref, alt, sites_ref=(), novel_alt=()):
    """Build a truth-consistent (y_ref, y_alt, ins_len, del_len).

    sites_ref : (ref_index, score) sites present in the reference. A site
                inside the deleted span has no alt counterpart; otherwise it is
                preserved, at alt index (ref_index - shift) when downstream.
    novel_alt : (alt_index, score) sites present only in the alt allele.
    """
    ref_len, alt_len = len(ref), len(alt)
    ins_len = max(alt_len - ref_len, 0)
    del_len = max(ref_len - alt_len, 0)
    shift = del_len - ins_len

    y_ref = np.zeros((L, 2))
    y_alt = np.zeros((L, 2))
    for ref_i, score in sites_ref:
        y_ref[ref_i, 0] = score
        if ref_i <= COV_HALF:
            alt_i = ref_i
        elif ref_i <= COV_HALF + del_len:
            continue                      # deleted by the variant
        else:
            alt_i = ref_i - shift
        if 0 <= alt_i < L:
            y_alt[alt_i, 0] = score
    for alt_i, score in novel_alt:
        y_alt[alt_i, 0] = score
    return y_ref, y_alt, ins_len, del_len


CASES = [
    ("SNV",      "A",        "A"),
    ("1bp DEL",  "AT",       "A"),
    ("5bp DEL",  "ATTTTT",   "A"),
    ("30bp DEL", "A" * 31,   "A"),
    ("1bp INS",  "A",        "AT"),
    ("4bp INS",  "A",        "ATTTT"),
    ("20bp INS", "A",        "A" + "T" * 20),
    ("60bp INS", "A",        "A" + "T" * 60),
]


def test_preserved_sites_give_zero_delta():
    """The defect this function exists to fix: a site that merely MOVED must
    not produce a simultaneous gain and loss."""
    for label, ref, alt in CASES:
        del_len = max(len(ref) - len(alt), 0)
        sites = [(COV_HALF, 1.0), (COV_HALF - 10, 0.8), (COV_HALF - 20, 0.7),
                 (COV_HALF + del_len + 5, 0.9), (COV_HALF + del_len + 15, 0.6)]
        sites = [(i, s) for i, s in sites if 0 <= i < L]
        y_ref, y_alt, ins_len, dl = build(ref, alt, sites_ref=sites)
        d = _position_corrected_diff(y_alt, y_ref, COV_HALF, ins_len, dl)
        assert np.abs(d).max() < TOL, (label, np.abs(d).max())


def test_no_spurious_delta_at_variant_position():
    """Regression: the remap once started at cov_half rather than past the
    shared anchor base, so p_alt(anchor) was compared against
    p_ref(anchor + shift) — injecting a delta of up to 1.0 at the variant
    position itself whenever a strong site sat `shift` bp away.

    The anchor base at cov_half is SHARED between ref and alt, so its delta
    must be plain subtraction regardless of del_len/ins_len.
    """
    for del_len in (1, 2, 5, 20):
        # A strong site just past the deleted span, preserved but shifted onto
        # a position `del_len` bp closer to the variant in alt coordinates.
        ref_i = COV_HALF + del_len + 1
        y_ref = np.zeros((L, 2))
        y_ref[ref_i, 0] = 1.0
        y_alt = np.zeros((L, 2))
        y_alt[ref_i - del_len, 0] = 1.0
        d = _position_corrected_diff(y_alt, y_ref, COV_HALF, 0, del_len)
        assert abs(d[COV_HALF, 0]) < TOL, ('deletion', del_len, d[COV_HALF, 0])
        assert np.abs(d).max() < TOL, ('deletion', del_len, np.abs(d).max())

    for ins_len in (1, 2, 5, 20):
        ref_i = COV_HALF + 1
        y_ref = np.zeros((L, 2))
        y_ref[ref_i, 0] = 1.0
        y_alt = np.zeros((L, 2))
        y_alt[ref_i + ins_len, 0] = 1.0
        d = _position_corrected_diff(y_alt, y_ref, COV_HALF, ins_len, 0)
        assert abs(d[COV_HALF, 0]) < TOL, ('insertion', ins_len, d[COV_HALF, 0])


def test_reported_index_is_a_reference_offset():
    """argmax index - cov_half must be a REFERENCE offset, since callers emit
    it as a genomic offset and match it against annotated exon boundaries."""
    del_len, ref_off = 6, 20
    y_ref = np.zeros((L, 2))
    y_alt = np.zeros((L, 2))
    y_alt[COV_HALF + ref_off - del_len, 0] = 0.91   # gain at reference offset +20
    d = _position_corrected_diff(y_alt, y_ref, COV_HALF, 0, del_len)
    assert int(d[:, 0].argmax()) - COV_HALF == ref_off, int(d[:, 0].argmax()) - COV_HALF


def test_site_inside_deleted_span_reports_loss():
    """A reference splice site removed by a deletion must report its full
    loss, not delta 0."""
    for del_len, site_off in [(3, 1), (6, 3), (10, 5), (30, 17)]:
        y_ref = np.zeros((L, 2))
        y_ref[COV_HALF + site_off, 0] = 0.97
        y_alt = np.zeros((L, 2))
        d = _position_corrected_diff(y_alt, y_ref, COV_HALF, 0, del_len)
        assert abs(d[:, 0].min() + 0.97) < TOL, (del_len, site_off, d[:, 0].min())
        assert int(d[:, 0].argmin()) - COV_HALF == site_off


def test_genuine_gain_inside_insertion_is_reported_at_variant():
    """Inserted sequence has no reference position; a site called there is
    attributed to the variant position."""
    y_ref = np.zeros((L, 2))
    y_alt = np.zeros((L, 2))
    y_alt[COV_HALF + 2, 0] = 0.95                   # inside a 4bp insertion
    d = _position_corrected_diff(y_alt, y_ref, COV_HALF, 4, 0)
    assert abs(d[COV_HALF, 0] - 0.95) < TOL, d[COV_HALF, 0]


def test_genuine_loss_downstream_is_reported():
    y_ref = np.zeros((L, 2))
    y_ref[COV_HALF + 20, 0] = 0.93
    y_alt = np.zeros((L, 2))
    d = _position_corrected_diff(y_alt, y_ref, COV_HALF, 0, 1)
    assert abs(d[:, 0].min() + 0.93) < TOL
    assert int(d[:, 0].argmin()) - COV_HALF == 20


def test_unmappable_tail_yields_zero_not_a_fabricated_call():
    """Where the alt window ends before a reference position, nothing can be
    said. Regression: this region was zero-filled, fabricating a maximal
    call at window positions whose only defect was running out of window.

    Only INSERTIONS produce such a tail: shift = del_len - ins_len is negative,
    so alt_idx = ref_idx - shift runs past L for the last ins_len reference
    positions.
    """
    for ins_len in (1, 8, 30):
        rng = np.random.default_rng(7)
        y_ref = np.zeros((L, 2))
        y_ref[-ins_len:, :] = rng.random((ins_len, 2))   # strong scores in the tail
        y_alt = y_ref.copy()
        d = _position_corrected_diff(y_alt, y_ref, COV_HALF, ins_len, 0)
        assert np.abs(d[-ins_len:, :]).max() < TOL, (ins_len, np.abs(d[-ins_len:, :]).max())


def test_snv_path_is_plain_subtraction():
    rng = np.random.default_rng(0)
    y_ref, y_alt = rng.random((L, 2)), rng.random((L, 2))
    d = _position_corrected_diff(y_alt, y_ref, COV_HALF, 0, 0)
    assert np.array_equal(d, y_alt - y_ref)


def test_shape_dtype_and_degenerate_inputs():
    rng = np.random.default_rng(1)
    y_ref = rng.random((L, 2)).astype(np.float32)
    y_alt = rng.random((L, 2)).astype(np.float32)
    d = _position_corrected_diff(y_alt, y_ref, COV_HALF, 0, 4)
    assert d.shape == (L, 2) and d.dtype == np.float32, (d.shape, d.dtype)

    # deletion longer than the remaining window
    d = _position_corrected_diff(np.zeros((L, 2)), np.zeros((L, 2)), COV_HALF, 0, 200)
    assert not np.isnan(d).any()


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in tests:
        try:
            fn()
            print('PASS  {}'.format(fn.__name__))
        except AssertionError as e:
            failed += 1
            print('FAIL  {}: {}'.format(fn.__name__, e))
    print('\n{}/{} passed'.format(len(tests) - failed, len(tests)))
    raise SystemExit(1 if failed else 0)
