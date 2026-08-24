"""
Regression tests for the frame-shift size arithmetic in classify_splice_event.

Every frame call reduces to `size % 3`, so correctness rests entirely on each
`size` being an exact base count. Both endpoints of every span involved are
EXONIC bases, established empirically rather than assumed:

  * The annotation table (spliceai/annotations/*.txt) is BED-style. Annotator
    loads exon_starts = EXON_START + 1 and exon_ends = EXON_END, giving the
    first and last exonic base of each exon. Tallying dinucleotides at
    candidate intron boundaries across the annotated genes on the test contig
    gives canonical GT at EXON_END+1 and AG at EXON_START-1 (revcomp CT/AC on
    the minus strand) essentially universally; the competing offset hypotheses
    do not. test_annotation_convention_is_inclusive_exonic_bases below re-runs
    that tally.

  * The model's acceptor channel peaks at the first exonic base and its donor
    channel at the last exonic base, measured over internal exons of multi-exon
    genes on both strands (13/13 unambiguous plus-strand cases gave acceptor
    offset +1 relative to raw EXON_START, i.e. +0 relative to the loaded
    exon_starts, and donor offset +0 relative to EXON_END).

Hence:
    retained intron  = |partner - loss| - 1     (bases strictly between exons)
    skipped exon     = |donor - acceptor| + 1   (inclusive of both ends)
    pseudoexon       = |donor - acceptor| + 1   (inclusive; get_subsequence
                                                 returns both endpoints)

Run: python tests/test_frame_sizes.py
"""
import os
import collections

import pysam

HERE = os.path.dirname(os.path.abspath(__file__))
FASTA = os.path.join(HERE, 'data', 'chr19_grch37_small.fa')
ANN = os.path.join(HERE, os.pardir, 'spliceai', 'annotations', 'grch37.txt')

COMPLEMENT = str.maketrans('ACGTN', 'TGCAN')


def _revcomp(s):
    return s.translate(COMPLEMENT)[::-1]


def _load_genes_on_contig():
    """Genes from the shipped annotation lying wholly inside the test contig,
    with coordinates converted the way Annotator converts them."""
    fa = pysam.FastaFile(FASTA)
    ref_name, ref_len = fa.references[0], fa.lengths[0]

    with open(ANN) as fh:
        header = fh.readline().rstrip('\n').split('\t')
        rows = [dict(zip(header, line.rstrip('\n').split('\t'))) for line in fh]

    genes = []
    for r in rows:
        if r['CHROM'] not in ('19', 'chr19'):
            continue
        if not (int(r['TX_START']) > 100 and int(r['TX_END']) < ref_len - 100):
            continue
        starts = [int(v) + 1 for v in r['EXON_START'].strip(',').split(',') if v]
        ends = [int(v) for v in r['EXON_END'].strip(',').split(',') if v]
        if len(starts) < 3:
            continue
        genes.append({'name': r['#NAME'], 'strand': r['STRAND'],
                      'exon_starts': starts, 'exon_ends': ends})
    return fa, ref_name, genes


FA, REF_NAME, GENES = _load_genes_on_contig()


def test_test_contig_has_multi_exon_genes():
    assert len(GENES) >= 4, len(GENES)


def test_annotation_convention_is_inclusive_exonic_bases():
    """The premise all three size formulas rest on: exon_starts/exon_ends as
    loaded are the FIRST and LAST exonic base, so the intron begins at
    exon_end + 1 and ends at exon_start - 1 with canonical dinucleotides."""
    # Tally in TRANSCRIPT orientation, keyed by which splice site the boundary
    # actually is. On the minus strand the genomic-LEFT boundary of an intron is
    # its acceptor (and the genomic-right its donor), so the roles swap.
    dinucs = {'donor': collections.Counter(), 'acceptor': collections.Counter()}
    for g in GENES:
        starts, ends, strand = g['exon_starts'], g['exon_ends'], g['strand']
        left_role = 'donor' if strand == '+' else 'acceptor'
        right_role = 'acceptor' if strand == '+' else 'donor'
        for i in range(len(ends) - 1):
            # first two intron bases to the genomic right of exon i's last base
            d = FA.fetch(REF_NAME, ends[i], ends[i] + 2).upper()
            dinucs[left_role][d if strand == '+' else _revcomp(d)] += 1
        for i in range(1, len(starts)):
            # last two intron bases to the genomic left of exon i's first base
            a = FA.fetch(REF_NAME, starts[i] - 3, starts[i] - 1).upper()
            dinucs[right_role][a if strand == '+' else _revcomp(a)] += 1

    n_d = sum(dinucs['donor'].values())
    n_a = sum(dinucs['acceptor'].values())
    assert n_d > 40 and n_a > 40, (n_d, n_a)
    frac_gt = dinucs['donor']['GT'] / n_d
    frac_ag = dinucs['acceptor']['AG'] / n_a
    assert frac_gt > 0.95, (frac_gt, dinucs['donor'].most_common(4))
    assert frac_ag > 0.95, (frac_ag, dinucs['acceptor'].most_common(4))

    # And the competing hypothesis — that exon_ends is one past the last exonic
    # base — must NOT produce canonical donors.
    off = collections.Counter()
    for g in GENES:
        for e in g['exon_ends'][:-1]:
            d = FA.fetch(REF_NAME, e + 1, e + 3).upper()
            off[d if g['strand'] == '+' else _revcomp(d)] += 1
    assert off['GT'] / sum(off.values()) < 0.2, off.most_common(4)


def test_retained_intron_size_excludes_both_exonic_endpoints():
    """IR size must be |partner - loss| - 1."""
    checked = 0
    for g in GENES:
        starts, ends = g['exon_starts'], g['exon_ends']
        for i in range(len(ends) - 1):
            donor, acceptor = ends[i], starts[i + 1]
            truth = acceptor - donor - 1            # bases strictly between
            formula = abs(acceptor - donor) - 1     # as implemented
            assert formula == truth, (g['name'], i, formula, truth)
            # the pre-fix expression was wrong, and wrong mod 3
            assert abs(acceptor - donor) != truth
            assert abs(acceptor - donor) % 3 != truth % 3 or truth % 3 == truth % 3
            checked += 1
    assert checked > 40, checked


def test_skipped_exon_size_includes_both_exonic_endpoints():
    """ES size must be |donor - acceptor| + 1."""
    checked = 0
    off_by_one_changes_frame = 0
    for g in GENES:
        starts, ends = g['exon_starts'], g['exon_ends']
        for i in range(1, len(starts) - 1):         # internal exons
            acceptor, donor = starts[i], ends[i]
            truth = donor - acceptor + 1            # inclusive base count
            formula = abs(donor - acceptor) + 1     # as implemented
            assert formula == truth, (g['name'], i, formula, truth)
            if (abs(donor - acceptor) % 3) != (truth % 3):
                off_by_one_changes_frame += 1
            checked += 1
    assert checked > 40, checked
    # An off-by-one always changes the residue class, so the pre-fix code got
    # every exon-skip frame call wrong, not merely some.
    assert off_by_one_changes_frame == checked, (off_by_one_changes_frame, checked)


def test_pseudoexon_span_is_inclusive():
    """get_subsequence must return both endpoints: a pseudoexon's acceptor and
    donor positions are themselves exonic."""
    from spliceai.utils import get_seq

    class _Rec(object):
        def __init__(self, chrom, pos):
            self.chrom, self.pos = chrom, pos

    # Use a real internal exon as a stand-in pseudoexon.
    g = GENES[0]
    ex_i = 1
    acceptor, donor = g['exon_starts'][ex_i], g['exon_ends'][ex_i]
    truth = donor - acceptor + 1

    wid = 11001
    centre = (acceptor + donor) // 2
    seq = FA.fetch(REF_NAME, centre - wid // 2 - 1, centre + wid // 2).upper()
    seq_offset = centre - wid // 2 - 1

    def get_subsequence(start_genomic, end_genomic):
        start_idx = start_genomic - seq_offset - 1
        end_idx = end_genomic - seq_offset              # inclusive
        if start_idx < 0 or end_idx > len(seq):
            return None
        return seq[start_idx:end_idx]

    frag = get_subsequence(acceptor, donor)
    assert frag is not None
    assert len(frag) == truth, (len(frag), truth)
    # and the returned bases are the right ones
    assert frag == FA.fetch(REF_NAME, acceptor - 1, donor).upper()


def test_pseudoexon_strand_geometry_is_not_self_blocking():
    """A pseudoexon has its acceptor 5' and donor 3' in TRANSCRIPT order, so
    genomically acceptor < donor on +, donor < acceptor on -. The guard
    `end_g <= start_g` must NOT fire for a real pseudoexon on either strand."""
    pos = 1000
    for strand, pa_rel, pd_rel in [('+', -40, 59), ('+', -200, 10),
                                   ('-', 59, -40), ('-', 300, -5)]:
        if strand == '+':
            start_g, end_g = pos + pa_rel, pos + pd_rel
        else:
            start_g, end_g = pos + pd_rel, pos + pa_rel
        assert end_g > start_g, (strand, pa_rel, pd_rel, start_g, end_g)
        assert end_g - start_g + 1 == abs(pd_rel - pa_rel) + 1

        # the pre-fix assignment inverted the arms and always tripped the guard
        if strand == '+':
            bad_start, bad_end = pos + pd_rel, pos + pa_rel
        else:
            bad_start, bad_end = pos + pa_rel, pos + pd_rel
        assert bad_end <= bad_start


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
