import os
import sys
import gzip
import tempfile
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from spliceai.utils import get_tsv_header_comments, get_tsv_header_line


def test_tsv_header_line_extended():
    header_line = get_tsv_header_line(orig=False)
    assert header_line.startswith("#CHROM\tPOS\tREF\tALT\tALLELE\tSYMBOL\t")
    cols = header_line.strip().split("\t")
    expected_cols = [
        '#CHROM', 'POS', 'REF', 'ALT', 'ALLELE', 'SYMBOL',
        'DSM_AG', 'DSM_AL', 'DSM_DG', 'DSM_DL',
        'DS_AG', 'DS_AL', 'DS_DG', 'DS_DL',
        'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL',
        'RS_AG', 'RS_AL', 'RS_DG', 'RS_DL',
        'MANEselectDonors', 'MANEselectAcceptors',
        'DonorsRSgt0.5', 'AcceptorsRSgt0.5',
        'EVENT_CLASS'
    ]
    assert cols == expected_cols


def test_tsv_header_line_orig():
    header_line = get_tsv_header_line(orig=True)
    assert header_line.startswith("#CHROM\tPOS\tREF\tALT\tALLELE\tSYMBOL\t")
    cols = header_line.strip().split("\t")
    expected_cols = [
        '#CHROM', 'POS', 'REF', 'ALT', 'ALLELE', 'SYMBOL',
        'DSM_AG', 'DSM_AL', 'DSM_DG', 'DSM_DL',
        'DS_AG', 'DS_AL', 'DS_DG', 'DS_DL',
        'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL'
    ]
    assert cols == expected_cols


def test_tsv_header_comments():
    comments_extended = get_tsv_header_comments(orig=False)
    assert comments_extended.startswith("# ")
    assert "EVENT_CLASS:" in comments_extended
    assert "MANEselectDonors:" in comments_extended

    comments_orig = get_tsv_header_comments(orig=True)
    assert comments_orig.startswith("# ")
    assert "SpliceAIv1.3.1-compatible" in comments_orig
    assert "EVENT_CLASS:" not in comments_orig


def test_tsv_gzipped_writer_output():
    with tempfile.NamedTemporaryFile(suffix=".tsv.gz", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with gzip.open(tmp_path, "wt") as f:
            f.write(get_tsv_header_comments(orig=False))
            f.write(get_tsv_header_line(orig=False))

            # Simulate record with 2 transcripts / alts
            dummy_record_chrom = "chr1"
            dummy_record_pos = 1000
            dummy_record_ref = "A"
            dummy_record_alts = ("C", "G")
            alt_str = ",".join(dummy_record_alts)

            score_strings = [
                "C|GENE1|0.00|0.00|0.50|0.00|0.00|0.00|0.50|0.00|0|0|10|0|0.01|0.02|0.52|0.01|.|. |.|.|NoChange",
                "C|GENE2|0.00|0.00|0.10|0.00|0.00|0.00|0.10|0.00|0|0|10|0|0.01|0.02|0.12|0.01|.|. |.|.|NoChange",
                "G|GENE1|0.00|0.00|0.80|0.00|0.00|0.00|0.80|0.00|0|0|10|0|0.01|0.02|0.82|0.01|.|. |.|.|NoChange",
            ]

            for score_str in score_strings:
                score_fields = score_str.split("|")
                row_fields = [dummy_record_chrom, str(dummy_record_pos), dummy_record_ref, alt_str] + score_fields
                f.write("\t".join(row_fields) + "\n")

        # Read back gzipped file
        with gzip.open(tmp_path, "rt") as f:
            lines = [line.rstrip("\n") for line in f]

        comment_lines = [l for l in lines if l.startswith("#") and not l.startswith("#CHROM")]
        header_lines = [l for l in lines if l.startswith("#CHROM")]
        data_lines = [l for l in lines if not l.startswith("#")]

        assert len(comment_lines) > 0
        assert len(header_lines) == 1
        assert len(data_lines) == 3

        # Verify first row fields
        row0_cols = data_lines[0].split("\t")
        assert row0_cols[0] == "chr1"
        assert row0_cols[1] == "1000"
        assert row0_cols[2] == "A"
        assert row0_cols[3] == "C,G"
        assert row0_cols[4] == "C"       # ALLELE
        assert row0_cols[5] == "GENE1"   # SYMBOL
        assert row0_cols[6] == "0.00"    # DSM_AG
        assert len(row0_cols) == 27      # 4 pos/ref/alt + 23 info fields

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
