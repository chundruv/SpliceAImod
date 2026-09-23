#!/usr/bin/env python3
"""Collapse a transcript-level SpliceAI annotation to one row per gene.

The bundled gencode.v49.annotation.txt has one row per TRANSCRIPT (507k rows, ENST names).
SpliceAI pads the input sequence to the transcript boundaries, so a variant that lies within
CL/2+D of several transcript ends of the same gene gets one GPU inference per distinct boundary
(measured on the near-splice set: ~9 inferences per variant). The original SpliceAI annotation
(grch38.txt, GENCODE v24) is gene-level: TX_START/TX_END are the gene span and EXON_START/END
are the union of every transcript's exon boundaries, so each variant costs ~1 inference and the
masked scores use all annotated splice sites of the gene.

This script rebuilds that gene-level layout from GENCODE v49 (needs the GTF for the
transcript -> gene mapping, which the annotation file itself does not carry):

  python scripts/collapse_annotation_to_genes.py \
      gencode.v49.primary_assembly.annotation.gtf.gz \
      spliceai/annotations/gencode.v49.annotation.txt \
      spliceai/annotations/gencode.v49.genes.txt

Then run with  -A spliceai/annotations/gencode.v49.genes.txt   (a path works for -A).
Only transcripts present in the input annotation are used, so the transcript set is unchanged.
CDS_START/CDS_END are set to -1: a gene has no single CDS, so the NMD/event-class fields that
need one are reported as unknown; the SpliceAI delta scores themselves do not use the CDS.
NAME is the gene symbol (gene_name), falling back to gene_id.
"""
import gzip, sys, collections


def main(gtf, tx_ann, out):
    tx2gene = {}
    opener = gzip.open if gtf.endswith('.gz') else open
    with opener(gtf, 'rt') as f:
        for line in f:
            if line[0] == '#':
                continue
            p = line.split('\t')
            if p[2] != 'transcript':
                continue
            tid = gid = gname = None
            for kv in p[8].split(';'):
                kv = kv.strip()
                if kv.startswith('transcript_id '):
                    tid = kv.split('"')[1]
                elif kv.startswith('gene_id '):
                    gid = kv.split('"')[1]
                elif kv.startswith('gene_name '):
                    gname = kv.split('"')[1]
            if tid:
                tx2gene[tid] = (gid, gname or gid)
    print(f"{len(tx2gene)} transcripts in GTF", file=sys.stderr)

    genes = {}   # (gene_id, chrom, strand) -> dict
    n_tx = n_missing = 0
    with open(tx_ann) as f:
        hdr = f.readline().rstrip('\n').split('\t')
        col = {c: i for i, c in enumerate(hdr)}
        for line in f:
            p = line.rstrip('\n').split('\t')
            n_tx += 1
            tid = p[col['#NAME']]
            g = tx2gene.get(tid) or tx2gene.get(tid.split('.')[0])
            if g is None:
                # versioned ids may differ between GTF and annotation; try unversioned match
                n_missing += 1
                continue
            key = (g[0], p[col['CHROM']], p[col['STRAND']])
            d = genes.setdefault(key, dict(name=g[1], start=int(p[col['TX_START']]), end=int(p[col['TX_END']]),
                                            es=set(), ee=set()))
            d['start'] = min(d['start'], int(p[col['TX_START']]))
            d['end'] = max(d['end'], int(p[col['TX_END']]))
            d['es'].update(int(x) for x in p[col['EXON_START']].split(',') if x)
            d['ee'].update(int(x) for x in p[col['EXON_END']].split(',') if x)
    if n_missing:
        print(f"WARNING: {n_missing}/{n_tx} transcripts not found in GTF (skipped)", file=sys.stderr)

    rows = sorted(genes.items(), key=lambda kv: (kv[0][1], kv[1]['start']))
    with open(out, 'w') as o:
        o.write("#NAME\tCHROM\tSTRAND\tTX_START\tTX_END\tEXON_START\tEXON_END\tCDS_START\tCDS_END\n")
        for (gid, chrom, strand), d in rows:
            es = ','.join(map(str, sorted(d['es']))) + ','
            ee = ','.join(map(str, sorted(d['ee']))) + ','
            o.write(f"{d['name']}\t{chrom}\t{strand}\t{d['start']}\t{d['end']}\t{es}\t{ee}\t-1\t-1\n")
    print(f"{n_tx} transcripts -> {len(rows)} genes -> {out}", file=sys.stderr)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
