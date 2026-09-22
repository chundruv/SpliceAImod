#!/usr/bin/env python3
"""Cut one or more VCFs into shards of ~N records and write the manifest — locally, with
no htslib dependency. The shard set is then copied to Drive once and every Colab session
consumes it (examples/colab_run.py with SHARDS_DIR set), so no session needs to scan or
region-query the full input.

Shards never split a position (all records at one POS stay together) and never span
chromosomes. Each shard is a self-contained gzip VCF with the input's header.

Usage:
  python examples/shard_vcf_local.py --out shards/ --per-shard 1500000 \\
      near_splice_1kb.snvs.vcf.gz near_splice_1kb.indels.vcf.gz

Writes  shards/shard_0000.vcf.gz ...  and  shards/shards.json  (the manifest the driver reads,
with 'file', 'chrom', 'start', 'end', 'n' per shard).

Merging results afterwards:  python examples/merge_shard_tsvs.py out/ merged.tsv.gz
"""
import argparse, gzip, json, os, sys


def opener(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-shard", type=int, default=1_500_000)
    ap.add_argument("--prefix", default="shard")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    shards, idx = [], 0
    w = None; cur = None; n = 0; first_pos = last_pos = None; header = None

    def close():
        nonlocal w, n
        if w is None:
            return
        w.close()
        shards.append({"id": f"{a.prefix}_{idx:04d}", "file": f"{a.prefix}_{idx:04d}.vcf.gz",
                       "chrom": cur, "start": first_pos, "end": last_pos, "n": n})
        print(f"{shards[-1]['id']}: {cur}:{first_pos}-{last_pos}  {n:,} records", flush=True)
        w = None

    def open_new():
        nonlocal w, idx, n, first_pos
        w = gzip.open(os.path.join(a.out, f"{a.prefix}_{idx:04d}.vcf.gz"), "wt", compresslevel=3)
        w.write(header); n = 0; first_pos = None

    for path in a.inputs:
        with opener(path) as f:
            hdr_lines = []
            for line in f:
                if line.startswith("#"):
                    hdr_lines.append(line); continue
                if header is None:
                    header = "".join(hdr_lines)
                chrom, pos, _ = line.split("\t", 2); pos = int(pos)
                if cur != chrom or (w is not None and n >= a.per_shard and pos != last_pos):
                    if w is not None:
                        close(); idx += 1
                    cur = chrom
                    open_new()
                w.write(line); n += 1
                if first_pos is None: first_pos = pos
                last_pos = pos
    close()
    json.dump(shards, open(os.path.join(a.out, "shards.json"), "w"), indent=1)
    tot = sum(s["n"] for s in shards)
    print(f"\n{len(shards)} shards, {tot:,} records -> {a.out}/shards.json")


if __name__ == "__main__":
    main()
