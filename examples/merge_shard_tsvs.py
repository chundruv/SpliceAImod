#!/usr/bin/env python3
"""Concatenate per-shard SpliceAImod TSVs (as downloaded from Drive's out/ folder) into one
gzip TSV with a single header block, in manifest order. Pure streaming; no dependencies.

Usage:  python examples/merge_shard_tsvs.py <out_dir_with_shard_*.tsv.gz> <merged.tsv.gz> [shards.json]

If shards.json is given, files are taken in manifest order and any missing shard aborts the
merge; otherwise shard_*.tsv.gz are taken in name order.
"""
import glob, gzip, json, os, sys


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    if len(sys.argv) > 3:
        shards = json.load(open(sys.argv[3]))
        files = [os.path.join(src, s["id"] + ".tsv.gz") for s in shards]
        missing = [f for f in files if not os.path.exists(f)]
        if missing:
            sys.exit(f"{len(missing)} shard result(s) missing, e.g. {missing[0]}")
    else:
        files = sorted(glob.glob(os.path.join(src, "shard_*.tsv.gz")))
    if any(glob.glob(os.path.join(src, "*.partial"))):
        sys.exit("a .partial file is present in the output dir: a shard is still being written")
    tmp = dst + ".partial"
    with gzip.open(tmp, "wt", compresslevel=4) as w:
        for i, f in enumerate(files):
            with gzip.open(f, "rt") as r:
                for line in r:
                    if line.startswith("#"):
                        if i == 0: w.write(line)
                        continue
                    w.write(line)
            print(f"{os.path.basename(f)} merged", flush=True)
    os.replace(tmp, dst)
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
