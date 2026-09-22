#!/usr/bin/env python3
"""Headless equivalent of cells 1-7 of examples/colab_sharded_run.ipynb.

Runs INSIDE a Colab VM (or any GPU box) and does, in order: install deps, put the
reference FASTA on local disk, shard the input by variant count, write
/content/run_config.json, copy the driver into place and launch it detached.
Idempotent: every step is skipped if its output already exists, so re-running
after a disconnect just relaunches the driver.

Configuration comes from environment variables (so it can be driven by
`colab exec` from a local terminal, see examples/colab_master.sh) with the same
defaults as the notebook's cell 1:

  DRIVE_ROOT            /content/drive/MyDrive/spliceai_run
  INPUT_BCF             $DRIVE_ROOT/input/variants.bcf
  REF_ON_DRIVE          $DRIVE_ROOT/ref/GRCh38.fa
  ANNOTATION            gencodev49            (or MANEv1.4)
  DISTANCE              500
  VARIANTS_PER_SHARD    1500000
  PRED_BATCH            8192
  TORCH_BATCH           256
  BATCH_WORKERS         4
  PRECISION             fp16
  EXTRA_FLAGS           "--compile --conv-impl valid_nhwc"
  SHARDS_DIR            $DRIVE_ROOT/shards   pre-cut shards + shards.json from
                                             examples/shard_vcf_local.py. If present, the input
                                             is never scanned or region-cut on the VM (preferred).
  SHARDS_URL            (none)  HTTPS base URL serving shards.json and shard_*.vcf.gz, e.g. a
                                GitHub release: https://github.com/<user>/<repo>/releases/download/<tag>
                                Used instead of SHARDS_DIR, so Drive holds no shards at all.
  GCS_ROOT              (none)  gs://bucket/prefix. Shards are read from $GCS_ROOT/shards/ and
                                results written to $GCS_ROOT/out/; Drive then holds only the
                                claims / .done markers / manifest / logs. Auth: in a browser
                                notebook run `from google.colab import auth; auth.authenticate_user()`
                                first; headless, put a service-account key on Drive and set
                                GCS_KEY_FILE to its path (roles/storage.objectAdmin on the bucket).
  GCS_KEY_FILE          (none)  service-account JSON for gsutil (see GCS_ROOT)
  CACHE_REF             0/1  copy a freshly downloaded reference back to Drive (default 0)
  REPO_DIR              /content/SpliceAImod  (must already be cloned + pip installed, or
                                               set INSTALL=1 to do it here)
  INSTALL               0/1  run apt/pip installs
  LAUNCH                0/1  launch the driver at the end (default 1)
  COLAB_AUTO_UNASSIGN   0/1  driver releases the VM when it exits (default 1)

Usage on the VM:   python examples/colab_run.py
"""
import os, sys, json, subprocess, shutil, time

E = os.environ.get
DRIVE_ROOT = E("DRIVE_ROOT", "/content/drive/MyDrive/spliceai_run")
CFG = dict(
    DRIVE_ROOT=DRIVE_ROOT,
    INPUT_BCF=E("INPUT_BCF", f"{DRIVE_ROOT}/input/variants.bcf"),
    SHARDS_DIR=E("SHARDS_DIR", f"{DRIVE_ROOT}/shards"),
    SHARDS_URL=E("SHARDS_URL", "").rstrip("/"),
    GCS_ROOT=E("GCS_ROOT", "").rstrip("/"),
    REF_ON_DRIVE=E("REF_ON_DRIVE", f"{DRIVE_ROOT}/ref/GRCh38.fa"),
    REF_URL=E("REF_URL", "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/"
                         "seqs_for_alignment_pipelines.ucsc_ids/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz"),
    REPO_URL=E("REPO_URL", "https://github.com/chundruv/SpliceAImod.git"),
    ANNOTATION=E("ANNOTATION", "gencodev49"),
    DISTANCE=int(E("DISTANCE", "500")),
    VARIANTS_PER_SHARD=int(E("VARIANTS_PER_SHARD", "1500000")),
    PRED_BATCH=int(E("PRED_BATCH", "8192")),
    TORCH_BATCH=int(E("TORCH_BATCH", "256")),
    BATCH_WORKERS=int(E("BATCH_WORKERS", "4")),
    PRECISION=E("PRECISION", "fp16"),
    EXTRA_FLAGS=E("EXTRA_FLAGS", "--compile --conv-impl valid_nhwc"),
    LOCAL=E("LOCAL", "/content/work"),
)
CFG.update(
    LOCAL_REF=f"{CFG['LOCAL']}/ref/GRCh38.fa",
    LOCAL_SHARDS=f"{CFG['LOCAL']}/shards",
    LOCAL_OUT=f"{CFG['LOCAL']}/out",
    LOCAL_TMP=f"{CFG['LOCAL']}/tmp",
    DRIVE_OUT=f"{DRIVE_ROOT}/out",
    DRIVE_LOGS=f"{DRIVE_ROOT}/logs",
    MANIFEST=f"{DRIVE_ROOT}/shards.json",
)
REPO_DIR = E("REPO_DIR", "/content/SpliceAImod")
INSTALL = E("INSTALL", "0") == "1"
LAUNCH = E("LAUNCH", "1") == "1"
AUTO_UNASSIGN = E("COLAB_AUTO_UNASSIGN", "1")


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def sh(cmd, **kw):
    log(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=True, **kw)


def main():
    for d in (f"{CFG['LOCAL']}/ref", CFG["LOCAL_SHARDS"], CFG["LOCAL_OUT"], CFG["LOCAL_TMP"],
              CFG["DRIVE_OUT"], CFG["DRIVE_LOGS"], f"{DRIVE_ROOT}/input", f"{DRIVE_ROOT}/ref"):
        os.makedirs(d, exist_ok=True)
    json.dump(CFG, open("/content/run_config.json", "w"), indent=1)
    log("config written to /content/run_config.json")

    if not os.path.isdir(DRIVE_ROOT):
        sys.exit(f"Drive not mounted: {DRIVE_ROOT}")
    if CFG["GCS_ROOT"]:
        if E("GCS_KEY_FILE"):
            sh(f"gcloud -q auth activate-service-account --key-file {E('GCS_KEY_FILE')}")
        # fail early and clearly if the bucket is not reachable
        sh(f"gsutil -q ls {CFG['GCS_ROOT']}/shards/shards.json")
        if not os.path.exists(CFG["MANIFEST"]):
            sh(f"gsutil -q cp {CFG['GCS_ROOT']}/shards/shards.json {CFG['MANIFEST']}")
    elif CFG["SHARDS_URL"] and not os.path.exists(CFG["MANIFEST"]):
        log(f"fetching manifest from {CFG['SHARDS_URL']}")
        sh(f"wget -q -O {CFG['MANIFEST']}.partial {CFG['SHARDS_URL']}/shards.json && mv {CFG['MANIFEST']}.partial {CFG['MANIFEST']}")
    presharded = bool(CFG["GCS_ROOT"] or CFG["SHARDS_URL"]) or os.path.exists(os.path.join(CFG["SHARDS_DIR"], "shards.json"))
    if not presharded and not os.path.exists(CFG["INPUT_BCF"]):
        sys.exit(f"neither pre-cut shards ({CFG['SHARDS_DIR']}/shards.json) nor input ({CFG['INPUT_BCF']}) found")

    # ---- install --------------------------------------------------------
    if INSTALL or not shutil.which("spliceai"):
        sh("apt-get -qq update && apt-get install -y --no-install-recommends bcftools samtools pigz")
        sh("pip install --progress-bar off pysam pyfaidx pandas numpy intervaltree numba h5py psutil nvidia-ml-py")
        if not os.path.isdir(REPO_DIR):
            sh(f"git clone -q {CFG['REPO_URL']} {REPO_DIR}")
            if E("REPO_BRANCH"):
                sh(f"cd {REPO_DIR} && git checkout -q {E('REPO_BRANCH')}")
        sh(f"cd {REPO_DIR} && pip install --progress-bar off -e .")
    import torch
    log(f"torch {torch.__version__} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'}")

    # ---- reference ------------------------------------------------------
    ref, ref_drive = CFG["LOCAL_REF"], CFG["REF_ON_DRIVE"]
    if not os.path.exists(ref + ".fai"):
        if os.path.exists(ref_drive) and os.path.exists(ref_drive + ".fai"):
            log("copying reference from Drive")
            shutil.copy(ref_drive, ref); shutil.copy(ref_drive + ".fai", ref + ".fai")
        else:
            log("downloading reference from NCBI (~1 GB, 2-3 min)")
            sh(f"wget -q -O - '{CFG['REF_URL']}' | pigz -dc > {ref} && samtools faidx {ref}")
            if E("CACHE_REF", "0") == "1":
                try:
                    shutil.copy(ref, ref_drive); shutil.copy(ref + ".fai", ref_drive + ".fai")
                    log("cached reference on Drive")
                except OSError as e:
                    log(f"could not cache reference on Drive ({e}); continuing")

    # ---- shards ---------------------------------------------------------
    if presharded:
        # Shards were cut locally (examples/shard_vcf_local.py) and copied to Drive once.
        # Only the manifest is needed now; each shard is converted to BCF when claimed.
        if not os.path.exists(CFG["MANIFEST"]):
            shutil.copy(os.path.join(CFG["SHARDS_DIR"], "shards.json"), CFG["MANIFEST"])
        shards = json.load(open(CFG["MANIFEST"]))
        log(f"pre-cut shards: {len(shards)} from {CFG['GCS_ROOT'] or CFG['SHARDS_URL'] or CFG['SHARDS_DIR']}")
        return launch_driver()

    inp = CFG["INPUT_BCF"]
    local_input = f"{CFG['LOCAL']}/input.bcf"
    if not os.path.exists(local_input + ".csi"):
        if inp.endswith(".bcf") and os.path.exists(inp + ".csi"):
            shutil.copy(inp, local_input); shutil.copy(inp + ".csi", local_input + ".csi")
        else:
            # .vcf / .vcf.gz (plain gzip is fine to read) / un-indexed .bcf -> indexed local BCF
            log("converting input to indexed BCF on local disk")
            sh(f"bcftools view -Ob -o {local_input} {inp} && bcftools index {local_input}")

    if os.path.exists(CFG["MANIFEST"]):
        shards = json.load(open(CFG["MANIFEST"]))
        log(f"manifest: {len(shards)} shards")
    else:
        log("scanning positions to build the shard manifest")
        proc = subprocess.Popen(["bcftools", "query", "-f", "%CHROM\t%POS\n", local_input],
                                stdout=subprocess.PIPE, text=True)
        raw, cur, start, count, last = [], None, None, 0, None
        for line in proc.stdout:
            chrom, pos = line.rstrip("\n").split("\t"); pos = int(pos)
            if chrom != cur:
                if cur is not None: raw.append((cur, start, last))
                cur, start, count = chrom, pos, 0
            elif count >= CFG["VARIANTS_PER_SHARD"] and pos != last:
                raw.append((cur, start, last)); start, count = pos, 0
            count += 1; last = pos
        if cur is not None: raw.append((cur, start, last))
        proc.wait()
        shards = [{"id": f"shard_{i:04d}", "chrom": c, "start": s, "end": e} for i, (c, s, e) in enumerate(raw)]
        tmp = CFG["MANIFEST"] + ".partial"
        json.dump(shards, open(tmp, "w"), indent=1); os.replace(tmp, CFG["MANIFEST"])
        log(f"manifest written: {len(shards)} shards")

    todo = [s for s in shards if not (os.path.exists(f"{CFG['DRIVE_OUT']}/{s['id']}.done")
                                       or os.path.exists(f"{CFG['DRIVE_OUT']}/{s['id']}.tsv.gz"))]
    log(f"{len(todo)} shards not yet finished")
    for s in todo:
        bcf = f"{CFG['LOCAL_SHARDS']}/{s['id']}.bcf"
        if os.path.exists(bcf + ".csi"): continue
        sh(f"bcftools view -r {s['chrom']}:{s['start']}-{s['end']} -Ob -o {bcf} {local_input} && bcftools index {bcf}")

    launch_driver()


def launch_driver():
    shutil.copy(os.path.join(REPO_DIR, "examples", "colab_driver.py"), "/content/colab_driver.py")
    if not LAUNCH:
        log("LAUNCH=0: setup complete, driver not started"); return
    if subprocess.run(["pgrep", "-f", "python[0-9.]* /content/[c]olab_driver.py"], capture_output=True).returncode == 0:
        log("driver already running"); return
    logf = f"{CFG['DRIVE_LOGS']}/driver_{time.strftime('%Y%m%d_%H%M%S')}.log"
    env = {**os.environ, "COLAB_AUTO_UNASSIGN": AUTO_UNASSIGN}
    subprocess.Popen(f"nohup python3 /content/colab_driver.py > {logf} 2>&1 &", shell=True, env=env)
    log(f"driver launched; log: {logf}")


if __name__ == "__main__":
    main()
