"""SpliceAImod shard driver for Colab (and any box with a mounted shared output dir).

Canonical copy: examples/colab_driver.py. The notebook copies it to /content; colab_run.py does
the same. Reads /content/run_config.json. See examples/colab_sharded_run.ipynb cell 7 for the
claim / merge / shutdown semantics.
"""
import os, sys, json, shutil, subprocess, time, glob, gzip, socket, threading
C = json.load(open("/content/run_config.json"))
STALE_MIN = 30                                  # claim untouched this long = dead session
SESSION = f"{socket.gethostname()}-{os.getpid()}-{int(time.time())}"
DRIVE_OUT = C["DRIVE_OUT"]

def log(msg): print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)

def try_claim(path):
    """Exclusive-create a claim file. Returns True if we own it."""
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, SESSION.encode()); os.close(fd)
        time.sleep(3)                               # let a racing creator surface on Drive
        return open(path).read().strip() == SESSION
    except FileExistsError:
        return False

def claim_is_stale(path):
    try:
        return (time.time() - os.path.getmtime(path)) > STALE_MIN * 60
    except FileNotFoundError:
        return False

def heartbeat(path, stop):
    while not stop.wait(120):
        try: os.utime(path, None)
        except Exception: pass

def fetch_shard(s):
    """Pre-cut shard on Drive -> indexed BCF on local disk (skipped if already there)."""
    sid = s["id"]
    bcf = f"{C['LOCAL_SHARDS']}/{sid}.bcf"
    if os.path.exists(bcf + ".csi"):
        return bcf
    src = os.path.join(C.get("SHARDS_DIR", ""), s.get("file", ""))
    if s.get("file") and os.path.exists(src):
        os.makedirs(C["LOCAL_SHARDS"], exist_ok=True)
        local_vcf = f"{C['LOCAL_SHARDS']}/{s['file']}"
        shutil.copy(src, local_vcf)
        rc = subprocess.run(f"bcftools view -Ob -o {bcf} {local_vcf} && bcftools index {bcf}", shell=True).returncode
        os.remove(local_vcf)
        return bcf if rc == 0 else None
    return None


def run_shard(s):
    sid = s["id"]
    bcf = fetch_shard(s)
    if bcf is None:
        log(f"{sid}: shard BCF missing on local disk and no pre-cut shard on Drive"); return False
    out = f"{C['LOCAL_OUT']}/{sid}.tsv.gz"
    for stale in glob.glob(f"{C['LOCAL_OUT']}/{sid}*"): os.remove(stale)
    cmd = ["spliceai", "-I", bcf, "-O", out, "-R", C["LOCAL_REF"], "-A", C["ANNOTATION"],
           "-D", str(C["DISTANCE"]), "-G", "all", "--precision", C["PRECISION"],
           "-B", str(C["PRED_BATCH"]), "-T", str(C["TORCH_BATCH"]),
           "--batch-workers", str(C["BATCH_WORKERS"]), "-t", C["LOCAL_TMP"], "-V"] + C["EXTRA_FLAGS"].split()
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    log(f"{sid}: start ({s['chrom']}:{s['start']}-{s['end']})")
    t0 = time.time()
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0 or not os.path.exists(out):
        log(f"{sid}: FAILED rc={rc}"); return False
    final = f"{DRIVE_OUT}/{sid}.tsv.gz"
    shutil.move(out, final + ".partial")
    os.replace(final + ".partial", final)
    log(f"{sid}: done in {(time.time()-t0)/3600:.2f} h -> {final}")
    shutil.rmtree(C["LOCAL_TMP"], ignore_errors=True); os.makedirs(C["LOCAL_TMP"], exist_ok=True)
    return True

def merge(shards):
    final = f"{C['DRIVE_ROOT']}/spliceai_all.tsv.gz"
    if os.path.exists(final):
        log("merged file already exists"); return
    if not try_claim(f"{DRIVE_OUT}/merge.claim"):
        log("another session is merging"); return
    log("merging …")
    tmp = f"{C['LOCAL_OUT']}/spliceai_all.tsv.gz"
    with gzip.open(tmp, "wt", compresslevel=4) as w:
        first = True
        for s in shards:
            with gzip.open(f"{DRIVE_OUT}/{s['id']}.tsv.gz", "rt") as r:
                for line in r:
                    if line.startswith("#"):
                        if first: w.write(line)
                        continue
                    w.write(line)
            first = False
    shutil.move(tmp, final + ".partial"); os.replace(final + ".partial", final)
    log(f"merged -> {final}")

shards = json.load(open(C["MANIFEST"]))
log(f"session {SESSION}")
while True:
    todo = [s for s in shards if not os.path.exists(f"{DRIVE_OUT}/{s['id']}.tsv.gz")]
    if not todo:
        log("all shards finished — merge locally with examples/merge_shard_tsvs.py "
            "(or set MERGE_ON_VM=1 to merge here)")
        if os.environ.get("MERGE_ON_VM") == "1":
            merge(shards)
        break
    picked = None
    for s in todo:
        claim = f"{DRIVE_OUT}/{s['id']}.claim"
        if os.path.exists(claim):
            if claim_is_stale(claim):
                log(f"{s['id']}: taking over stale claim"); os.remove(claim)
            else:
                continue
        if try_claim(claim):
            picked = s; break
    if picked is None:
        log(f"{len(todo)} shard(s) still running in other sessions; nothing left to claim here")
        break
    stop = threading.Event()
    hb = threading.Thread(target=heartbeat, args=(f"{DRIVE_OUT}/{picked['id']}.claim", stop), daemon=True); hb.start()
    ok = run_shard(picked)
    stop.set()
    if not ok:
        try: os.remove(f"{DRIVE_OUT}/{picked['id']}.claim")
        except FileNotFoundError: pass
        sys.exit(1)
log("DRIVER EXIT")
if os.environ.get("COLAB_AUTO_UNASSIGN") == "1":
    try:
        from google.colab import runtime
        log("releasing runtime"); runtime.unassign()
    except Exception as e:
        log(f"could not release runtime: {e}")
