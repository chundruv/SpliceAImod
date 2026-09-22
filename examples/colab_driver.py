"""SpliceAImod shard driver.

Canonical copy: examples/colab_driver.py (the notebook and colab_run.py copy it to /content).
Reads /content/run_config.json. Runs shards from a manifest, one at a time, until none are left
to claim, then exits (and releases the Colab runtime if COLAB_AUTO_UNASSIGN=1).

Shared state (claims, .done markers, manifest, logs) and bulk data (shards, results) live in ONE
of two stores, chosen by the config:

  GCS_ROOT set      -> everything in the bucket:  $GCS_ROOT/shards/  $GCS_ROOT/out/
                       $GCS_ROOT/state/{<shard>.claim,<shard>.done,logs/<session>.log}
                       Claims are atomic (object create with if_generation_match=0). No Drive.
  otherwise         -> a shared directory (Google Drive): $DRIVE_OUT/{<shard>.tsv.gz,.claim,.done}
                       Claims are exclusive-create files.

Several sessions can run this against the same store at once. A claim is refreshed every 2 min
while its shard runs; one untouched for STALE_MIN minutes belongs to a dead session and is taken
over, so a lost session costs at most the shard in flight.
"""
import os, sys, json, shutil, subprocess, time, glob, socket, threading

C = json.load(open("/content/run_config.json"))
STALE_MIN = 30
SESSION = f"{socket.gethostname()}-{os.getpid()}-{int(time.time())}"
LOG_PATH = os.environ.get("DRIVER_LOG", "")          # local log file (mirrored to the store)


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


# ----------------------------------------------------------------------------- stores
class DriveStore:
    """Shared directory (Drive mount). Results, claims and markers all live in DRIVE_OUT."""

    def __init__(self, cfg):
        self.out = cfg["DRIVE_OUT"]
        self.shards_dir = cfg.get("SHARDS_DIR", "")
        self.shards_url = cfg.get("SHARDS_URL", "")
        os.makedirs(self.out, exist_ok=True)

    def finished(self, sid):
        return os.path.exists(f"{self.out}/{sid}.done") or os.path.exists(f"{self.out}/{sid}.tsv.gz")

    def try_claim(self, sid):
        p = f"{self.out}/{sid}.claim"
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, SESSION.encode()); os.close(fd)
            time.sleep(3)
            return open(p).read().strip() == SESSION
        except FileExistsError:
            return False

    def claim_state(self, sid):
        p = f"{self.out}/{sid}.claim"
        if not os.path.exists(p):
            return None
        return (time.time() - os.path.getmtime(p)) / 60

    def touch_claim(self, sid):
        try: os.utime(f"{self.out}/{sid}.claim", None)
        except Exception: pass

    def release_claim(self, sid):
        try: os.remove(f"{self.out}/{sid}.claim")
        except FileNotFoundError: pass

    def fetch_shard(self, s, local_vcf):
        if self.shards_url:
            return subprocess.run(f"wget -q -O {local_vcf} {self.shards_url}/{s['file']}", shell=True).returncode == 0
        src = os.path.join(self.shards_dir, s["file"])
        if os.path.exists(src):
            shutil.copy(src, local_vcf); return True
        return False

    def put_result(self, sid, local_path):
        final = f"{self.out}/{sid}.tsv.gz"
        shutil.move(local_path, final + ".partial"); os.replace(final + ".partial", final)
        return final

    def mark_done(self, sid, text):
        with open(f"{self.out}/{sid}.done", "w") as f: f.write(text)

    def sync_log(self):
        pass                                   # the log already lives on Drive


class GCSStore:
    """Everything in a GCS bucket. Uses google-cloud-storage (preinstalled on Colab)."""

    def __init__(self, cfg):
        from google.cloud import storage
        root = cfg["GCS_ROOT"][len("gs://"):]
        self.bucket_name, _, self.prefix = root.partition("/")
        self.prefix = self.prefix.strip("/")
        self.client = storage.Client(project=cfg.get("GCP_PROJECT") or "colab")
        self.bucket = self.client.bucket(self.bucket_name)

    def _p(self, *parts):
        return "/".join(x.strip("/") for x in (self.prefix, *parts) if x)

    def _blob(self, *parts):
        return self.bucket.blob(self._p(*parts))

    def finished(self, sid):
        return self._blob("state", f"{sid}.done").exists() or self._blob("out", f"{sid}.tsv.gz").exists()

    def try_claim(self, sid):
        from google.api_core.exceptions import PreconditionFailed
        try:
            self._blob("state", f"{sid}.claim").upload_from_string(SESSION, if_generation_match=0)
            return True
        except PreconditionFailed:
            return False

    def claim_state(self, sid):
        b = self._blob("state", f"{sid}.claim")
        if not b.exists():
            return None
        b.reload()
        return (time.time() - b.updated.timestamp()) / 60

    def touch_claim(self, sid):
        try: self._blob("state", f"{sid}.claim").upload_from_string(SESSION)   # new generation => new 'updated'
        except Exception: pass

    def release_claim(self, sid):
        try: self._blob("state", f"{sid}.claim").delete()
        except Exception: pass

    def fetch_shard(self, s, local_vcf):
        b = self._blob("shards", s["file"])
        if not b.exists():
            return False
        b.download_to_filename(local_vcf); return True

    def put_result(self, sid, local_path):
        b = self._blob("out", f"{sid}.tsv.gz")
        b.upload_from_filename(local_path, timeout=600)       # single-object upload: atomic
        os.remove(local_path)
        return f"gs://{self.bucket_name}/{b.name}"

    def mark_done(self, sid, text):
        self._blob("state", f"{sid}.done").upload_from_string(text)

    def sync_log(self):
        if LOG_PATH and os.path.exists(LOG_PATH):
            try: self._blob("state", "logs", f"{SESSION}.log").upload_from_filename(LOG_PATH)
            except Exception: pass


store = GCSStore(C) if C.get("GCS_ROOT") else DriveStore(C)


# ----------------------------------------------------------------------------- work
def heartbeat(sid, stop):
    while not stop.wait(120):
        store.touch_claim(sid)
        store.sync_log()


def run_shard(s):
    sid = s["id"]
    os.makedirs(C["LOCAL_SHARDS"], exist_ok=True)
    bcf = f"{C['LOCAL_SHARDS']}/{sid}.bcf"
    if not os.path.exists(bcf + ".csi"):
        local_vcf = f"{C['LOCAL_SHARDS']}/{s['file']}"
        if not store.fetch_shard(s, local_vcf):
            log(f"{sid}: shard file {s.get('file')} not found in the store"); return False
        rc = subprocess.run(f"bcftools view -Ob -o {bcf} {local_vcf} && bcftools index {bcf}", shell=True).returncode
        os.remove(local_vcf)
        if rc != 0:
            log(f"{sid}: bcftools conversion failed"); return False
    out = f"{C['LOCAL_OUT']}/{sid}.tsv.gz"
    for stale in glob.glob(f"{C['LOCAL_OUT']}/{sid}*"): os.remove(stale)
    cmd = ["spliceai", "-I", bcf, "-O", out, "-R", C["LOCAL_REF"], "-A", C["ANNOTATION"],
           "-D", str(C["DISTANCE"]), "-G", "all", "--precision", C["PRECISION"],
           "-B", str(C["PRED_BATCH"]), "-T", str(C["TORCH_BATCH"]),
           "--batch-workers", str(C["BATCH_WORKERS"]), "-t", C["LOCAL_TMP"], "-V"] + C["EXTRA_FLAGS"].split()
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    log(f"{sid}: start ({s['chrom']}:{s['start']}-{s['end']}, {s.get('n', '?')} records)")
    t0 = time.time()
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0 or not os.path.exists(out):
        log(f"{sid}: FAILED rc={rc}"); return False
    size = os.path.getsize(out)
    final = store.put_result(sid, out)
    store.mark_done(sid, f"{SESSION} {time.strftime('%Y-%m-%dT%H:%M:%S')} {size} {final}\n")
    log(f"{sid}: done in {(time.time()-t0)/3600:.2f} h, {size/1e6:.0f} MB -> {final}")
    shutil.rmtree(C["LOCAL_TMP"], ignore_errors=True); os.makedirs(C["LOCAL_TMP"], exist_ok=True)
    return True


shards = json.load(open(C["MANIFEST"]))
log(f"session {SESSION}; store {'GCS ' + C['GCS_ROOT'] if C.get('GCS_ROOT') else 'Drive ' + C['DRIVE_OUT']}")
while True:
    todo = [s for s in shards if not store.finished(s["id"])]
    if not todo:
        log("all shards finished"); break
    picked = None
    for s in todo:
        age = store.claim_state(s["id"])
        if age is not None:
            if age > STALE_MIN:
                log(f"{s['id']}: taking over stale claim ({age:.0f} min)"); store.release_claim(s["id"])
            else:
                continue
        if store.try_claim(s["id"]):
            picked = s; break
    if picked is None:
        log(f"{len(todo)} shard(s) still running in other sessions; nothing left to claim here"); break
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(picked["id"], stop), daemon=True).start()
    ok = run_shard(picked)
    stop.set()
    store.release_claim(picked["id"])
    store.sync_log()
    if not ok:
        sys.exit(1)

log("DRIVER EXIT")
store.sync_log()
if os.environ.get("COLAB_AUTO_UNASSIGN") == "1":
    try:
        from google.colab import runtime
        log("releasing runtime"); runtime.unassign()
    except Exception as e:
        log(f"could not release runtime: {e}")
