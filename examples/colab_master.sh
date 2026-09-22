#!/usr/bin/env bash
# Launch N Colab GPU sessions from your own terminal, set each one up, start the shard
# driver on each, then (optionally) watch them and release each VM when its driver exits.
#
# Needs the official Colab CLI:  pip install google-colab-cli   (then `colab auth` once)
# and the input BCF (+ .csi) already at  $DRIVE_ROOT/input/  on your Google Drive.
#
# Usage:
#   examples/colab_master.sh launch [N] [GPU]      # default N=1, GPU=A100
#   examples/colab_master.sh bootstrap SESSION     # (re)run setup+launch on an existing session
#   examples/colab_master.sh status                # tail each session's driver log
#   examples/colab_master.sh watch                 # poll; `colab stop` each session once its driver exits
#   examples/colab_master.sh stop                  # stop every session now
#
# Everything the on-VM step does lives in examples/colab_run.py; this script only moves it
# there and runs it. Per-run settings are environment variables passed through to it:
#   DRIVE_ROOT INPUT_BCF ANNOTATION DISTANCE VARIANTS_PER_SHARD PRED_BATCH TORCH_BATCH
#   BATCH_WORKERS PRECISION EXTRA_FLAGS REPO_BRANCH
# e.g.  REPO_BRANCH=bench INPUT_BCF=/content/drive/MyDrive/spliceai_run/input/near_splice.bcf \
#       examples/colab_master.sh launch 4
set -euo pipefail

PREFIX=${SESSION_PREFIX:-spliceai}
REPO_URL=${REPO_URL:-https://github.com/chundruv/SpliceAImod.git}
REPO_BRANCH=${REPO_BRANCH:-}
PASS_VARS="DRIVE_ROOT INPUT_BCF REF_ON_DRIVE ANNOTATION DISTANCE VARIANTS_PER_SHARD PRED_BATCH TORCH_BATCH BATCH_WORKERS PRECISION EXTRA_FLAGS REPO_BRANCH"

sessions() { colab ls-sessions 2>/dev/null | grep -o "${PREFIX}-[0-9]*" | sort -u || true; }

# python snippet run on the VM via `colab exec` (stdin): clone repo, export settings, run colab_run.py
bootstrap_py() {
  local envs=""
  for v in $PASS_VARS; do
    if [ -n "${!v:-}" ]; then envs+="os.environ[$(printf '%q' "$v")] = $(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "${!v}")"$'\n'; fi
  done
  cat <<PY
import os, subprocess, runpy
${envs}
os.environ["INSTALL"] = "1"
os.environ["COLAB_AUTO_UNASSIGN"] = "1"
if not os.path.isdir("/content/SpliceAImod"):
    subprocess.run(["git", "clone", "-q", "${REPO_URL}", "/content/SpliceAImod"], check=True)
    if "${REPO_BRANCH}":
        subprocess.run(["git", "-C", "/content/SpliceAImod", "checkout", "-q", "${REPO_BRANCH}"], check=True)
runpy.run_path("/content/SpliceAImod/examples/colab_run.py", run_name="__main__")
PY
}

# `colab exec` occasionally reports "Connection was lost." right after a drivemount or a long
# idle; the kernel is fine and a fresh command reconnects. Retry a few times before giving up.
exec_retry() {  # exec_retry SESSION  (code on stdin)
  local S=$1 code; code=$(cat)
  for attempt in 1 2 3 4; do
    if printf '%s' "$code" | colab exec -s "$S"; then return 0; fi
    echo "== $S: exec failed (attempt $attempt), retrying in 15s"; sleep 15
  done
  return 1
}

driver_alive_py='import subprocess;print("ALIVE" if subprocess.run(["pgrep","-f","python /content/[c]olab_driver.py"],capture_output=True).returncode==0 else "DEAD")'
tail_py='import glob;l=sorted(glob.glob("/content/drive/MyDrive/spliceai_run/logs/driver_*.log"));print(open(l[-1]).read()[-1500:] if l else "no log")'

cmd=${1:-launch}
case "$cmd" in
  launch)
    N=${2:-1}; GPU=${3:-A100}
    for i in $(seq 1 "$N"); do
      S="${PREFIX}-${i}"
      echo "== $S: provisioning $GPU"
      colab new -s "$S" --gpu "$GPU"
      colab drivemount -s "$S"
      sleep 5
      echo "== $S: bootstrapping (install, reference, shards) and launching driver"
      bootstrap_py | exec_retry "$S"
    done
    echo "launched $N session(s). Run: $0 status | $0 watch"
    ;;
  bootstrap)
    S=${2:?session name}
    echo "== $S: bootstrapping (install, reference, shards) and launching driver"
    bootstrap_py | exec_retry "$S"
    ;;
  status)
    for S in $(sessions); do
      echo "===== $S"; echo "$driver_alive_py" | colab exec -s "$S"; echo "$tail_py" | colab exec -s "$S"
    done
    ;;
  watch)
    # The driver releases its own VM (COLAB_AUTO_UNASSIGN=1) when it exits; this loop is the
    # belt-and-braces path in case that fails, and gives you a terminal to leave open.
    while :; do
      live=0
      for S in $(sessions); do
        st=$(echo "$driver_alive_py" | colab exec -s "$S" 2>/dev/null | tail -1 || echo GONE)
        echo "$(date +%H:%M) $S $st"
        if [ "$st" = "DEAD" ]; then colab stop -s "$S" || true; else live=$((live+1)); fi
      done
      [ "$live" -eq 0 ] && { echo "all sessions finished"; break; }
      sleep 600
    done
    ;;
  stop)
    for S in $(sessions); do colab stop -s "$S" || true; done
    ;;
  *) echo "unknown command $cmd"; exit 2;;
esac
