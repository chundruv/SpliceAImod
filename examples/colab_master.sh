#!/usr/bin/env bash
# Launch N Colab GPU sessions from your own terminal, set each one up, start the shard
# driver on each, then (optionally) watch them and release each VM when its driver exits.
#
# Needs the official Colab CLI:  pip install google-colab-cli   (then `colab auth` once)
# and the input BCF (+ .csi) already at  $DRIVE_ROOT/input/  on your Google Drive.
#
# Usage:
#   examples/colab_master.sh launch [N] [GPU]      # default N=1, GPU=A100
#   examples/colab_master.sh add [N] [GPU]         # same, later: adds N more sessions to a running job
#   examples/colab_master.sh bootstrap SESSION     # (re)run setup+launch on an existing session
#   examples/colab_master.sh status                # tail each session's driver log
#   examples/colab_master.sh watch                 # poll; `colab stop` each session once its driver exits
#   examples/colab_master.sh stop                  # stop every session now
#   examples/colab_master.sh sessions              # list the sessions this script knows about
#   examples/colab_master.sh diag SESSION          # processes / GPU / outputs / worker log on one VM
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

# Sessions this script launched are recorded locally (one name per line); the CLI's own
# session listing is not relied on. Override with SESSIONS="spliceai-1 spliceai-2" if needed.
SESSIONS_FILE=${SESSIONS_FILE:-$HOME/.spliceai_colab_sessions}
sessions() {
  if [ -n "${SESSIONS:-}" ]; then echo "$SESSIONS" | tr ' ' '\n'; return; fi
  [ -f "$SESSIONS_FILE" ] && sort -u "$SESSIONS_FILE" || true
}
remember() { grep -qx "$1" "$SESSIONS_FILE" 2>/dev/null || echo "$1" >> "$SESSIONS_FILE"; }
forget()   { [ -f "$SESSIONS_FILE" ] && grep -vx "$1" "$SESSIONS_FILE" > "$SESSIONS_FILE.tmp" && mv "$SESSIONS_FILE.tmp" "$SESSIONS_FILE" || true; }

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
_ = runpy.run_path("/content/SpliceAImod/examples/colab_run.py", run_name="__main__")
del _
PY
}

# `colab exec` occasionally reports "Connection was lost." right after a drivemount or a long
# idle; the kernel is fine and a fresh command reconnects. Retry a few times before giving up.
# `colab exec --timeout` is an OUTPUT-IDLE timeout (default 30 s, undocumented): the CLI gives up
# if the kernel prints nothing for that long, even though the code keeps running on the VM. The
# bootstrap has silent stretches (pip resolving, the reference copy), so it gets a long one.
EXEC_TIMEOUT=${EXEC_TIMEOUT:-1800}
exec_retry() {  # exec_retry SESSION [TIMEOUT]  (code on stdin)
  local S=$1 T=${2:-$EXEC_TIMEOUT} code; code=$(cat)
  for attempt in 1 2 3 4; do
    if printf '%s' "$code" | colab exec -s "$S" --timeout "$T"; then return 0; fi
    echo "== $S: exec failed (attempt $attempt), retrying in 15s"; sleep 15
  done
  return 1
}

driver_alive_py='import subprocess;print("ALIVE" if subprocess.run(["pgrep","-f","python[0-9.]* /content/[c]olab_driver.py"],capture_output=True).returncode==0 else "DEAD")'
tail_py='import glob;l=sorted(glob.glob("/content/drive/MyDrive/spliceai_run/logs/driver_*.log"));print(open(l[-1]).read()[-1500:] if l else "no log")'

cmd=${1:-launch}
case "$cmd" in
  launch|add)
    # Session names continue from the highest index already recorded, so `add 2` after
    # `launch 1` gives spliceai-2 and spliceai-3 and never collides with a running session.
    N=${2:-1}; GPU=${3:-A100}
    last=$(sessions | sed -n "s/^${PREFIX}-\([0-9]*\)$/\1/p" | sort -n | tail -1)
    start=$(( ${last:-0} + 1 ))
    for i in $(seq "$start" $(( start + N - 1 ))); do
      S="${PREFIX}-${i}"
      echo "== $S: provisioning $GPU"
      colab new -s "$S" --gpu "$GPU"
      remember "$S"
      colab drivemount -s "$S"
      sleep 5
      echo "== $S: bootstrapping (install, reference, shards) and launching driver"
      bootstrap_py | exec_retry "$S"
    done
    echo "launched $N session(s). Run: $0 status | $0 watch"
    ;;
  bootstrap)
    S=${2:?session name}
    remember "$S"
    echo "== $S: bootstrapping (install, reference, shards) and launching driver"
    bootstrap_py | exec_retry "$S"
    ;;
  status)
    for S in $(sessions); do
      echo "===== $S"
      echo "$driver_alive_py" | colab exec -s "$S" --timeout 120 2>/dev/null || { echo "(session unreachable — released or never bootstrapped)"; continue; }
      echo "$tail_py" | colab exec -s "$S" --timeout 120 2>/dev/null || true
    done
    ;;
  watch)
    # The driver releases its own VM (COLAB_AUTO_UNASSIGN=1) when it exits; this loop is the
    # belt-and-braces path in case that fails, and gives you a terminal to leave open.
    while :; do
      live=0
      for S in $(sessions); do
        st=$(echo "$driver_alive_py" | colab exec -s "$S" --timeout 120 2>/dev/null | tail -1 || echo GONE)
        echo "$(date +%H:%M) $S $st"
        if [ "$st" = "DEAD" ] || [ "$st" = "GONE" ]; then colab stop -s "$S" || true; forget "$S"; else live=$((live+1)); fi
      done
      [ "$live" -eq 0 ] && { echo "all sessions finished"; break; }
      sleep 600
    done
    ;;
  stop)
    for S in $(sessions); do colab stop -s "$S" || true; forget "$S"; done
    ;;
  sessions)
    sessions
    ;;
  diag)
    # what is actually running on the VM: driver/spliceai processes, GPU, claims, GPU-worker log tail
    S=${2:?session name}
    cat <<'PY' | colab exec -s "$S" --timeout 120
import subprocess, glob
cmd = r'''
echo "--- processes"; ps -eo pid,etime,pcpu,cmd | grep -E "colab_driver|spliceai|batch\.py" | grep -v grep
echo "--- gpu"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "--- drive out/"; ls /content/drive/MyDrive/spliceai_run/out 2>/dev/null | tail -n 20
echo "--- driver log tail"; tail -n 8 $(ls -t /content/drive/MyDrive/spliceai_run/logs/driver_*.log | head -1)
echo "--- gpu worker stderr tail"; tail -n 12 /content/work/tmp/*/GPU_0_w0.stderr 2>/dev/null || echo none
'''
print(subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout)
PY
    ;;
  *) echo "unknown command $cmd"; exit 2;;
esac
