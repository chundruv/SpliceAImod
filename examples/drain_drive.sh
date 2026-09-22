#!/usr/bin/env bash
# Move finished shard results off Google Drive as they land, so Drive only ever holds the
# few shards in flight. Run on your own machine and leave it running (needs rclone with a
# Drive remote: `rclone config` once, pick "drive").
#
#   examples/drain_drive.sh gdrive:spliceai_run/out /path/to/local/out [interval_s]
#
# Only moves shard_*.tsv.gz whose shard_*.done marker exists (i.e. fully written), never the
# .partial / .claim / .done files themselves — the drivers keep treating the shard as finished
# because they check the .done marker. Merge afterwards with examples/merge_shard_tsvs.py.
set -euo pipefail
REMOTE=${1:?rclone remote path, e.g. gdrive:spliceai_run/out}
LOCAL=${2:?local destination dir}
EVERY=${3:-300}
mkdir -p "$LOCAL"
while :; do
  for done in $(rclone lsf "$REMOTE" --include "shard_*.done" 2>/dev/null); do
    tsv="${done%.done}.tsv.gz"
    if rclone lsf "$REMOTE" --include "$tsv" 2>/dev/null | grep -qx "$tsv"; then
      echo "$(date +%H:%M) moving $tsv"
      rclone moveto "$REMOTE/$tsv" "$LOCAL/$tsv" --checksum && echo "$(date +%H:%M) moved  $tsv"
    fi
  done
  echo "$(date +%H:%M) local: $(ls "$LOCAL"/shard_*.tsv.gz 2>/dev/null | wc -l | tr -d ' ') shard results; sleeping ${EVERY}s"
  sleep "$EVERY"
done
