#!/usr/bin/env bash
# Push STATUS.md and the small result files to the branch w227-status of the GitHub repo.
# One-time setup on w227 (see README):  a deploy key with write access, or a token.
#   STATUS_REMOTE  (default git@github.com:wkusner/Discrepancy.git)
#   STATUS_DIR     (default ~/capdisc-status: a separate clone used only for pushing)
set -u
cd "$(dirname "$0")"
REMOTE=${STATUS_REMOTE:-git@github.com:wkusner/Discrepancy.git}
DIR=${STATUS_DIR:-$HOME/capdisc-status}
BR=w227-status
if [ ! -d "$DIR/.git" ]; then
  git init -q "$DIR" && git -C "$DIR" remote add origin "$REMOTE"
  if git -C "$DIR" fetch -q --depth 1 origin $BR 2>/dev/null; then
    git -C "$DIR" checkout -q -b $BR FETCH_HEAD
  else
    git -C "$DIR" checkout -q --orphan $BR
  fi
fi
mkdir -p "$DIR/results" "$DIR/logs"
cp STATUS.md "$DIR/"
# small text results only (no survivor binaries, no progress files)
for f in results/*.out results/*.json results/*.txt; do [ -f "$f" ] && [ $(stat -c %s "$f") -lt 20000000 ] && cp "$f" "$DIR/results/"; done
for f in logs/*.log; do tail -n 200 "$f" > "$DIR/logs/$(basename $f)"; done
cd "$DIR"
git add -A
git -c user.name="w227 capdisc" -c user.email="woden@w227" commit -q -m "status $(date '+%F %T')" || exit 0
git push -q origin HEAD:$BR && echo "$(date '+%F %T') pushed" || echo "$(date '+%F %T') push FAILED"
