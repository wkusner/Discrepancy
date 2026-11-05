#!/usr/bin/env bash
# make_sidecar.sh DIR P [--force]
#
# Writes DIR/P.surv.sha256, the sidecar that binds the survivor files of a mesh stamp written before 2026-11-04 (no
# survivor hashes in the stamp) to that stamp, in the form check/check_chain.py accepts (PROOF.md, Section 3a):
#
#   # stamp P.done sha256 <SHA-256 of DIR/P.done>  hashed <UTC time>
#   <SHA-256>  P.surv.<t>          one line per survivor file, t ascending
#
# Exactly the files P.surv.<digits> are listed (never the sidecar itself, no temporaries, no other pass), by file name
# only. Run it right after the pass has written P.done and before anything reads or copies the files. It refuses to
# run without P.done and refuses to replace an existing sidecar unless --force is given, because the hashing time is
# the point of the file.
set -euo pipefail

usage() { echo "usage: $0 DIR P [--force]   (P is the prefix of the pass: mesh or mesh2)" >&2; exit 2; }
[ $# -ge 2 ] || usage
DIR=$1; P=$2; FORCE=0
shift 2
for a in "$@"; do case "$a" in --force) FORCE=1 ;; *) usage ;; esac; done
case "$P" in */*|"") usage ;; esac

if command -v sha256sum > /dev/null 2>&1; then SHA="sha256sum"; else SHA="shasum -a 256"; fi
hash_of() { $SHA "$1" | cut -c1-64; }

[ -d "$DIR" ] || { echo "$DIR: not a directory" >&2; exit 1; }
cd "$DIR"
[ -f "$P.done" ] || { echo "$DIR/$P.done is missing: the pass has not completed, nothing to bind" >&2; exit 1; }
OUT="$P.surv.sha256"
if [ -e "$OUT" ] && [ "$FORCE" != 1 ]; then
    echo "$DIR/$OUT exists; it records the files as they were when it was written. Use --force to replace it." >&2; exit 1
fi

# the survivor files of this pass: P.surv.<digits>, in numeric order
FILES=$(ls -1 | grep -E "^$(printf '%s' "$P" | sed 's/[.[\*^$]/\\&/g')\.surv\.[0-9]+$" | sort -t. -k3,3n || true)
STAMP=$(hash_of "$P.done")
TMP="$OUT.tmp.$$"
{
    echo "# stamp $P.done sha256 $STAMP  hashed $(date -u +%FT%TZ)"
    for f in $FILES; do echo "$(hash_of "$f")  $f"; done
} > "$TMP"
mv -f "$TMP" "$OUT"
n=$(printf '%s\n' "$FILES" | grep -c . || true)
echo "$DIR/$OUT: $n survivor file(s) of $P bound to $P.done ($STAMP)"
