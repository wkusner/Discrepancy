#!/usr/bin/env bash
# Build the C mesh programs capcert (floating point) and capcert_rig (interval arithmetic) (OpenMP if available) and run a quick self-test.
#
# Provenance (PROOF.md Section 6, TRUST.md): the commit, whether csrc/ or build.sh differ from it, the SHA-256 of the
# C sources (the output of `sha256sum` on SRC_FILES, run in this directory, hashed once more) and the compiler flags
# are compiled into both binaries (-D macros); the compiler version comes from __VERSION__. capcert_rig prints them
# with --version and writes them into every completion stamp; check/check_chain.py recomputes the source hash from
# the recorded commit.
set -e
cd "$(dirname "$0")"
CC=${CC:-gcc}
SRC_FILES="csrc/capcert.c csrc/fo.h csrc/ia.h csrc/rig.h csrc/sha256.h"   # capcert.c and the headers it includes
if command -v sha256sum > /dev/null; then SHA="sha256sum"; else SHA="shasum -a 256"; fi
COMMIT=$(git rev-parse HEAD 2> /dev/null || echo unknown)
if [ "$COMMIT" = unknown ]; then DIRTY=unknown
elif [ -n "$(git status --porcelain --untracked-files=all -- csrc build.sh)" ]; then DIRTY=1; else DIRTY=0; fi
SRC_SHA=$($SHA $SRC_FILES | $SHA | cut -c1-64)
SRC_LIST=$(echo $SRC_FILES | sed 's|csrc/||g; s| |,|g')
prov() {   # -D macros for the flags "$*"
  printf '%s\n' "-DCAPCERT_COMMIT=\"$COMMIT\"" "-DCAPCERT_TREE_DIRTY=\"$DIRTY\"" "-DCAPCERT_SRC_FILES=\"$SRC_LIST\"" \
                "-DCAPCERT_SRC_SHA=\"$SRC_SHA\"" "-DCAPCERT_FLAGS=\"$*\""
}
if echo 'int main(void){return 0;}' | $CC -fopenmp -x c - -o /dev/null 2>/dev/null; then
  echo "building capcert and capcert_rig with OpenMP"; OMP="-fopenmp"
else
  echo "OpenMP NOT available: building single-threaded"; OMP=""
fi
FLOAT_FLAGS="-O3 -march=native -Wall $OMP"
RIG_FLAGS="-O3 -march=native -Wall -Wno-unused-function $OMP -DRIGOROUS -ffp-contract=off -fno-fast-math"
IFS=$'\n' read -r -d '' -a PF < <(prov $FLOAT_FLAGS && printf '\0') || true
IFS=$'\n' read -r -d '' -a PR < <(prov $RIG_FLAGS && printf '\0') || true
$CC $FLOAT_FLAGS "${PF[@]}" -o capcert csrc/capcert.c -lm
$CC $RIG_FLAGS "${PR[@]}" -o capcert_rig csrc/capcert.c -lm
./capcert_rig --version
[ "$DIRTY" = 0 ] || echo "WARNING: csrc/ or build.sh differ from commit $COMMIT (tree_dirty=$DIRTY): capcert_rig cannot be used for a proof run"
echo "self-test (expect two FLOAT-CHECKED lines and one INTERVAL-VERIFIED line):"
TMP=$(mktemp -d)
ok=0
./capcert 3 0.499 1e-3 "$TMP/st3" --threads 4 2>/dev/null | tail -1 | tee "$TMP/o1"
./capcert 4 0.374 2.5e-4 "$TMP/st4" --threads 4 2>/dev/null | tail -1 | tee "$TMP/o2"
./capcert_rig 4 0.374 2.5e-4 "$TMP/st5" --threads 4 2>/dev/null | tail -1 | tee "$TMP/o3"
grep -q '^FLOAT-CHECKED' "$TMP/o1" && grep -q '^FLOAT-CHECKED' "$TMP/o2" && grep -q '^INTERVAL-VERIFIED' "$TMP/o3" && ok=1
rm -rf "$TMP"
[ "$ok" = 1 ] || { echo "SELF-TEST FAILED"; exit 1; }
