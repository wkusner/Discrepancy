#!/usr/bin/env bash
# Rebuild bin/capcert_arb-linux-x86_64-static.gz: FLINT 3.0.1 from source as a static library, then a
# fully static capcert_arb (needs gcc, static libgmp/libmpfr, and network access to GitHub).
set -eu
cd "$(dirname "$0")"; W=$(mktemp -d)
curl -sSL -o "$W/flint.tar.gz" https://github.com/flintlib/flint/releases/download/v3.0.1/flint-3.0.1.tar.gz
tar xzf "$W/flint.tar.gz" -C "$W"
( cd "$W/flint-3.0.1" && ./configure --prefix="$W/inst" --disable-shared --enable-static >/dev/null && make -j8 >/dev/null && make install >/dev/null )
gcc -O2 -fopenmp -static -I"$W/inst/include" -o "$W/capcert_arb" csrc/capcert_arb.c "$W/inst/lib/libflint.a" -lmpfr -lgmp -lm -lpthread
strip "$W/capcert_arb"; mkdir -p bin
gzip -9 -c "$W/capcert_arb" > bin/capcert_arb-linux-x86_64-static.gz
( cd "$W" && sha256sum capcert_arb ) > bin/capcert_arb.sha256
echo "rebuilt bin/capcert_arb-linux-x86_64-static.gz"
