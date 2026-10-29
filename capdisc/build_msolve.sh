#!/usr/bin/env bash
# Build msolve (and GMP, MPFR, FLINT, m4) from source into ~/local -- no root, no autotools.
# msolve comes pre-configured from third_party/ (made with `make dist`), so autoconf/automake
# are not needed.  Takes ~15-25 minutes with 16 jobs.  Log: logs/build_msolve.log
set -eu
cd "$(dirname "$0")"; ROOT=$(pwd)
P=${PREFIX:-$HOME/local}; J=${JOBS:-16}
SRC=$HOME/local-src; mkdir -p "$P" "$SRC" logs
exec > >(tee -a logs/build_msolve.log) 2>&1
export PATH="$P/bin:$PATH" CPPFLAGS="-I$P/include" LDFLAGS="-L$P/lib -Wl,-rpath,$P/lib"
GNUCC="gcc -std=gnu17"          # GMP 6.3 / m4 configure tests break under GCC 15's C23 default

fetch() {  # fetch FILE URL [URL...]
  local f=$1; shift
  [ -s "$SRC/$f" ] && return 0
  for u in "$@"; do python3 - "$u" "$SRC/$f" <<'PY' && return 0
import sys, urllib.request
req = urllib.request.Request(sys.argv[1], headers={"User-Agent": "Mozilla/5.0"})
open(sys.argv[2], "wb").write(urllib.request.urlopen(req, timeout=120).read())
PY
  done
  echo "could not download $f"; return 1
}
step() { echo "=== $(date '+%T') $*"; }

if ! [ -x "$P/bin/m4" ]; then
  step m4
  fetch m4-1.4.20.tar.xz https://ftp.gnu.org/gnu/m4/m4-1.4.20.tar.xz https://ftpmirror.gnu.org/m4/m4-1.4.20.tar.xz
  cd "$SRC" && rm -rf m4-1.4.20 && tar xJf m4-1.4.20.tar.xz && cd m4-1.4.20
  CC="$GNUCC" ./configure -q --prefix="$P" && make -s -j$J && make -s install; cd "$ROOT"
fi
if ! [ -f "$P/include/gmp.h" ]; then
  step gmp
  fetch gmp-6.3.0.tar.xz https://gmplib.org/download/gmp/gmp-6.3.0.tar.xz https://ftp.gnu.org/gnu/gmp/gmp-6.3.0.tar.xz
  cd "$SRC" && rm -rf gmp-6.3.0 && tar xJf gmp-6.3.0.tar.xz && cd gmp-6.3.0
  CC="$GNUCC" ./configure -q --prefix="$P" --enable-shared && make -s -j$J && make -s check -j$J >/dev/null && make -s install; cd "$ROOT"
fi
if ! [ -f "$P/include/mpfr.h" ]; then
  step mpfr
  fetch mpfr-4.2.1.tar.xz https://www.mpfr.org/mpfr-4.2.1/mpfr-4.2.1.tar.xz https://ftp.gnu.org/gnu/mpfr/mpfr-4.2.1.tar.xz
  cd "$SRC" && rm -rf mpfr-4.2.1 && tar xJf mpfr-4.2.1.tar.xz && cd mpfr-4.2.1
  CC="$GNUCC" ./configure -q --prefix="$P" --with-gmp="$P" && make -s -j$J && make -s install; cd "$ROOT"
fi
if ! [ -f "$P/include/flint/flint.h" ]; then
  step flint
  fetch flint-3.1.3.tar.gz https://github.com/flintlib/flint/releases/download/v3.1.3/flint-3.1.3.tar.gz
  cd "$SRC" && rm -rf flint-3.1.3 && tar xzf flint-3.1.3.tar.gz && cd flint-3.1.3
  ./configure --prefix="$P" --with-gmp="$P" --with-mpfr="$P" > /dev/null && make -s -j$J && make -s install; cd "$ROOT"
fi
if ! [ -x "$P/bin/msolve" ]; then
  step msolve
  cd "$SRC" && rm -rf msolve-0.11.0 && tar xzf "$ROOT/third_party/msolve-0.11.0.tar.gz" && cd msolve-0.11.0
  ./configure -q --prefix="$P" && make -s -j$J ACLOCAL=true AUTOMAKE=true AUTOCONF=true AUTOHEADER=true \
    && make -s install ACLOCAL=true AUTOMAKE=true AUTOCONF=true AUTOHEADER=true; cd "$ROOT"
fi
step test
"$P/bin/msolve" -f "$ROOT/msolve/N5_C2v.ms" -P 1 -o /tmp/msolve_selftest.out
grep -q "173504, 2726960" /tmp/msolve_selftest.out && echo "msolve OK: reproduces the N=5 degree-12 polynomial ($P/bin/msolve)" || { echo "msolve self-test FAILED"; exit 1; }
