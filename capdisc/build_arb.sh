#!/usr/bin/env bash
# Build capcert_arb (second-kernel mesh stage, Arb ball arithmetic from FLINT) without root:
# use system headers if present, otherwise download libflint-dev/libgmp-dev/libmpfr-dev .debs
# (apt-get download works as an ordinary user) and unpack them into ~/local/deb.
set -u
cd "$(dirname "$0")"
CC=${CC:-gcc}
P=${PREFIX:-$HOME/local}; DEB=$P/deb; mkdir -p "$DEB/pkgs"
try_build() {
  $CC -O2 -Wall -fopenmp "$@" -o capcert_arb csrc/capcert_arb.c -lflint -lgmp -lm 2>build_arb.err
}
if try_build; then echo "capcert_arb built against system FLINT"; else
  echo "system FLINT headers not usable; fetching development packages into $DEB"
  for pkg in libflint-dev libgmp-dev libmpfr-dev $(apt-cache depends libflint-dev 2>/dev/null | awk '/Depends:/{print $2}' | tr -d '<>'); do
    dpkg -s "$pkg" >/dev/null 2>&1 && continue
    ( cd "$DEB/pkgs" && apt-get download "$pkg" >/dev/null 2>&1 ) || echo "  (could not download $pkg)"
  done
  for f in "$DEB"/pkgs/*.deb; do dpkg-deb -x "$f" "$DEB"; done
  L="$DEB/usr/lib/x86_64-linux-gnu"
  try_build -I"$DEB/usr/include" -L"$L" -Wl,-rpath,"$L" || { cat build_arb.err; exit 1; }
  echo "capcert_arb built against FLINT from $DEB"
fi
echo "self-test (expect two ARB-VERIFIED lines):"
./capcert_arb 3 0.499 1e-3 /tmp/capcert_arb_t3 --threads 4 2>/dev/null | tail -1
./capcert_arb 4 0.374 2.5e-4 /tmp/capcert_arb_t4 --threads 4 2>/dev/null | tail -1
rm -f /tmp/capcert_arb_t*
