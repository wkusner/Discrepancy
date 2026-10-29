#!/usr/bin/env bash
# Get msolve WITHOUT compiling and without root: download the distribution's .deb packages
# (apt-get download works as an ordinary user), unpack them into ~/local/deb, and install a
# wrapper ~/local/bin/msolve.  Falls back to fetching just m4 (needed to build GMP) if the
# distribution has no msolve package.  Exit 0 iff a working msolve is installed.
set -u
cd "$(dirname "$0")"; ROOT=$(pwd)
P=${PREFIX:-$HOME/local}; DEB=$P/deb; mkdir -p "$P/bin" "$DEB/pkgs"
LIBDIRS="$DEB/usr/lib/x86_64-linux-gnu:$DEB/usr/lib"
command -v apt-get >/dev/null || { echo "no apt-get"; exit 2; }

fetch_deb() {   # download a package unless it is already installed system-wide
  local pkg=$1
  dpkg -s "$pkg" >/dev/null 2>&1 && return 0
  ( cd "$DEB/pkgs" && apt-get download "$pkg" >/dev/null 2>&1 ) || { echo "  (could not download $pkg)"; return 1; }
}

if apt-cache show msolve >/dev/null 2>&1; then
  echo "== msolve $(apt-cache policy msolve | awk '/Candidate/{print $2}') available as a package"
  fetch_deb msolve
  for dep in $(apt-cache depends msolve | awk '/Depends:/{print $2}' | tr -d '<>'); do
    fetch_deb "$dep"
    # one more level for the msolve/flint libraries themselves
    for d2 in $(apt-cache depends "$dep" 2>/dev/null | awk '/Depends:/{print $2}' | tr -d '<>'); do fetch_deb "$d2"; done
  done
  for f in "$DEB"/pkgs/*.deb; do [ -f "$f" ] && dpkg-deb -x "$f" "$DEB"; done
  cat > "$P/bin/msolve" <<EOF
#!/bin/sh
LD_LIBRARY_PATH="$LIBDIRS\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}" exec "$DEB/usr/bin/msolve" "\$@"
EOF
  chmod +x "$P/bin/msolve"
  if "$P/bin/msolve" -f "$ROOT/msolve/N5_C2v.ms" -P 1 -o /tmp/msolve_selftest_$$.out >/dev/null 2>&1 \
     && grep -q "173504, 2726960" /tmp/msolve_selftest_$$.out; then
    rm -f /tmp/msolve_selftest_$$.out
    echo "msolve OK (from .deb): reproduces the N=5 degree-12 polynomial ($P/bin/msolve)"; exit 0
  fi
  echo "packaged msolve did not pass the self-test"; rm -f "$P/bin/msolve"
fi
echo "== falling back: fetching m4 so that build_msolve.sh can build GMP"
fetch_deb m4 || true
for f in "$DEB"/pkgs/m4_*.deb; do [ -f "$f" ] && dpkg-deb -x "$f" "$DEB"; done
if [ -x "$DEB/usr/bin/m4" ]; then ln -sf "$DEB/usr/bin/m4" "$P/bin/m4"; echo "m4 installed from .deb"; elif command -v m4 >/dev/null; then echo "system m4 present"; fi
exit 1
