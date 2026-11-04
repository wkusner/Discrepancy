#!/usr/bin/env bash
# User-space environment for the Python jobs (no root needed).
# Creates ~/sphcap-env with python 3.12, numpy, scipy, sympy, mpmath, python-flint
# and (if conda-forge is reachable) msolve.
#   Route A: micromamba (static binary from GitHub) + conda-forge   -> includes msolve
#   Route B: uv (static binary from GitHub) + PyPI                  -> no msolve
# Linux x86_64 and aarch64, macOS x86_64 and arm64. Settings (environment variables):
#   SPHCAP_ENV      target directory (default ~/sphcap-env)
#   PYTHON_VERSION  Python of the new environment, 3.9 or newer (default 3.12)
set -u
ENV=${SPHCAP_ENV:-$HOME/sphcap-env}
PYV=${PYTHON_VERSION:-3.12}
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)             MM=linux-64;   UVT=x86_64-unknown-linux-gnu ;;
  Linux-aarch64|Linux-arm64) MM=linux-aarch64; UVT=aarch64-unknown-linux-gnu ;;
  Darwin-x86_64)            MM=osx-64;     UVT=x86_64-apple-darwin ;;
  Darwin-arm64)             MM=osx-arm64;  UVT=aarch64-apple-darwin ;;
  *) echo "unsupported platform $(uname -s)-$(uname -m)"; exit 1 ;;
esac
BIN=$HOME/.local/bin
dl() {   # dl FILE URL : curl (always on macOS) or wget
  if command -v curl >/dev/null 2>&1; then curl -fsSL -o "$1" "$2"; else wget -q -O "$1" "$2"; fi
}
mkdir -p "$BIN"
cd /tmp

try_conda() {
  echo "== route A: micromamba + conda-forge"
  dl "$BIN/micromamba" https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-$MM || return 1
  chmod +x "$BIN/micromamba"
  export MAMBA_ROOT_PREFIX=$HOME/.micromamba
  "$BIN/micromamba" create -y -q -p "$ENV" -c conda-forge \
      python=$PYV numpy scipy sympy mpmath python-flint msolve || return 1
}

try_uv() {
  echo "== route B: uv + PyPI (msolve will be unavailable)"
  dl uv-$UVT.tar.gz https://github.com/astral-sh/uv/releases/latest/download/uv-$UVT.tar.gz || return 1
  tar xzf uv-$UVT.tar.gz && cp uv-$UVT/uv* "$BIN/" || return 1
  rm -rf "$ENV"
  "$BIN/uv" venv -q --python "$PYV" "$ENV" || return 1
  "$BIN/uv" pip install -q --python "$ENV/bin/python" numpy scipy sympy mpmath python-flint || return 1
}

if [ -x "$ENV/bin/python" ] && "$ENV/bin/python" -c "import numpy, scipy, sympy, mpmath, flint" 2>/dev/null; then
  echo "environment already present"
else
  try_conda || { echo "route A failed"; try_uv || { echo "route B failed too"; exit 1; }; }
fi

echo "== check"
"$ENV/bin/python" -c "import numpy, scipy, sympy, mpmath, flint; print('python ok:', numpy.__version__, scipy.__version__, sympy.__version__, flint.__version__)"
if [ -x "$ENV/bin/msolve" ]; then echo "msolve ok: $ENV/bin/msolve"; else echo "msolve: not available"; fi
echo "use:  $ENV/bin/python jobs/<script>.py ..."
