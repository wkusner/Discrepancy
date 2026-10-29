#!/usr/bin/env bash
# User-space environment for the Python jobs on w227 (no root needed).
# Creates ~/sphcap-env with python 3.12, numpy, scipy, sympy, mpmath, python-flint
# and (if conda-forge is reachable) msolve.
#   Route A: micromamba (static binary from GitHub) + conda-forge   -> includes msolve
#   Route B: uv (static binary from GitHub) + PyPI                  -> no msolve
set -u
ENV=$HOME/sphcap-env
BIN=$HOME/.local/bin
mkdir -p "$BIN"
cd /tmp

try_conda() {
  echo "== route A: micromamba + conda-forge"
  wget -q -O "$BIN/micromamba" https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64 || return 1
  chmod +x "$BIN/micromamba"
  export MAMBA_ROOT_PREFIX=$HOME/.micromamba
  "$BIN/micromamba" create -y -q -p "$ENV" -c conda-forge \
      python=3.12 numpy scipy sympy mpmath python-flint msolve || return 1
}

try_uv() {
  echo "== route B: uv + PyPI (msolve will be unavailable)"
  wget -q https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz || return 1
  tar xzf uv-x86_64-unknown-linux-gnu.tar.gz && cp uv-x86_64-unknown-linux-gnu/uv* "$BIN/" || return 1
  rm -rf "$ENV"
  "$BIN/uv" venv -q --python 3.12 "$ENV" || return 1
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
