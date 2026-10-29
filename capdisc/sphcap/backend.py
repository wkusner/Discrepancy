"""
sphcap.backend
==============

A tiny "detect what's available and use it" layer so the rest of the
package can request embarrassingly-parallel batched evaluation without
caring whether it ends up on a GPU, on multiple CPU cores, or in a
plain single-threaded NumPy loop.

Design
------
Everything the discrepancy/optimize modules need boils down to one
primitive: given a big batch of "candidates" (rows of some array),
apply a NumPy-shaped function to chunks of it and concatenate the
results. That is exactly the shape of

    * evaluating local discrepancies at many (w, t) candidates,
    * evaluating the objective/gradient at many configurations at once
      (population-based optimization / multi-start).

``Backend.map_chunks`` is the one entry point every other module calls.
It picks, in order of preference:

    1. a CUDA-capable ``torch`` install                -> GPU, batched
    2. ``cupy``                                          -> GPU, batched
    3. plain NumPy + ``multiprocessing`` pool             -> CPU, parallel
    4. plain NumPy, single process                        -> CPU, serial

Nothing above NumPy/SciPy is a hard dependency: if torch/cupy are not
installed, or no GPU is visible, we fall back silently (and say so in
``Backend.describe()``).
"""
from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


def _detect_torch_cuda():
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return torch
    except Exception:
        pass
    return None


def _detect_torch_mps():
    """Apple-silicon GPU via torch's Metal backend."""
    try:
        import torch  # type: ignore
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch
    except Exception:
        pass
    return None


def _detect_cupy():
    try:
        import cupy  # type: ignore
        cupy.cuda.runtime.getDeviceCount()  # raises if no GPU
        return cupy
    except Exception:
        return None


@dataclass
class Backend:
    kind: str                 # "torch-cuda" | "torch-mps" | "cupy" | "mp" | "numpy"
    n_workers: int = 1
    chunk_size: int = 2048

    @classmethod
    def auto(cls, prefer_gpu: bool = True, max_workers: int | None = None) -> "Backend":
        if prefer_gpu:
            if _detect_torch_cuda() is not None:
                return cls(kind="torch-cuda")
            if _detect_torch_mps() is not None:
                return cls(kind="torch-mps")
            if _detect_cupy() is not None:
                return cls(kind="cupy")
        n_cpu = max_workers or os.cpu_count() or 1
        if n_cpu > 1:
            return cls(kind="mp", n_workers=n_cpu)
        return cls(kind="numpy")

    def describe(self) -> str:
        if self.kind == "torch-cuda":
            import torch
            return f"torch/CUDA on {torch.cuda.get_device_name(0)}"
        if self.kind == "torch-mps":
            return "torch/MPS (Apple GPU)"
        if self.kind == "cupy":
            return "cupy/CUDA"
        if self.kind == "mp":
            return f"NumPy + multiprocessing ({self.n_workers} workers)"
        return "NumPy, single process"

    # -- the one primitive everything else uses -----------------------
    def map_chunks(self, fn: Callable[[np.ndarray], np.ndarray],
                    array: np.ndarray, chunk_size: int | None = None) -> np.ndarray:
        """Apply ``fn`` to row-chunks of ``array`` and concatenate.

        ``fn`` must map an (m, ...) array to an (m, ...) array (same
        leading dimension). This is the "embarrassingly parallel"
        primitive: chunk index i is independent of chunk index j.
        """
        cs = chunk_size or self.chunk_size
        chunks = [array[i:i + cs] for i in range(0, len(array), cs)]
        if not chunks:
            return np.empty((0,))

        if self.kind in ("torch-cuda", "torch-mps"):
            import torch
            device = "cuda" if self.kind == "torch-cuda" else "mps"
            out = []
            for c in chunks:
                t = torch.as_tensor(c, dtype=torch.float64, device=device)
                r = fn(t)
                out.append(r.detach().cpu().numpy() if hasattr(r, "detach") else np.asarray(r))
            return np.concatenate(out, axis=0)

        if self.kind == "cupy":
            import cupy as cp
            out = []
            for c in chunks:
                r = fn(cp.asarray(c))
                out.append(cp.asnumpy(r) if hasattr(r, "get") or hasattr(r, "__cuda_array_interface__") else np.asarray(r))
            return np.concatenate(out, axis=0)

        if self.kind == "mp" and len(chunks) > 1:
            # multiprocessing requires a picklable top-level function;
            # we ship the plain-numpy fn and let workers import it.
            with mp.Pool(self.n_workers) as pool:
                out = pool.map(fn, chunks)
            return np.concatenate(out, axis=0)

        # plain numpy, serial
        return np.concatenate([np.asarray(fn(c)) for c in chunks], axis=0)


def get_default_backend(**kwargs) -> Backend:
    """Convenience constructor used throughout the package."""
    return Backend.auto(**kwargs)
