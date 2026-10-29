`capcert_arb-linux-x86_64-static.gz`: a statically linked build of `csrc/capcert_arb.c` (FLINT 3.0.1 built
from source as a static library, GMP/MPFR/libgomp static), for machines without FLINT development
files. `build_arb.sh` uses it only if compiling from source fails, and checks the SHA-256 in
`capcert_arb.sha256` and the N=3, N=4 self-tests first. Rebuild with `bash build_arb_static.sh`.
