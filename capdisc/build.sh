#!/usr/bin/env bash
# Build the C certifier (OpenMP if available) and run a quick self-test.
set -e
cd "$(dirname "$0")"
CC=${CC:-gcc}
if echo 'int main(void){return 0;}' | $CC -fopenmp -x c - -o /dev/null 2>/dev/null; then
  echo "building capcert with OpenMP"; $CC -O3 -march=native -Wall -fopenmp -o capcert csrc/capcert.c -lm
else
  echo "OpenMP NOT available: building single-threaded"; $CC -O3 -march=native -Wall -o capcert csrc/capcert.c -lm
fi
echo "self-test (expect two CERTIFIED lines):"
./capcert 3 0.499 1e-3 /tmp/capcert_selftest3 --threads 4 2>/dev/null | tail -1
./capcert 4 0.374 2.5e-4 /tmp/capcert_selftest4 --threads 4 2>/dev/null | tail -1
rm -f /tmp/capcert_selftest*
