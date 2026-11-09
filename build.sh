#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
cc=${CC:-cc}
cxx=${CXX:-g++}
objects=()
for name in lbfgsb linesearch subalgorithms linpack miniCBLAS print timer; do
  "$cc" -O2 -fPIC -fvisibility=hidden -Ithird_party/lbfgsb -c "third_party/lbfgsb/$name.c" -o "build/$name.o"
  objects+=("$PWD/build/$name.o")
done
"$cxx" -O2 -std=c++17 -pthread tests/core_probe.cpp src/KombiOptimizer.cpp "${objects[@]}" -o build/core_probe
export PLUMED_MKLIB_LDFLAGS="${objects[*]} -pthread ${PLUMED_MKLIB_LDFLAGS:-}"
export PLUMED_MKLIB_CFLAGS="-std=c++17 -pthread ${PLUMED_MKLIB_CFLAGS:-}"
cd src
"${PLUMED:-plumed}" mklib -o Kombi.so Kombi.cpp KombiOptimizer.cpp
mv Kombi.so ../build/Kombi.so
