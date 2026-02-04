#!/usr/bin/env bash
# Build single-file Haskell to WASM
set -euo pipefail

mkdir -p build
cd build

# Copy Aleph.Nix infrastructure (make writable for .hi files)
cp -r "@alephModules@"/Aleph Aleph
chmod -R u+w Aleph

# Copy user's package as Pkg.hs, wrapper as Main.hs
cp "$src" Pkg.hs
cp "@wrapperMain@" Main.hs

"@ghcWasm@"/bin/wasm32-wasi-ghc \
  -optl-mexec-model=reactor \
  '-optl-Wl,--allow-undefined' \
  '-optl-Wl,--export=hs_init' \
  '-optl-Wl,--export=nix_wasm_init_v1' \
  '-optl-Wl,--export=pkg' \
  -O2 \
  Main.hs \
  -o plugin.wasm

"@ghcWasm@"/bin/wasm-opt -O3 plugin.wasm -o "$out"
