#!/usr/bin/env bash
# Build a single-file Haskell package to WASM
# Expects: src, alephModules, wrapperMain environment variables
mkdir -p build
cd build
cp -r "$alephModules" Aleph
chmod -R u+w Aleph
cp "$src" Pkg.hs
cp "$wrapperMain" Main.hs
wasm32-wasi-ghc \
	-optl-mexec-model=reactor \
	-optl-Wl,--allow-undefined \
	-optl-Wl,--export=hs_init \
	-optl-Wl,--export=nix_wasm_init_v1 \
	-optl-Wl,--export=pkg \
	-O2 \
	Main.hs \
	-o plugin.wasm
wasm-opt -O3 plugin.wasm -o "$out"
