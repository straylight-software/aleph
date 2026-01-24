-- nix/prelude/scripts/wasm-build-plugin.dhall
--
-- Build script for WASM plugins
-- Environment variables are injected by render.dhall-with-vars

let moduleFiles : Text = env:MODULE_FILES as Text
let ghcWasm : Text = env:GHC_WASM as Text
let exportFlags : Text = env:EXPORT_FLAGS as Text
let ghcFlags : Text = env:GHC_FLAGS as Text
let mainModulePath : Text = env:MAIN_MODULE_PATH as Text

in ''
# Create working directory with sources
mkdir -p build
cd build

# Copy all source files preserving directory structure
for mod in ${moduleFiles}; do
	mkdir -p "$(dirname "$mod")"
	cp "$src/$mod" "$mod"
done

# Compile to WASM reactor module
# -optl-mexec-model=reactor: WASI reactor model (exports, not _start)
# -optl-Wl,--allow-undefined: Allow undefined symbols (imported from host)
# -optl-Wl,--export=<name>: Export our foreign export ccall functions
# -O2: Optimize
#
# NOTE: We do NOT use -no-hs-main because:
# 1. GHC WASM reactor modules need the RTS initialization code that -no-hs-main excludes
# 2. The _initialize export will call hs_init() when properly linked
# 3. We export hs_init for explicit initialization by the host
${ghcWasm}/bin/wasm32-wasi-ghc \
	-optl-mexec-model=reactor \
	'-optl-Wl,--allow-undefined' \
	'-optl-Wl,--export=hs_init' \
	${exportFlags} \
	-O2 \
	${ghcFlags} \
	${mainModulePath} \
	-o plugin.wasm

# Optionally optimize with wasm-opt
${ghcWasm}/bin/wasm-opt -O3 plugin.wasm -o "$out"
''
