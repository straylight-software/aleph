# nix/prelude/wasm-plugin.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                              // wasm-plugin //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     The matrix had a different dream. A dream of an artificial island,
#     a floating city of data, where the old distinctions between programs
#     and people, between code and consciousness, were just fading memories.
#
#                                                         — Neuromancer
#
# Build typed Haskell package definitions to WASM modules for use with
# builtins.wasm in straylight-nix.
#
# This bridges the type-safe Haskell world with Nix evaluation.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# FEATURE REQUIREMENTS:
#   - Building WASM plugins: ghc-wasm-meta input (available)
#   - Loading WASM plugins: builtins.wasm (requires straylight-nix)
#
# USAGE:
#   # From Nix, evaluate typed expressions:
#   aleph.eval "Aleph.Packages.Nvidia.nccl" {}
#   aleph.eval "Aleph.Build.withFlags" { pkg = myPkg; flags = ["-O3"]; }
#
#   # Import a module as an attrset:
#   nvidia = aleph.import "Aleph.Packages.Nvidia"
#   nvidia.nccl  # → derivation
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  lib,
  stdenv,
  ghc-wasm-meta,
  runCommand,
}:
let
  # ──────────────────────────────────────────────────────────────────────────
  # Feature Detection
  # ──────────────────────────────────────────────────────────────────────────

  features = {
    # Can we build WASM plugins? (requires ghc-wasm-meta)
    can-build = ghc-wasm-meta != null;

    # Can we load WASM plugins? (requires straylight-nix with builtins.wasm)
    can-load = builtins ? wasm;

    # Human-readable status
    status =
      if features.can-build && features.can-load then
        "Full WASM support available"
      else if features.can-build then
        "Can build WASM plugins, but loading requires straylight-nix"
      else
        "WASM support unavailable (missing ghc-wasm-meta input)";
  };

  # Get the all-in-one GHC WASM bundle (includes ghc, cabal, wasi-sdk, etc.)
  ghc-wasm =
    if features.can-build then ghc-wasm-meta.packages.${stdenv.hostPlatform.system}.all_9_12 else null;

  # ──────────────────────────────────────────────────────────────────────────
  #                        // build-wasm-plugin //
  # ──────────────────────────────────────────────────────────────────────────
  # Build a WASM plugin from Haskell source files.
  #
  # This is internal infrastructure. Users should use aleph.eval directly.
  #
  buildWasmPlugin =
    {
      name,
      src,
      mainModule,
      extraModules ? [ ],
      ghcFlags ? [ ],
      # List of function names to export (must match foreign export ccall names)
      exports ? [ ],
    }:
    let
      # Convert module names to file paths
      moduleToPath = mod: builtins.replaceStrings [ "." ] [ "/" ] mod + ".hs";
      allModules = [ mainModule ] ++ extraModules;
      moduleFiles = map moduleToPath allModules;

      # Generate linker flags to export each function
      # GHC WASM doesn't automatically export foreign export ccall symbols,
      # so we need to explicitly tell the linker to export them.
      exportFlags = map (e: "'-optl-Wl,--export=${e}'") exports;
    in
    runCommand "${name}.wasm"
      {
        inherit src;
        nativeBuildInputs = [ ghc-wasm ];
        passthru = {
          inherit name mainModule exports;
        };
      }
      ''
        # Create working directory with sources
        mkdir -p build
        cd build

        # Copy all source files preserving directory structure
        for mod in ${lib.escapeShellArgs moduleFiles}; do
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
        ${ghc-wasm}/bin/wasm32-wasi-ghc \
          -optl-mexec-model=reactor \
          '-optl-Wl,--allow-undefined' \
          '-optl-Wl,--export=hs_init' \
          ${lib.concatStringsSep " " exportFlags} \
          -O2 \
          ${lib.escapeShellArgs ghcFlags} \
          ${moduleToPath mainModule} \
          -o plugin.wasm

        # Optionally optimize with wasm-opt
        ${ghc-wasm}/bin/wasm-opt -O3 plugin.wasm -o "$out"
      '';

  # ──────────────────────────────────────────────────────────────────────────
  #                        // aleph wasm module //
  # ──────────────────────────────────────────────────────────────────────────
  # The compiled typed package definitions. Internal implementation detail.
  #
  alephWasm = buildWasmPlugin {
    name = "aleph";
    src = ../scripts;
    # Use Main module to get proper GHC RTS initialization in reactor mode
    mainModule = "Main";
    extraModules = [
      "Aleph.Nix"
      "Aleph.Nix.FFI"
      "Aleph.Nix.Types"
      "Aleph.Nix.Value"
      "Aleph.Nix.Derivation"
      "Aleph.Nix.Syntax"
      "Aleph.Script.Tools.CMake"
      # Typed build tools
      "Aleph.Nix.Tools"
      "Aleph.Nix.Tools.Jq"
      "Aleph.Nix.Tools.PatchElf"
      "Aleph.Nix.Tools.Install"
      "Aleph.Nix.Tools.Substitute"
      # Packages
      "Aleph.Nix.Packages.ZlibNg"
      "Aleph.Nix.Packages.Fmt"
      "Aleph.Nix.Packages.Mdspan"
      "Aleph.Nix.Packages.Cutlass"
      "Aleph.Nix.Packages.Rapidjson"
      "Aleph.Nix.Packages.NlohmannJson"
      "Aleph.Nix.Packages.Spdlog"
      "Aleph.Nix.Packages.Catch2"
      "Aleph.Nix.Packages.AbseilCpp"
      # NVIDIA SDK
      "Aleph.Nix.Packages.Nvidia"
      # Test packages for typed actions
      "Aleph.Nix.Packages.Jq"
      "Aleph.Nix.Packages.HelloWrapped"
    ];
    # These must match the foreign export ccall names in Main.hs
    exports = [
      "nix_wasm_init_v1"
      "zlib_ng"
      "fmt"
      "mdspan"
      "cutlass"
      "rapidjson"
      "nlohmann_json"
      "spdlog"
      "catch2"
      "abseil_cpp"
      # NVIDIA SDK
      "nvidia_nccl"
      "nvidia_cudnn"
      "nvidia_tensorrt"
      "nvidia_cutensor"
      "nvidia_cusparselt"
      "nvidia_cutlass"
      # Test packages
      "jq"
      "hello_wrapped"
    ];
  };

  # ──────────────────────────────────────────────────────────────────────────
  #                        // action-to-shell //
  # ──────────────────────────────────────────────────────────────────────────
  # Convert a typed Action to shell commands.
  #
  # This is the interpreter that makes typed phases work. Each Action variant
  # becomes a safe, properly-quoted shell command. No string interpolation bugs.
  #
  actionToShell =
    action:
    if action.action == "writeFile" then
      # WriteFile path content → write content to $out/path
      # Use a content-addressed delimiter to prevent injection attacks.
      # The delimiter includes a hash of the content, making it impossible
      # for the content to contain the exact delimiter string.
      let
        contentHash = builtins.hashString "sha256" action.content;
        delimiter = "__WEYL_EOF_${builtins.substring 0 16 contentHash}__";
      in
      ''
        mkdir -p "$out/$(dirname '${action.path}')"
        cat > "$out/${action.path}" << '${delimiter}'
        ${action.content}
        ${delimiter}
      ''
    else if action.action == "install" then
      # Install mode src dst → install -m<mode> src $out/dst
      let
        modeStr = toString action.mode;
      in
      ''
        mkdir -p "$out/$(dirname '${action.dst}')"
        install -m${modeStr} '${action.src}' "$out/${action.dst}"
      ''
    else if action.action == "mkdir" then
      # Mkdir path → mkdir -p $out/path
      ''
        mkdir -p "$out/${action.path}"
      ''
    else if action.action == "symlink" then
      # Symlink target link → ln -s target $out/link
      ''
        mkdir -p "$out/$(dirname '${action.link}')"
        ln -s '${action.target}' "$out/${action.link}"
      ''
    else if action.action == "copy" then
      # Copy src dst → cp -r src $out/dst
      ''
        mkdir -p "$out/$(dirname '${action.dst}')"
        cp -r '${action.src}' "$out/${action.dst}"
      ''
    else if action.action == "remove" then
      # Remove path → rm -rf $out/path
      ''
        rm -rf "$out/${action.path}"
      ''
    else if action.action == "unzip" then
      # Unzip $src to dest directory (for wheel extraction)
      ''
        unzip -q "$src" -d '${action.dest}'
      ''
    else if action.action == "patchelfRpath" then
      # PatchElfRpath path rpaths → set rpath on binary
      # Paths are relative to $out unless they start with /
      let
        resolveRpath = p: if lib.hasPrefix "/" p then p else "$out/${p}";
        rpathStr = lib.concatMapStringsSep ":" resolveRpath action.rpaths;
      in
      ''
        patchelf --set-rpath '${rpathStr}' "$out/${action.path}"
      ''
    else if action.action == "patchelfAddRpath" then
      # PatchElfAddRpath path rpaths → add to rpath
      let
        rpathStr = lib.concatStringsSep ":" action.rpaths;
      in
      ''
        patchelf --add-rpath '${rpathStr}' "$out/${action.path}"
      ''
    else if action.action == "substitute" then
      # Substitute file replacements → substituteInPlace with typed pairs
      let
        replaceArgs = lib.concatMapStringsSep " " (
          r: "--replace-fail ${lib.escapeShellArg r.from} ${lib.escapeShellArg r.to}"
        ) action.replacements;
      in
      ''
        substituteInPlace '${action.file}' ${replaceArgs}
      ''
    else if action.action == "wrap" then
      # Wrap program wrapActions → wrapProgram with typed actions
      let
        wrapArg =
          wa:
          if wa.type == "prefix" then
            "--prefix ${wa.var} : ${lib.escapeShellArg wa.value}"
          else if wa.type == "suffix" then
            "--suffix ${wa.var} : ${lib.escapeShellArg wa.value}"
          else if wa.type == "set" then
            "--set ${wa.var} ${lib.escapeShellArg wa.value}"
          else if wa.type == "setDefault" then
            "--set-default ${wa.var} ${lib.escapeShellArg wa.value}"
          else if wa.type == "unset" then
            "--unset ${wa.var}"
          else if wa.type == "addFlags" then
            "--add-flags ${lib.escapeShellArg wa.flags}"
          else
            throw "Unknown wrap action type: ${wa.type}";
        wrapArgs = lib.concatMapStringsSep " " wrapArg action.wrapActions;
      in
      ''
        wrapProgram "$out/${action.program}" ${wrapArgs}
      ''
    else if action.action == "run" then
      # Run cmd args → escape hatch, run arbitrary command
      # Don't escape args that look like shell variables ($foo)
      let
        escapeArg = arg: if lib.hasPrefix "$" arg then arg else lib.escapeShellArg arg;
      in
      ''
        ${action.cmd} ${lib.concatMapStringsSep " " escapeArg action.args}
      ''
    else if action.action == "toolRun" then
      # ToolRun pkg args → typed tool invocation
      # The tool is automatically added to nativeBuildInputs by extractToolDeps
      let
        escapeArg = arg: if lib.hasPrefix "$" arg then arg else lib.escapeShellArg arg;
      in
      ''
        ${action.pkg} ${lib.concatMapStringsSep " " escapeArg action.args}
      ''
    else
      throw "Unknown action type: ${action.action}";

  # Convert a list of actions to a shell script
  actionsToShell = actions: lib.concatMapStringsSep "\n" actionToShell actions;

  # ──────────────────────────────────────────────────────────────────────────
  #                        // build-from-spec //
  # ──────────────────────────────────────────────────────────────────────────
  # Convert a package spec (attrset from WASM) to an actual derivation.
  #
  buildFromSpec =
    {
      spec,
      pkgs,
      stdenvFn ? pkgs.stdenv.mkDerivation,
    }:
    let
      # Resolve source
      src =
        if spec.src == null then
          null
        else if spec.src.type == "github" then
          pkgs.fetchFromGitHub {
            inherit (spec.src) owner;
            inherit (spec.src) repo;
            inherit (spec.src) rev;
            inherit (spec.src) hash;
          }
        else if spec.src.type == "url" then
          pkgs.fetchurl {
            inherit (spec.src) url;
            inherit (spec.src) hash;
          }
        else if spec.src.type == "store" then
          spec.src.path
        else
          throw "Unknown source type: ${spec.src.type}";

      # Resolve dependencies by name (supports dotted paths like "stdenv.cc.cc.lib")
      resolveDep =
        name:
        let
          parts = lib.splitString "." name;
          resolved = lib.foldl' (acc: part: acc.${part} or null) pkgs parts;
        in
        if resolved != null then resolved else throw "Unknown package: ${name}";
      resolveDeps = names: map resolveDep names;

      # Extract tool dependencies from typed actions (ToolRun)
      extractToolDeps =
        actions:
        lib.unique (
          lib.concatMap (action: if action.action or "" == "toolRun" then [ action.pkg ] else [ ]) actions
        );

      # Collect all tool deps from all phases
      phases' = spec.phases or { };
      allActions =
        (phases'.postPatch or [ ])
        ++ (phases'.preConfigure or [ ])
        ++ (phases'.installPhase or [ ])
        ++ (phases'.postInstall or [ ])
        ++ (phases'.postFixup or [ ]);
      toolDeps = extractToolDeps allActions;

      deps = spec.deps or { };
      nativeBuildInputs = resolveDeps ((deps.nativeBuildInputs or [ ]) ++ toolDeps);
      buildInputs = resolveDeps (deps.buildInputs or [ ]);
      propagatedBuildInputs = resolveDeps (deps.propagatedBuildInputs or [ ]);
      checkInputs = resolveDeps (deps.checkInputs or [ ]);

      # Build phase based on builder type
      builder = spec.builder or { type = "none"; };

      builderAttrs =
        if builder.type == "cmake" then
          {
            nativeBuildInputs = nativeBuildInputs ++ [ pkgs.cmake ];
            cmakeFlags = builder.flags or [ ];
          }
        else if builder.type == "autotools" then
          {
            configureFlags = builder.configureFlags or [ ];
            makeFlags = builder.makeFlags or [ ];
          }
        else if builder.type == "meson" then
          {
            nativeBuildInputs = nativeBuildInputs ++ [
              pkgs.meson
              pkgs.ninja
            ];
            mesonFlags = builder.flags or [ ];
          }
        else if builder.type == "custom" then
          {
            inherit (builder) buildPhase;
            inherit (builder) installPhase;
          }
        else
          { };

      # Typed phases → shell scripts
      phases =
        spec.phases or {
          postPatch = [ ];
          preConfigure = [ ];
          installPhase = [ ];
          postInstall = [ ];
          postFixup = [ ];
        };

      phaseAttrs =
        { }
        // (
          if (phases.postPatch or [ ]) != [ ] then { postPatch = actionsToShell phases.postPatch; } else { }
        )
        // (
          if (phases.preConfigure or [ ]) != [ ] then
            { preConfigure = actionsToShell phases.preConfigure; }
          else
            { }
        )
        // (
          if (phases.installPhase or [ ]) != [ ] then
            { installPhase = actionsToShell phases.installPhase; }
          else
            { }
        )
        // (
          if (phases.postInstall or [ ]) != [ ] then
            { postInstall = actionsToShell phases.postInstall; }
          else
            { }
        )
        // (
          if (phases.postFixup or [ ]) != [ ] then { postFixup = actionsToShell phases.postFixup; } else { }
        );

      # Environment variables
      env = spec.env or { };

    in
    stdenvFn (
      {
        inherit (spec) pname;
        inherit (spec) version;
        inherit
          src
          buildInputs
          propagatedBuildInputs
          checkInputs
          ;
        nativeBuildInputs = builderAttrs.nativeBuildInputs or nativeBuildInputs;

        strictDeps = spec.strictDeps or true;
        doCheck = spec.doCheck or false;
        dontUnpack = spec.dontUnpack or false;

        meta = {
          description = spec.meta.description or "";
          homepage = spec.meta.homepage or null;
          license = lib.licenses.${spec.meta.license or "unfree"} or lib.licenses.unfree;
          platforms = if (spec.meta.platforms or [ ]) == [ ] then lib.platforms.all else spec.meta.platforms;
          mainProgram = spec.meta.mainProgram or null;
        };
      }
      // builderAttrs
      // phaseAttrs
      // env
    );

  # ──────────────────────────────────────────────────────────────────────────
  #                           // load-wasm-packages //
  # ──────────────────────────────────────────────────────────────────────────
  # Load a WASM plugin and convert its package specs to derivations.
  #
  # This is the Nix-side consumer that:
  # 1. Loads the WASM module
  # 2. Calls exported functions to get package specs
  # 3. Resolves dependency names to actual packages
  # 4. Calls the appropriate builder (cmake, autotools, etc.)
  #
  # FEATURE REQUIREMENT: builtins.wasm (straylight-nix)
  #
  loadWasmPackages =
    {
      wasmFile,
      pkgs,
      stdenvFn ? pkgs.stdenv.mkDerivation,
    }:
    let
      # Check for WASM support at call time with a clear error message
      requireWasm =
        if features.can-load then
          true
        else
          throw ''
            ════════════════════════════════════════════════════════════════════════════════
            WASM plugin loading requires straylight-nix with builtins.wasm support.
            ════════════════════════════════════════════════════════════════════════════════

            Current status: ${features.status}

            To enable WASM plugin support:

              1. Install straylight-nix:
                 nix build github:straylight-software/straylight-nix

              2. Use it as your Nix binary:
                 export PATH="$HOME/.nix-profile/bin:$PATH"
                 # or add to your shell configuration

              3. Verify WASM support:
                 nix eval --expr 'builtins ? wasm'
                 # Should return: true

            Until then, you can:
              - Use the Haskell types as documentation
              - Write manual Nix package definitions based on the type signatures
              - Build WASM plugins (just not load them)

            See: https://github.com/straylight-software/straylight-nix
            ════════════════════════════════════════════════════════════════════════════════
          '';
    in
    {
      # Feature info for introspection
      inherit features;

      # Call a package function from the WASM plugin
      # Usage: call "nvidia_nccl" {}
      # Returns: derivation
      call =
        name: args:
        assert requireWasm;
        let
          # builtins.wasm wasmPath functionName arg
          # Returns the Nix value from the WASM function
          spec = builtins.wasm wasmFile name args;
        in
        buildFromSpec { inherit spec pkgs stdenvFn; };

      # Get raw spec without building (for debugging)
      spec =
        name: args:
        assert requireWasm;
        builtins.wasm wasmFile name args;
    };

  # ──────────────────────────────────────────────────────────────────────────
  #                           // zero-bash-builder //
  # ──────────────────────────────────────────────────────────────────────────
  # Build a derivation using the zero-bash architecture (RFC-007).
  #
  # Instead of converting actions to shell strings, this creates a derivation
  # that uses aleph-exec as the builder. aleph-exec reads the spec directly
  # and executes actions via Haskell I/O.
  #
  # This is opt-in. Set `zeroBash = true` in the spec to enable.
  #
  # FEATURE REQUIREMENT: aleph-exec must be built and available
  #
  buildFromSpecZeroBash =
    {
      spec,
      pkgs,
      aleph-exec,
    }:
    let
      typedBuilder = import ./typed-builder.nix {
        inherit lib;
        inherit (pkgs)
          writeText
          runCommand
          stdenv
          fetchFromGitHub
          fetchurl
          ;
        inherit aleph-exec;
      };
    in
    typedBuilder.buildTypedDerivation { inherit spec pkgs; };

in
{
  inherit
    # Feature detection
    features

    # WASM plugin building (requires ghc-wasm-meta)
    buildWasmPlugin

    # The compiled aleph WASM module (internal)
    alephWasm

    # WASM plugin loading (requires straylight-nix with builtins.wasm)
    buildFromSpec
    loadWasmPackages

    # Zero-bash builder (RFC-007)
    # Use this instead of buildFromSpec when you have aleph-exec available
    buildFromSpecZeroBash

    # Action interpreter - use this to interpret typed phases from any source
    # DEPRECATED: Use buildFromSpecZeroBash instead for new code
    actionToShell
    actionsToShell
    ;

  # NOTE: The aleph interface (aleph.eval, aleph.import) is in ./aleph.nix
  # Import it directly:
  #   aleph = import ./prelude/aleph.nix { inherit lib pkgs; wasmFile = ...; };
}
