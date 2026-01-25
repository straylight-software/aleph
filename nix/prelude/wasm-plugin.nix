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
  writeText,
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
  build-wasm-plugin =
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
      module-to-path = mod: builtins.replaceStrings [ "." ] [ "/" ] mod + ".hs";
      all-modules = [ mainModule ] ++ extraModules;
      module-files = map module-to-path all-modules;

      # Generate linker flags to export each function
      # GHC WASM doesn't automatically export foreign export ccall symbols,
      # so we need to explicitly tell the linker to export them.
      export-flags = map (e: "'-optl-Wl,--export=${e}'") exports;
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
        for mod in ${lib.escapeShellArgs module-files}; do
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
          ${lib.concatStringsSep " " export-flags} \
          -O2 \
          ${lib.escapeShellArgs ghcFlags} \
          ${module-to-path mainModule} \
          -o plugin.wasm

        # Optionally optimize with wasm-opt
        ${ghc-wasm}/bin/wasm-opt -O3 plugin.wasm -o "$out"
      '';

  # ──────────────────────────────────────────────────────────────────────────
  #                        // aleph wasm module //
  # ──────────────────────────────────────────────────────────────────────────
  # The compiled typed package definitions. Internal implementation detail.
  #
  aleph-wasm = build-wasm-plugin {
    name = "aleph";
    src = ../../src/tools/scripts;
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
  action-to-shell =
    action:
    if action.action == "writeFile" then
      # WriteFile path content → write content to $out/path
      # Use writeText to avoid heredocs entirely
      let
        content-hash = builtins.hashString "sha256" action.content;
        content-file = writeText "wasm-content-${builtins.substring 0 8 content-hash}" action.content;
      in
      ''
        mkdir -p "$out/$(dirname '${action.path}')"
        cp ${content-file} "$out/${action.path}"
      ''
    else if action.action == "install" then
      # Install mode src dst → install -m<mode> src $out/dst
      let
        mode-str = toString action.mode;
      in
      ''
        mkdir -p "$out/$(dirname '${action.dst}')"
        install -m${mode-str} '${action.src}' "$out/${action.dst}"
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
        resolve-rpath = p: if lib.hasPrefix "/" p then p else "$out/${p}";
        rpath-str = lib.concatMapStringsSep ":" resolve-rpath action.rpaths;
      in
      ''
        patchelf --set-rpath '${rpath-str}' "$out/${action.path}"
      ''
    else if action.action == "patchelfAddRpath" then
      # PatchElfAddRpath path rpaths → add to rpath
      let
        rpath-str = lib.concatStringsSep ":" action.rpaths;
      in
      ''
        patchelf --add-rpath '${rpath-str}' "$out/${action.path}"
      ''
    else if action.action == "substitute" then
      # Substitute file replacements → substituteInPlace with typed pairs
      let
        replace-args = lib.concatMapStringsSep " " (
          r: "--replace-fail ${lib.escapeShellArg r.from} ${lib.escapeShellArg r.to}"
        ) action.replacements;
      in
      ''
        substituteInPlace '${action.file}' ${replace-args}
      ''
    else if action.action == "wrap" then
      # Wrap program wrapActions → wrapProgram with typed actions
      let
        wrap-arg =
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
        wrap-args = lib.concatMapStringsSep " " wrap-arg action.wrapActions;
      in
      ''
        wrapProgram "$out/${action.program}" ${wrap-args}
      ''
    else if action.action == "run" then
      # Run cmd args → escape hatch, run arbitrary command
      # Don't escape args that look like shell variables ($foo)
      let
        escape-arg = arg: if lib.hasPrefix "$" arg then arg else lib.escapeShellArg arg;
      in
      ''
        ${action.cmd} ${lib.concatMapStringsSep " " escape-arg action.args}
      ''
    else if action.action == "toolRun" then
      # ToolRun pkg args → typed tool invocation
      # The tool is automatically added to nativeBuildInputs by extract-tool-deps
      let
        escape-arg = arg: if lib.hasPrefix "$" arg then arg else lib.escapeShellArg arg;
      in
      ''
        ${action.pkg} ${lib.concatMapStringsSep " " escape-arg action.args}
      ''
    else
      throw "Unknown action type: ${action.action}";

  # Convert a list of actions to a shell script
  actions-to-shell = actions: lib.concatMapStringsSep "\n" action-to-shell actions;

  # ──────────────────────────────────────────────────────────────────────────
  #                        // build-from-spec //
  # ──────────────────────────────────────────────────────────────────────────
  # Convert a package spec (attrset from WASM) to an actual derivation.
  #
  build-from-spec =
    {
      spec,
      pkgs,
      stdenv-fn ? pkgs.stdenv.mkDerivation,
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
      resolve-dep =
        name:
        let
          parts = lib.splitString "." name;
          resolved = lib.foldl' (acc: part: acc.${part} or null) pkgs parts;
        in
        if resolved != null then resolved else throw "Unknown package: ${name}";
      resolve-deps = names: map resolve-dep names;

      # Extract tool dependencies from typed actions (ToolRun)
      extract-tool-deps =
        actions:
        lib.unique (
          lib.concatMap (action: if action.action or "" == "toolRun" then [ action.pkg ] else [ ]) actions
        );

      # Collect all tool deps from all phases
      phases' = spec.phases or { };
      all-actions =
        (phases'.postPatch or [ ])
        ++ (phases'.preConfigure or [ ])
        ++ (phases'.installPhase or [ ])
        ++ (phases'.postInstall or [ ])
        ++ (phases'.postFixup or [ ]);
      tool-deps = extract-tool-deps all-actions;

      deps = spec.deps or { };
      nativeBuildInputs = resolve-deps ((deps.nativeBuildInputs or [ ]) ++ tool-deps);
      buildInputs = resolve-deps (deps.buildInputs or [ ]);
      propagatedBuildInputs = resolve-deps (deps.propagatedBuildInputs or [ ]);
      checkInputs = resolve-deps (deps.checkInputs or [ ]);

      # Build phase based on builder type
      builder = spec.builder or { type = "none"; };

      builder-attrs =
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

      phase-attrs =
        { }
        // (
          if (phases.postPatch or [ ]) != [ ] then { postPatch = actions-to-shell phases.postPatch; } else { }
        )
        // (
          if (phases.preConfigure or [ ]) != [ ] then
            { preConfigure = actions-to-shell phases.preConfigure; }
          else
            { }
        )
        // (
          if (phases.installPhase or [ ]) != [ ] then
            { installPhase = actions-to-shell phases.installPhase; }
          else
            { }
        )
        // (
          if (phases.postInstall or [ ]) != [ ] then
            { postInstall = actions-to-shell phases.postInstall; }
          else
            { }
        )
        // (
          if (phases.postFixup or [ ]) != [ ] then { postFixup = actions-to-shell phases.postFixup; } else { }
        );

      # Environment variables
      env = spec.env or { };

    in
    stdenv-fn (
      {
        inherit (spec) pname;
        inherit (spec) version;
        inherit
          src
          buildInputs
          propagatedBuildInputs
          checkInputs
          ;
        nativeBuildInputs = builder-attrs.nativeBuildInputs or nativeBuildInputs;

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
      // builder-attrs
      // phase-attrs
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
  load-wasm-packages =
    {
      wasm-file,
      pkgs,
      stdenv-fn ? pkgs.stdenv.mkDerivation,
    }:
    let
      # Check for WASM support at call time with a clear error message
      require-wasm =
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
        assert require-wasm;
        let
          # builtins.wasm wasmPath functionName arg
          # Returns the Nix value from the WASM function
          spec = builtins.wasm wasm-file name args;
        in
        build-from-spec { inherit spec pkgs stdenv-fn; };

      # Get raw spec without building (for debugging)
      spec =
        name: args:
        assert require-wasm;
        builtins.wasm wasm-file name args;
    };

in
{
  inherit
    # Feature detection
    features

    # WASM plugin building (requires ghc-wasm-meta)
    build-wasm-plugin

    # The compiled aleph WASM module (internal)
    aleph-wasm

    # WASM plugin loading (requires straylight-nix with builtins.wasm)
    build-from-spec
    load-wasm-packages

    # Action interpreter - use this to interpret typed phases from any source
    action-to-shell
    actions-to-shell
    ;

  # NOTE: The aleph interface (aleph.eval, aleph.import) is in ./aleph.nix
  # Import it directly:
  #   aleph = import ./prelude/aleph.nix { inherit lib pkgs; wasmFile = ...; };
}
