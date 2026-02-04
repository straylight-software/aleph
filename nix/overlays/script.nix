# nix/overlays/script.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                           // aleph.script //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     He'd operated on an almost permanent adrenaline high, a byproduct of
#     youth and proficiency, jacked into a custom cyberspace deck that
#     projected his disembodied consciousness into the consensual
#     hallucination that was the matrix.
#
#                                                         — Neuromancer
#
# Typed CLI wrapper generation for Aleph.Script.
#
# Transforms --help output from CLI tools into type-safe Haskell wrappers.
# Supports both clap (Rust) and GNU getopt_long formats.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#   aleph.script.gen-wrapper    unified wrapper generator (auto-detects format)
#   aleph.script.check          validation script for all tooling
#   aleph.script.ghc            GHC with Aleph.Script modules
#   aleph.script.tools          pre-generated tool wrappers
#   aleph.script.compiled.*     compiled Haskell scripts for container/VM ops
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
final: prev:
let
  inherit (prev) lib;

  # Source directories for Aleph.Script
  # Library modules (Aleph.*) are in nix/script/lib/
  # The -i flag points to nix/script/lib so GHC finds Aleph/Script.hs etc.
  # Executable scripts are in nix/script/exe/
  # :: Path
  # :: Path
  # :: Path
  aleph-src = ../script/lib;
  script-src = ../script/exe;
  corpus-src = ../../src/tools/corpus;
# :: t15

  # Use GHC 9.12 consistently across the codebase
  # This matches nix/prelude/versions.nix and aligns with Buck2 toolchain
  # :: t16 -> [t38]
  hs-pkgs = final.haskell.packages.ghc912;

  # Haskell dependencies for Aleph.Script
  # These must match SCRIPT_PACKAGES in src/tools/scripts/BUCK
  hs-deps = p: [
    p.megaparsec
    p.text
    p.shelly
    p.foldl
    p.aeson
    p.dhall # Dhall config parsing
    p.directory
    p.filepath
    # For unshare-gpu and typed wrappers
    # Note: dhall brings in crypton, so we use that instead of cryptonite
    # (they have the same Crypto.Hash API)
    p.crypton # SHA256 hashing (same API as cryptonite)
    p.memory # crypton dependency
    p.unordered-containers # HashMap for JSON
    p.vector # Arrays for JSON
    p.unix # executeFile
    p.async # concurrency
    p.bytestring
    p.process
    p.containers
    p.transformers
    p.mtl
    p.time
    # nix-compile.nix: bash parsing and type inference
    # :: t40
    p.ShellCheck # bash AST parser
    p.hnix # Nix expression parser (for store path extraction)
  # :: t41 -> [t44]
  ];

  # GHC with Aleph.Script dependencies
  ghc-with-script = hs-pkgs.ghcWithPackages hs-deps;

  # QuickCheck deps for property tests
  # :: t46
  test-deps =
    p:
    hs-deps p
    ++ [
      p.QuickCheck
      p.deepseq
    ];
  ghc-with-tests = hs-pkgs.ghcWithPackages test-deps;

  # ────────────────────────────────────────────────────────────────────────────
  # // compiled script builder //
  # ────────────────────────────────────────────────────────────────────────────
  #
  # Compiles a Haskell script from src/tools/scripts/ to a binary, wrapping it
  # with runtime dependencies as needed.
  #
  # Build paths:
  #   - Nix:   mkCompiledScript uses ghc912 from nixpkgs (this builder)
  # :: { config-expr : Null, deps : [t48], name : t47 } -> t55
  #   - Buck2: haskell_script in src/tools/scripts/BUCK (via NativeLink)
  #
  # Both use identical GHC version (9.12) and package sets (hsDeps/SCRIPT_PACKAGES).
  # For iteration, use runghc with ghcWithScript in the devshell.
  #
  # If configExpr is provided, it generates a Dhall config file that the
  # :: Bool
  # script reads at startup via CONFIG_FILE environment variable.
# :: t53

  mk-compiled-script =
    {
      # :: Path
      name,
      deps ? [ ], # Runtime dependencies (wrapped into PATH)
      config-expr ? null, # Dhall expression (Nix string with store paths)
    }:
    let
      has-config = config-expr != null;
      # Generate config.dhall as a separate derivation
      config-file = final.writeText "${name}-config.dhall" config-expr;
    in
    final.stdenv.mkDerivation {
      inherit name;
      src = script-src;
      "dontUnpack" = true;

      "nativeBuildInputs" = [
        ghc-with-script
      ]
      ++ lib.optional (deps != [ ] || has-config) final.makeWrapper;

      "buildPhase" = ''
        runHook preBuild
        ghc -O2 -Wall -Wno-unused-imports \
          -hidir . -odir . \
          -i${aleph-src} -i${script-src} \
          -o ${name} ${script-src}/${name}.hs
        runHook postBuild
      '';

      "installPhase" = ''
        runHook preInstall
        mkdir -p $out/bin
        cp ${name} $out/bin/
        ${lib.optionalString has-config ''
          mkdir -p $out/share/aleph
          cp ${config-file} $out/share/aleph/config.dhall
        ''}
        runHook postInstall
      '';
# :: { description : "Compiled Haskell script for container/VM operations" }
# :: "Compiled Haskell script for container/VM operations"

      "postFixup" =
        let
          wrap-args =
            lib.optional (deps != [ ]) "--prefix PATH : ${lib.makeBinPath deps}"
            ++ lib.optional has-config "--set CONFIG_FILE $out/share/aleph/config.dhall";
        in
        lib.optionalString (wrap-args != [ ]) ''
          wrapProgram $out/bin/${name} \
            ${lib.concatStringsSep " \\\n    " wrap-args}
        '';

      meta = {
        description = "Compiled Haskell script for container/VM operations";
      };
    };
# :: Path
# :: Path

  # :: t57
  # :: "nix-compile"
  # ────────────────────────────────────────────────────────────────────────────
  # :: String
  # // nix-compile //
  # ────────────────────────────────────────────────────────────────────────────
  #
  # Type inference for bash scripts at Nix eval time.
  # Uses ShellCheck to parse bash AST, then extracts facts about:
  #   - Variable usage and defaults
  #   - Command invocations and arguments
  #   - Store path references
  #   - config.* structured assignments
  #
  # Source: nix/nix-compile/

  nix-compile-lib = ../nix-compile/lib;
  nix-compile-app = ../nix-compile/app;

  nix-compile-cli = final.writeShellApplication {
    name = "nix-compile";
    "runtimeInputs" = [ ghc-with-script ];
    text = ''
      exec runghc -i${nix-compile-lib} ${nix-compile-app}/nix-compile.hs "$@"
    '';
  };

  nix-compile-compiled = final.stdenv.mkDerivation {
    name = "nix-compile";
    src = ../nix-compile;
    "dontUnpack" = true;
    # :: { nix-compile : { check : t155 -> t159, cli : Path, compiled : t57, parse : t146 -> t154, shell : t162, src : { app : Path, lib : { config-expr : Null, deps : [t48], name : t47 } -> t55 } }, script : { all-compiled : t140, check : t62, compiled : { cloud-hypervisor-gpu : t136, cloud-hypervisor-run : t136, combine-archive : t136, crane-inspect : t136, crane-pull : t136, fhs-run : t136, gpu-run : t136, isospin-build : t136, isospin-run : t136, lint-init : t136, lint-link : t136, nvidia-extract : t136, nvidia-sdk : t136, nvidia-sdk-extract : t136, nvidia-wheel-extract : t136, unshare-gpu : t136, unshare-run : t136, vfio-bind : t136, vfio-list : t136, vfio-unbind : t136 }, corpus : Path, gen-wrapper : t60, ghc : t16 -> [t38], lib : t2, nix-ci : t144, nix-dev : t142, props : t64, shell : t66, src : Path, tools : { all : t69, clap : ["rg"], gnu : ["ls"], handcrafted : ["jq"] } } }
    # :: { all-compiled : t140, check : t62, compiled : { cloud-hypervisor-gpu : t136, cloud-hypervisor-run : t136, combine-archive : t136, crane-inspect : t136, crane-pull : t136, fhs-run : t136, gpu-run : t136, isospin-build : t136, isospin-run : t136, lint-init : t136, lint-link : t136, nvidia-extract : t136, nvidia-sdk : t136, nvidia-sdk-extract : t136, nvidia-wheel-extract : t136, unshare-gpu : t136, unshare-run : t136, vfio-bind : t136, vfio-list : t136, vfio-unbind : t136 }, corpus : Path, gen-wrapper : t60, ghc : t16 -> [t38], lib : t2, nix-ci : t144, nix-dev : t142, props : t64, shell : t66, src : Path, tools : { all : t69, clap : ["rg"], gnu : ["ls"], handcrafted : ["jq"] } }
    "nativeBuildInputs" = [ ghc-with-script ];
    "buildPhase" = ''
      runHook preBuild
      ghc -O2 -Wall -Wno-unused-imports \
        # :: Path
        # :: t2
        # :: Path
        -hidir . -odir . \
        -i${nix-compile-lib} \
        -o nix-compile ${nix-compile-app}/nix-compile.hs
      runHook postBuild
    '';
    "installPhase" = ''
      # :: t16 -> [t38]
      runHook preInstall
      mkdir -p $out/bin
      cp nix-compile $out/bin/
      runHook postInstall
    '';
  };

in
{
  aleph = (prev.aleph or { }) // {
    script = {
      # ──────────────────────────────────────────────────────────────────────
      # // source //
      # :: t60
      # :: "aleph-gen-wrapper"
      # ──────────────────────────────────────────────────────────────────────
# :: String

      src = script-src;
      lib = aleph-src;
      corpus = corpus-src;

      # ──────────────────────────────────────────────────────────────────────
      # // ghc //
      # ──────────────────────────────────────────────────────────────────────

      # GHC with Aleph.Script modules available
      # :: t62
      # :: "aleph-script-check"
      ghc = ghc-with-script;
      # :: String
      inherit ghc-with-tests;

      # ──────────────────────────────────────────────────────────────────────
      # // gen-wrapper //
      # ──────────────────────────────────────────────────────────────────────
      #
      # Unified wrapper generator. Auto-detects clap vs GNU format.
      #
      # Usage:
      #   aleph.script.gen-wrapper rg              # stdout
      # :: t64
      # :: "aleph-script-props"
      #   aleph.script.gen-wrapper grep --gnu     # force GNU format
      # :: String
      #   aleph.script.gen-wrapper fd --write     # write to Tools/Fd.hs

      gen-wrapper = final.writeShellApplication {
        name = "aleph-gen-wrapper";
        "runtimeInputs" = [ ghc-with-script ];
        text = ''
          exec runghc -i${aleph-src} -i${script-src} ${script-src}/gen-wrapper.hs "$@"
        '';
      };

      # :: t66
      # :: "aleph-script-shell"
      # ──────────────────────────────────────────────────────────────────────
      # // check //
      # ──────────────────────────────────────────────────────────────────────
      #
      # Quick validation: compiles all wrappers, parses corpus, checks invariants.

      check = final.writeShellApplication {
        name = "aleph-script-check";
        "runtimeInputs" = [ ghc-with-script ];
        text = ''
          exec runghc -i${aleph-src} -i${script-src} ${script-src}/check.hs "$@"
        '';
      };

      # ──────────────────────────────────────────────────────────────────────
      # // props //
      # ──────────────────────────────────────────────────────────────────────
      #
      # Property tests: parser totality, idempotence, preservation, compilation.

      props = final.writeShellApplication {
        name = "aleph-script-props";
        "runtimeInputs" = [ ghc-with-tests ];
        text = ''
          exec runghc -i${aleph-src} -i${script-src} ${script-src}/Props.hs "$@"
        '';
      };

      # :: { all : t69, clap : ["rg"], gnu : ["ls"], handcrafted : ["jq"] }
      # ──────────────────────────────────────────────────────────────────────
      # :: ["rg"]
      # // shell //
      # ──────────────────────────────────────────────────────────────────────
      #
      # Development shell for working on Aleph.Script.

      shell = final.mkShell {
        name = "aleph-script-shell";
        "buildInputs" = [
          ghc-with-tests
          # CLI tools for testing wrappers
          final.ripgrep
          final.fd
          final.bat
          final.delta
          # :: ["ls"]
          final.dust
          final.tokei
          final.hyperfine
          final.deadnix
          final.statix
        ];
        "shellHook" = ''
          echo "Aleph.Script development shell"
          echo "  runghc -i${aleph-src} -i${script-src} ${script-src}/check.hs"
          echo "  runghc -i${aleph-src} -i${script-src} ${script-src}/Props.hs"
          echo "  runghc -i${aleph-src} -i${script-src} ${script-src}/gen-wrapper.hs <tool>"
        # :: ["jq"]
        '';
      };

      # ──────────────────────────────────────────────────────────────────────
      # // tools //
      # :: t69
      # ──────────────────────────────────────────────────────────────────────
      #
      # Pre-generated tool wrappers (for reference/import).
      # 21 tools: 12 clap + 9 GNU

      tools = {
        # Clap (Rust) tools
        clap = [
          "rg"
          "fd"
          "bat"
          "delta"
          "dust"
          "tokei"
          "hyperfine"
          # :: { cloud-hypervisor-gpu : t136, cloud-hypervisor-run : t136, combine-archive : t136, crane-inspect : t136, crane-pull : t136, fhs-run : t136, gpu-run : t136, isospin-build : t136, isospin-run : t136, lint-init : t136, lint-link : t136, nvidia-extract : t136, nvidia-sdk : t136, nvidia-sdk-extract : t136, nvidia-wheel-extract : t136, unshare-gpu : t136, unshare-run : t136, vfio-bind : t136, vfio-list : t136, vfio-unbind : t136 }
          "deadnix"
          # :: t71
          # :: "vfio-bind"
          # :: [t70]
          "statix"
          "stylua"
          # :: t73
          # :: "vfio-unbind"
          # :: [t72]
          "taplo"
          "zoxide"
        # :: t75
        # :: "vfio-list"
        # :: [t74]
        ];
        # GNU getopt_long tools
        gnu = [
          # :: t78
          # :: "crane-inspect"
          # :: [t77]
          "ls"
          "grep"
          "sed"
          "find"
          "xargs"
          # :: t80
          # :: "crane-pull"
          # :: [t79]
          "tar"
          "gzip"
          "wget"
          # :: t84
          # :: "unshare-run"
          # :: [t83]
          "rsync"
        ];
        # Hand-crafted domain-specific wrappers
        handcrafted = [
          "jq" # JSON processor
          "crane" # OCI image tool
          # :: t89
          # :: "unshare-gpu"
          # :: [t88]
          "bwrap" # bubblewrap sandbox
        ];
        # All tools
        all =
          final.aleph.script.tools.clap
          ++ final.aleph.script.tools.gnu
          ++ final.aleph.script.tools.handcrafted;
      };
# :: t91
# :: "fhs-run"
# :: [t90]

      # ──────────────────────────────────────────────────────────────────────
      # :: t94
      # :: "gpu-run"
      # :: [t93]
      # // compiled //
      # ──────────────────────────────────────────────────────────────────────
      #
      # Compiled Haskell scripts for container/VM operations.
      # These replace the bash scripts in nix/modules/flake/container/.
      #
      # :: t96
      # :: "isospin-run"
      # :: [t95]
      # Each script is compiled to a static binary and wrapped with its
      # runtime dependencies.
# :: t100
# :: "isospin-build"
# :: [t99]

      compiled = {
        # VFIO scripts - PCI device binding for GPU passthrough
        vfio-bind = mk-compiled-script {
          name = "vfio-bind";
          deps = [ final.pciutils ]; # lspci for device info
        };
# :: t102
# :: "cloud-hypervisor-run"
# :: [t101]

        vfio-unbind = mk-compiled-script {
          # :: t105
          # :: "cloud-hypervisor-gpu"
          # :: [t104]
          name = "vfio-unbind";
          deps = [ final.pciutils ];
        };

        vfio-list = mk-compiled-script {
          name = "vfio-list";
          # :: t110
          # :: "nvidia-extract"
          # :: [t109]
          deps = [ final.pciutils ];
        };

        # Crane - OCI image operations (no runtime, just image manipulation)
        crane-inspect = mk-compiled-script {
          name = "crane-inspect";
          deps = [
            final.crane
            final.jq
          # :: t117
          # :: "nvidia-sdk-extract"
          # :: [t116]
          ];
        };

        crane-pull = mk-compiled-script {
          name = "crane-pull";
          deps = [ final.crane ];
        };

        # Unshare - bwrap/namespace runners for OCI images
        unshare-run = mk-compiled-script {
          # :: t122
          # :: "nvidia-wheel-extract"
          # :: [t121]
          name = "unshare-run";
          deps = [
            final.bubblewrap # Container sandbox
            final.crane # OCI image tool
            final.jq # JSON processing
          ];
        };

        unshare-gpu = mk-compiled-script {
          # :: t130
          # :: "nvidia-sdk"
          # :: [t129]
          name = "unshare-gpu";
          deps = [
            final.bubblewrap
            final.crane
            final.jq
            final.pciutils # GPU detection
          ];
        };

        # FHS/GPU scripts - namespace environment wrappers
        fhs-run = mk-compiled-script {
          name = "fhs-run";
          # :: t132
          # :: "combine-archive"
          # :: [t131]
          deps = [ final.bubblewrap ];
        };

        # :: t134
        # :: "lint-init"
        # :: [t133]
        gpu-run = mk-compiled-script {
          name = "gpu-run";
          deps = [
            # :: t136
            # :: "lint-link"
            # :: [t135]
            final.bubblewrap
            final.pciutils
          ];
        };
# :: t140
# :: "aleph-scripts"
# :: [Any]

        # Isospin - Firecracker fork for microVM management
        isospin-run = mk-compiled-script {
          name = "isospin-run";
          deps = [ final.firecracker ]; # TODO: replace with isospin package
        };

        isospin-build = mk-compiled-script {
          name = "isospin-build";
          deps = [
            final.e2fsprogs # mke2fs for rootfs
            final.cpio # initramfs
            final.gzip
          ];
        };
# :: t142
# :: "nix-dev"

        # Cloud Hypervisor - VM management
        cloud-hypervisor-run = mk-compiled-script {
          name = "cloud-hypervisor-run";
          # :: String
          deps = [ final.cloud-hypervisor ];
        };

        cloud-hypervisor-gpu = mk-compiled-script {
          # :: t144
          # :: "nix-ci"
          name = "cloud-hypervisor-gpu";
          deps = [
            final.cloud-hypervisor
            final.pciutils # GPU detection
          # :: String
          ];
        };

        # NVIDIA SDK extraction - pull from NGC, extract CUDA/cuDNN/TensorRT
        nvidia-extract = mk-compiled-script {
          name = "nvidia-extract";
          deps = [
            final.crane # OCI image tool
            final.gnutar # tar extraction
            final.patchelf # ELF RPATH fixing
            final.file # ELF detection
          ];
        };

        # NVIDIA SDK extraction v2 - comprehensive extraction from containers/tarballs
        # Handles CUDA, cuDNN, NCCL, TensorRT, cuTensor, Tritonserver
        nvidia-sdk-extract = mk-compiled-script {
          name = "nvidia-sdk-extract";
          deps = [
            final.crane # OCI image tool
            final.gnutar # tar extraction
            # :: { check : t155 -> t159, cli : Path, compiled : t57, parse : t146 -> t154, shell : t162, src : { app : Path, lib : { config-expr : Null, deps : [t48], name : t47 } -> t55 } }
            final.patchelf # ELF RPATH fixing
            final.file # ELF detection
            final.curl # tarball downloads
            final.findutils # find for patchelf
          ];
        };

        # NVIDIA wheel extraction - extract from PyPI wheels (no redistribution issues)
        nvidia-wheel-extract = mk-compiled-script {
          name = "nvidia-wheel-extract";
          deps = [
            final.curl # download wheels
            final.unzip # extract wheels
            final.patchelf # ELF RPATH fixing
            final.findutils # find for patchelf
          ];
        };

        # NVIDIA SDK - unified extraction (wheels + containers)
        # Typed Haskell replacement for packages.nix shell scripts
        nvidia-sdk = mk-compiled-script {
          name = "nvidia-sdk";
          deps = [
            final.curl # download wheels
            final.unzip # extract wheels
            final.crane # OCI image tool
            final.gnutar # tar extraction
            final.patchelf # ELF RPATH fixing
            final.file # ELF detection
            final.findutils # find for patchelf
          ];
        };

        # combine-archive - Combines multiple .a files into one
        # Used by libmodern overlay for static library aggregation
        combine-archive = mk-compiled-script {
          name = "combine-archive";
          deps = [ ]; # No runtime deps, uses ar from stdenv
        };

        # lint-init - Initialize lint configs in a project
        lint-init = mk-compiled-script {
          name = "lint-init";
          deps = [ ]; # No runtime deps
        };

        # lint-link - Symlink lint configs from aleph
        lint-link = mk-compiled-script {
          name = "lint-link";
          deps = [ ]; # No runtime deps
        };
      };

      # Convenience: build all compiled scripts
      all-compiled = final.symlinkJoin {
        name = "aleph-scripts";
        paths = builtins.attrValues final.aleph.script.compiled;
      };

      # ──────────────────────────────────────────────────────────────────────
      # // nix invocation profiles //
      # ──────────────────────────────────────────────────────────────────────
      #
      # Context-aware Nix wrappers. See RFC-005.
      # Logic in Haskell, thin shell shim for environment setup.
      #
      #   nix-dev   Development (--no-eval-cache, verbose)
      #   nix-ci    CI pipelines (cached, verbose)
      #
      # Usage:
      #   nix-dev build .#foo   # Always re-evaluates, no stale cache

      nix-dev = final.writeShellApplication {
        name = "nix-dev";
        "runtimeInputs" = [
          ghc-with-script
          final.nix
        ];
        text = ''
          exec runghc -i${aleph-src} -i${script-src} ${script-src}/nix-dev.hs "$@"
        '';
      };

      nix-ci = final.writeShellApplication {
        name = "nix-ci";
        "runtimeInputs" = [
          ghc-with-script
          final.nix
        ];
        text = ''
          exec runghc -i${aleph-src} -i${script-src} ${script-src}/nix-ci.hs "$@"
        '';
      };
    };

    # ──────────────────────────────────────────────────────────────────────────
    # // nix-compile //
    # ──────────────────────────────────────────────────────────────────────────
    #
    # Type inference for bash scripts.
    #
    #   aleph.nix-compile.cli         CLI: nix-compile parse|infer|check
    #   aleph.nix-compile.compiled    Compiled binary (faster)
    #   aleph.nix-compile.parse       IFD: script -> Nix attrset schema
    #   aleph.nix-compile.check       Derivation that fails on policy violations
    #
    # Usage:
    #   nix-compile infer ./deploy.sh | jq '.env'
    #   schema = aleph.nix-compile.parse ./deploy.sh;
    #   checks.deploy = aleph.nix-compile.check ./deploy.sh;

    nix-compile =
      let
        inherit (prev) lib;

        # ────────────────────────────────────────────────────────────────────
        # // mkScript //
        # ────────────────────────────────────────────────────────────────────
        #
        # Build a shell script with nix-compile.nix analysis and emit-config.
        #
        # Simple form:
        #   aleph.nix-compile.mkScript "my-script" ''
        #     PORT="''${PORT:-8080}"
        #     config.server.port=$PORT
        #     exec ${pkgs.myapp}/bin/myapp --config <(emit-config)
        #   ''
        #
        # Full form:
        #   aleph.nix-compile.mkScript {
        #     name = "my-script";
        #     script = ''...'';
        #     deps = [ pkgs.jq ];           # Added to PATH
        #     requireStorePaths = true;     # Fail on bare commands (default: true)
        #     injectEmitConfig = true;      # Add emit-config function (default: true)
        #   }

        # :: Path
        mkScript =
          arg1: arg2:
          # :: t57
          let
            # Parse arguments: mkScript "name" "script" or mkScript { ... }
            args =
              if builtins.isString arg1 && builtins.isString arg2 then
                {
                  name = arg1;
                  # :: t146 -> t154
                  script = arg2;
                }
              # :: [Path]
              # :: t151
              else if builtins.isAttrs arg1 then
                arg1
              else
                throw "nix-compile.mkScript: expected (name, script) or { name, script, ... }";

            # :: t155 -> t159
            name = args.name or (throw "nix-compile.mkScript: 'name' is required");
            script = args.script or (throw "nix-compile.mkScript: 'script' is required");
            deps = args.deps or [ ];
            # :: [Path]
            requireStorePaths = args.requireStorePaths or true;
            injectEmitConfig = args.injectEmitConfig or true;

            # Write script to a file for analysis
            scriptFile = final.writeText "${name}-source.sh" script;

            # Generate emit-config function
            emitConfigDrv =
              final.runCommand "${name}-emit-config"
                {
                  # :: { app : Path, lib : { config-expr : Null, deps : [t48], name : t47 } -> t55 }
                  # :: { config-expr : Null, deps : [t48], name : t47 } -> t55
                  # :: Path
                  nativeBuildInputs = [ nix-compile-cli ];
                }
                ''
                  nix-compile emit ${scriptFile} > $out
                '';

            # :: t162
            # :: "nix-compile-shell"
            # :: [t16 -> [t38]]
            # Policy check derivation (bare commands)
            policyCheckDrv =
              final.runCommand "${name}-policy-check"
                {
                  # :: String
                  nativeBuildInputs = [ nix-compile-cli ];
                }
                ''
                  nix-compile check ${scriptFile}
                  touch $out
                '';

            # Parse schema (for passthru)
            parseSchema =
              let
                result =
                  final.runCommand "${name}-schema"
                    {
                      nativeBuildInputs = [ nix-compile-cli ];
                    }
                    ''
                      nix-compile infer ${scriptFile} > $out
                    '';
              in
              builtins.fromJSON (builtins.readFile result);

            # Build the final script with emit-config injected
            finalScript =
              let
                emitConfigContent = if injectEmitConfig then builtins.readFile emitConfigDrv else "";
              in
              ''
                ${emitConfigContent}
                ${script}
              '';

          in
          final.writeShellApplication {
            inherit name;
            runtimeInputs = deps;
            text = finalScript;
            # Exclude ShellCheck warnings for config.* convention
            # SC2276: "This is interpreted as a command name containing '='"
            #         We use config.x.y=$VAR intentionally as a DSL
            # SC2086: "Double quote to prevent globbing and word splitting"
            #         Unquoted vars in config.* are intentional (numeric types)
            excludeShellChecks = [
              "SC2276"
              "SC2086"
            ];
            derivationArgs = {
              passthru = {
                # The analyzed schema (IFD)
                schema = parseSchema;
                # The emit-config function source
                emitConfig = emitConfigDrv;
                # Source script for debugging
                sourceScript = scriptFile;
                # Policy check derivation
                policyCheck = policyCheckDrv;
              };
              # Policy check as build dependency (fails build if violations)
              nativeBuildInputs = lib.optional requireStorePaths policyCheckDrv;
            };
          };

      in
      {
        # ──────────────────────────────────────────────────────────────────────
        # // mkScript //
        # ──────────────────────────────────────────────────────────────────────

        inherit mkScript;

        # ──────────────────────────────────────────────────────────────────────
        # // CLI //
        # ──────────────────────────────────────────────────────────────────────

        # CLI tool (interpreted, fast iteration)
        cli = nix-compile-cli;

        # Compiled binary (for CI)
        compiled = nix-compile-compiled;

        # ──────────────────────────────────────────────────────────────────────
        # // Nix integration //
        # ──────────────────────────────────────────────────────────────────────

        # Parse a script, return schema as Nix attrset (IFD)
        parse =
          scriptPath:
          let
            result = final.runCommand "nix-compile-schema" { nativeBuildInputs = [ nix-compile-cli ]; } ''
              nix-compile infer ${scriptPath} > $out
            '';
          in
          builtins.fromJSON (builtins.readFile result);

        # Check derivation - fails if script has policy violations
        check =
          scriptPath:
          final.runCommand "nix-compile-check-${builtins.baseNameOf scriptPath}"
            {
              nativeBuildInputs = [ nix-compile-cli ];
            }
            ''
              nix-compile check ${scriptPath}
              touch $out
            '';

        # ──────────────────────────────────────────────────────────────────────
        # // Source //
        # ──────────────────────────────────────────────────────────────────────

        src = {
          lib = nix-compile-lib;
          app = nix-compile-app;
        };

        # ──────────────────────────────────────────────────────────────────────
        # // Shell //
        # ──────────────────────────────────────────────────────────────────────

        shell = final.mkShell {
          name = "nix-compile-shell";
          buildInputs = [
            ghc-with-script
            nix-compile-cli
            final.jq
          ];
          shellHook = ''
            echo "nix-compile.nix development shell"
            echo "  nix-compile parse <script>   Show facts"
            echo "  nix-compile infer <script>   Show schema (JSON)"
            echo "  nix-compile check <script>   Check policies"
          '';
        };
      };
  };
}
