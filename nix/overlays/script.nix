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
  # Library modules (Aleph.*) are in src/haskell/Aleph/
  # The -i flag should point to src/haskell so GHC finds Aleph/Script.hs etc.
  # Executable scripts are in src/tools/scripts/
  aleph-src = ../../src/haskell;
  script-src = ../../src/tools/scripts;
  corpus-src = ../../src/tools/corpus;

  # Use GHC 9.12 consistently across the codebase
  # This matches nix/prelude/versions.nix and aligns with Buck2 toolchain
  hs-pkgs = final.haskell.packages.ghc912;

  # Haskell dependencies for Aleph.Script
  # These must match SCRIPT_PACKAGES in src/tools/scripts/BUCK
  hs-deps =
    p: with p; [
      megaparsec
      text
      shelly
      foldl
      aeson
      dhall # Dhall config parsing
      directory
      filepath
      # For unshare-gpu and typed wrappers
      # Note: dhall brings in crypton, so we use that instead of cryptonite
      # (they have the same Crypto.Hash API)
      crypton # SHA256 hashing (same API as cryptonite)
      memory # crypton dependency
      unordered-containers # HashMap for JSON
      vector # Arrays for JSON
      unix # executeFile
      async # concurrency
      bytestring
      process
      containers
      transformers
      mtl
      time
    ];

  # GHC with Aleph.Script dependencies
  ghc-with-script = hs-pkgs.ghcWithPackages hs-deps;

  # QuickCheck deps for property tests
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
  #   - Buck2: haskell_script in src/tools/scripts/BUCK (via NativeLink)
  #
  # Both use identical GHC version (9.12) and package sets (hsDeps/SCRIPT_PACKAGES).
  # For iteration, use runghc with ghcWithScript in the devshell.
  #
  # If configExpr is provided, it generates a Dhall config file that the
  # script reads at startup via CONFIG_FILE environment variable.

  mk-compiled-script =
    {
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

  # ────────────────────────────────────────────────────────────────────────────
  # // dhall config generator //
  # ────────────────────────────────────────────────────────────────────────────
  #
  # Helper to generate Dhall expressions from Nix attrsets.
  # Handles store paths, naturals, text, and optional values.

in
{
  aleph = (prev.aleph or { }) // {
    script = {
      # ──────────────────────────────────────────────────────────────────────
      # // source //
      # ──────────────────────────────────────────────────────────────────────

      src = script-src;
      lib = aleph-src;
      corpus = corpus-src;

      # ──────────────────────────────────────────────────────────────────────
      # // ghc //
      # ──────────────────────────────────────────────────────────────────────

      # GHC with Aleph.Script modules available
      ghc = ghc-with-script;
      ghc-with-tests = ghc-with-tests;

      # ──────────────────────────────────────────────────────────────────────
      # // gen-wrapper //
      # ──────────────────────────────────────────────────────────────────────
      #
      # Unified wrapper generator. Auto-detects clap vs GNU format.
      #
      # Usage:
      #   aleph.script.gen-wrapper rg              # stdout
      #   aleph.script.gen-wrapper grep --gnu     # force GNU format
      #   aleph.script.gen-wrapper fd --write     # write to Tools/Fd.hs

      gen-wrapper = final.writeShellApplication {
        name = "aleph-gen-wrapper";
        "runtimeInputs" = [ ghc-with-script ];
        text = ''
          exec runghc -i${aleph-src} -i${script-src} ${script-src}/gen-wrapper.hs "$@"
        '';
      };

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

      # ──────────────────────────────────────────────────────────────────────
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
        '';
      };

      # ──────────────────────────────────────────────────────────────────────
      # // tools //
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
          "deadnix"
          "statix"
          "stylua"
          "taplo"
          "zoxide"
        ];
        # GNU getopt_long tools
        gnu = [
          "ls"
          "grep"
          "sed"
          "find"
          "xargs"
          "tar"
          "gzip"
          "wget"
          "rsync"
        ];
        # Hand-crafted domain-specific wrappers
        handcrafted = [
          "jq" # JSON processor
          "crane" # OCI image tool
          "bwrap" # bubblewrap sandbox
        ];
        # All tools
        all =
          final.aleph.script.tools.clap
          ++ final.aleph.script.tools.gnu
          ++ final.aleph.script.tools.handcrafted;
      };

      # ──────────────────────────────────────────────────────────────────────
      # // compiled //
      # ──────────────────────────────────────────────────────────────────────
      #
      # Compiled Haskell scripts for container/VM operations.
      # These replace the bash scripts in nix/modules/flake/container/.
      #
      # Each script is compiled to a static binary and wrapped with its
      # runtime dependencies.

      compiled = {
        # VFIO scripts - PCI device binding for GPU passthrough
        vfio-bind = mk-compiled-script {
          name = "vfio-bind";
          deps = [ final.pciutils ]; # lspci for device info
        };

        vfio-unbind = mk-compiled-script {
          name = "vfio-unbind";
          deps = [ final.pciutils ];
        };

        vfio-list = mk-compiled-script {
          name = "vfio-list";
          deps = [ final.pciutils ];
        };

        # Crane - OCI image operations (no runtime, just image manipulation)
        crane-inspect = mk-compiled-script {
          name = "crane-inspect";
          deps = [
            final.crane
            final.jq
          ];
        };

        crane-pull = mk-compiled-script {
          name = "crane-pull";
          deps = [ final.crane ];
        };

        # Unshare - bwrap/namespace runners for OCI images
        unshare-run = mk-compiled-script {
          name = "unshare-run";
          deps = [
            final.bubblewrap # Container sandbox
            final.crane # OCI image tool
            final.jq # JSON processing
          ];
        };

        unshare-gpu = mk-compiled-script {
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
          deps = [ final.bubblewrap ];
        };

        gpu-run = mk-compiled-script {
          name = "gpu-run";
          deps = [
            final.bubblewrap
            final.pciutils
          ];
        };

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

        # Cloud Hypervisor - VM management
        cloud-hypervisor-run = mk-compiled-script {
          name = "cloud-hypervisor-run";
          deps = [ final.cloud-hypervisor ];
        };

        cloud-hypervisor-gpu = mk-compiled-script {
          name = "cloud-hypervisor-gpu";
          deps = [
            final.cloud-hypervisor
            final.pciutils # GPU detection
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
  };
}
