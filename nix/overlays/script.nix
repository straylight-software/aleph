# nix/overlays/script.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                           // straylight.script //
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
#   straylight.script.gen-wrapper    unified wrapper generator (auto-detects format)
#   straylight.script.check          validation script for all tooling
#   straylight.script.ghc            GHC with Aleph.Script modules
#   straylight.script.tools          pre-generated tool wrappers
#   straylight.script.compiled.*     compiled Haskell scripts for container/VM ops
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
final: prev:
let
  inherit (prev) lib;

  # Source directory for Aleph.Script
  scriptSrc = ../scripts;

  # Haskell dependencies for Aleph.Script
  hsDeps =
    p: with p; [
      megaparsec
      text
      shelly
      foldl
      aeson
      dhall # Dhall config parsing
      directory
      # For oci-gpu and typed wrappers
      # Note: dhall brings in crypton, so we use that instead of cryptonite
      # (they have the same Crypto.Hash API)
      crypton # SHA256 hashing (same API as cryptonite)
      memory # crypton dependency
      unordered-containers # HashMap for JSON
      vector # Arrays for JSON
      unix # executeFile
      async # concurrency
      bytestring
    ];

  # GHC with Aleph.Script dependencies
  ghcWithScript = final.haskellPackages.ghcWithPackages hsDeps;

  # QuickCheck deps for property tests
  testDeps =
    p:
    hsDeps p
    ++ [
      p.QuickCheck
      p.deepseq
    ];
  ghcWithTests = final.haskellPackages.ghcWithPackages testDeps;

  # ────────────────────────────────────────────────────────────────────────────
  # // compiled script builder //
  # ────────────────────────────────────────────────────────────────────────────
  #
  # Compiles a Haskell script from nix/scripts/ to a binary, wrapping it
  # with runtime dependencies as needed.
  #
  # If configExpr is provided, it generates a Dhall config file that the
  # script reads at startup via CONFIG_FILE environment variable.

  mkCompiledScript =
    {
      name,
      deps ? [ ], # Runtime dependencies (wrapped into PATH)
      configExpr ? null, # Dhall expression (Nix string with store paths)
    }:
    let
      hasConfig = configExpr != null;
      # Generate config.dhall as a separate derivation
      configFile = final.writeText "${name}-config.dhall" configExpr;
    in
    final.stdenv.mkDerivation {
      inherit name;
      src = scriptSrc;
      dontUnpack = true;

      nativeBuildInputs = [ ghcWithScript ] ++ lib.optional (deps != [ ] || hasConfig) final.makeWrapper;

      buildPhase = ''
        runHook preBuild
        ghc -O2 -Wall -Wno-unused-imports \
          -hidir . -odir . \
          -i$src -o ${name} $src/${name}.hs
        runHook postBuild
      '';

      installPhase = ''
        runHook preInstall
        mkdir -p $out/bin
        cp ${name} $out/bin/
        ${lib.optionalString hasConfig ''
          mkdir -p $out/share/straylight
          cp ${configFile} $out/share/straylight/config.dhall
        ''}
        runHook postInstall
      '';

      postFixup =
        let
          wrapArgs =
            lib.optional (deps != [ ]) "--prefix PATH : ${lib.makeBinPath deps}"
            ++ lib.optional hasConfig "--set CONFIG_FILE $out/share/straylight/config.dhall";
        in
        lib.optionalString (wrapArgs != [ ]) ''
          wrapProgram $out/bin/${name} \
            ${lib.concatStringsSep " \\\n    " wrapArgs}
        '';
    };

  # ────────────────────────────────────────────────────────────────────────────
  # // dhall config generator //
  # ────────────────────────────────────────────────────────────────────────────
  #
  # Helper to generate Dhall expressions from Nix attrsets.
  # Handles store paths, naturals, text, and optional values.

in
{
  straylight = (prev.straylight or { }) // {
    script = {
      # ──────────────────────────────────────────────────────────────────────
      # // source //
      # ──────────────────────────────────────────────────────────────────────

      src = scriptSrc;

      # ──────────────────────────────────────────────────────────────────────
      # // ghc //
      # ──────────────────────────────────────────────────────────────────────

      # GHC with Aleph.Script modules available
      ghc = ghcWithScript;
      ghc-with-tests = ghcWithTests;

      # ──────────────────────────────────────────────────────────────────────
      # // gen-wrapper //
      # ──────────────────────────────────────────────────────────────────────
      #
      # Unified wrapper generator. Auto-detects clap vs GNU format.
      #
      # Usage:
      #   straylight.script.gen-wrapper rg              # stdout
      #   straylight.script.gen-wrapper grep --gnu     # force GNU format
      #   straylight.script.gen-wrapper fd --write     # write to Tools/Fd.hs

      gen-wrapper = final.writeShellApplication {
        name = "straylight-gen-wrapper";
        runtimeInputs = [ ghcWithScript ];
        text = ''
          cd ${scriptSrc}
          exec runghc -i. gen-wrapper.hs "$@"
        '';
      };

      # ──────────────────────────────────────────────────────────────────────
      # // check //
      # ──────────────────────────────────────────────────────────────────────
      #
      # Quick validation: compiles all wrappers, parses corpus, checks invariants.

      check = final.writeShellApplication {
        name = "straylight-script-check";
        runtimeInputs = [ ghcWithScript ];
        text = ''
          cd ${scriptSrc}
          exec runghc -i. check.hs "$@"
        '';
      };

      # ──────────────────────────────────────────────────────────────────────
      # // props //
      # ──────────────────────────────────────────────────────────────────────
      #
      # Property tests: parser totality, idempotence, preservation, compilation.

      props = final.writeShellApplication {
        name = "straylight-script-props";
        runtimeInputs = [ ghcWithTests ];
        text = ''
          cd ${scriptSrc}
          exec runghc -i. Props.hs "$@"
        '';
      };

      # ──────────────────────────────────────────────────────────────────────
      # // shell //
      # ──────────────────────────────────────────────────────────────────────
      #
      # Development shell for working on Aleph.Script.

      shell = final.mkShell {
        name = "straylight-script-shell";
        buildInputs = [
          ghcWithTests
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
        shellHook = ''
          cd ${scriptSrc}
          echo "Aleph.Script development shell"
          echo "  runghc -i. check.hs     # quick validation"
          echo "  runghc -i. Props.hs     # property tests"
          echo "  runghc -i. gen-wrapper.hs <tool>"
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
          final.straylight.script.tools.clap
          ++ final.straylight.script.tools.gnu
          ++ final.straylight.script.tools.handcrafted;
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
        vfio-bind = mkCompiledScript {
          name = "vfio-bind";
          deps = [ final.pciutils ]; # lspci for device info
        };

        vfio-unbind = mkCompiledScript {
          name = "vfio-unbind";
          deps = [ final.pciutils ];
        };

        vfio-list = mkCompiledScript {
          name = "vfio-list";
          deps = [ final.pciutils ];
        };

        # OCI scripts - container image and runtime operations
        oci-run = mkCompiledScript {
          name = "oci-run";
          deps = [
            final.bubblewrap # Container sandbox
            final.crane # OCI image tool
            final.jq # JSON processing
          ];
        };

        oci-gpu = mkCompiledScript {
          name = "oci-gpu";
          deps = [
            final.bubblewrap
            final.crane
            final.jq
            final.pciutils # GPU detection
          ];
        };

        oci-inspect = mkCompiledScript {
          name = "oci-inspect";
          deps = [
            final.crane
            final.jq
          ];
        };

        oci-pull = mkCompiledScript {
          name = "oci-pull";
          deps = [ final.crane ];
        };

        # FHS/GPU scripts - namespace environment wrappers
        fhs-run = mkCompiledScript {
          name = "fhs-run";
          deps = [ final.bubblewrap ];
        };

        gpu-run = mkCompiledScript {
          name = "gpu-run";
          deps = [
            final.bubblewrap
            final.pciutils
          ];
        };

        # Firecracker scripts - microVM management
        fc-run = mkCompiledScript {
          name = "fc-run";
          deps = [ final.firecracker ];
        };

        fc-build = mkCompiledScript {
          name = "fc-build";
          deps = [
            final.e2fsprogs # mke2fs for rootfs
            final.cpio # initramfs
            final.gzip
          ];
        };

        # Cloud Hypervisor scripts - VM management
        ch-run = mkCompiledScript {
          name = "ch-run";
          deps = [ final.cloud-hypervisor ];
        };

        ch-gpu = mkCompiledScript {
          name = "ch-gpu";
          deps = [
            final.cloud-hypervisor
            final.pciutils # GPU detection
          ];
        };

        # NVIDIA SDK extraction - pull from NGC, extract CUDA/cuDNN/TensorRT
        nvidia-extract = mkCompiledScript {
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
        nvidia-sdk-extract = mkCompiledScript {
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
        nvidia-wheel-extract = mkCompiledScript {
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
        nvidia-sdk = mkCompiledScript {
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

        # Build verification scripts
        verify-static-only = mkCompiledScript {
          name = "verify-static-only";
          deps = [ final.findutils ]; # find
        };
      };

      # Convenience: build all compiled scripts
      all-compiled = final.symlinkJoin {
        name = "straylight-scripts";
        paths = builtins.attrValues final.straylight.script.compiled;
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
        runtimeInputs = [
          ghcWithScript
          final.nix
        ];
        text = ''
          exec runghc -i${scriptSrc} ${scriptSrc}/nix-dev.hs "$@"
        '';
      };

      nix-ci = final.writeShellApplication {
        name = "nix-ci";
        runtimeInputs = [
          ghcWithScript
          final.nix
        ];
        text = ''
          exec runghc -i${scriptSrc} ${scriptSrc}/nix-ci.hs "$@"
        '';
      };
    };
  };
}
