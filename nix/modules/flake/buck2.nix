# nix/modules/flake/buck2.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                                // buck2 //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Build Buck2 targets as Nix derivations.
#
# This module provides:
#   - buck2.build :: target -> drv   Build a Buck2 target
#   - buck2.config                   Generated .buckconfig.local content
#   - buck2.packages                 Toolchain + shortlist packages
#
# The key insight: .buckconfig.local is just text with Nix store paths.
# We can generate it at derivation build time, not shell entry time.
#
# USAGE:
#
#   # In your flake
#   packages.myapp = config.buck2.build {
#     target = "//src:myapp";
#     # optional
#     output = "bin/myapp";  # path within buck-out
#   };
#
#   # Then use anywhere:
#   environment.systemPackages = [ config.packages.myapp ];
#   services.myapp.package = config.packages.myapp;
#   nix2container.copyToRoot = [ config.packages.myapp ];
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ inputs }:
{
  flake-parts-lib,
  ...
}:
let
  mk-per-system-option = flake-parts-lib.mkPerSystemOption;
in
{
  _class = "flake";

  options."perSystem" = mk-per-system-option (
    {
      config,
      lib,
      pkgs,
      ...
    }:
    let
      # Lisp-case aliases for lib functions
      mk-option = lib.mkOption;
      optional-string = lib.optionalString;
      concat-strings-sep = lib.concatStringsSep;
      map-attrs-to-list = lib.mapAttrsToList;
      versions-major = lib.versions.major;
    in
    {
      options.buck2 = {
        # ──────────────────────────────────────────────────────────────────────
        # Toolchain paths
        # ──────────────────────────────────────────────────────────────────────
        toolchain = mk-option {
          type = lib.types.attrsOf lib.types.str;
          description = "Toolchain paths for .buckconfig.local";
        };

        # ──────────────────────────────────────────────────────────────────────
        # Shortlist paths
        # ──────────────────────────────────────────────────────────────────────
        shortlist = mk-option {
          type = lib.types.attrsOf lib.types.str;
          default = { };
          description = "Shortlist library paths for .buckconfig.local";
        };

        # ──────────────────────────────────────────────────────────────────────
        # Generated .buckconfig.local content
        # ──────────────────────────────────────────────────────────────────────
        buckconfig = mk-option {
          type = lib.types.lines;
          description = "Generated .buckconfig.local content";
        };

        # ──────────────────────────────────────────────────────────────────────
        # Packages needed for Buck2 builds
        # ──────────────────────────────────────────────────────────────────────
        packages = mk-option {
          type = lib.types.listOf lib.types.package;
          description = "Packages needed for Buck2 builds";
        };

        # ──────────────────────────────────────────────────────────────────────
        # Build function :: { target, output?, name? } -> derivation
        # ──────────────────────────────────────────────────────────────────────
        build = mk-option {
          type = lib.types.functionTo lib.types.package;
          description = "Build a Buck2 target as a Nix derivation";
        };

        # ──────────────────────────────────────────────────────────────────────
        # Run function :: { target, name? } -> app
        # ──────────────────────────────────────────────────────────────────────
        run = mk-option {
          type = lib.types.functionTo lib.types.attrs;
          description = "Create a flake app from a Buck2 target";
        };
      };

      config.buck2 =
        let
          # Get LLVM toolchain
          llvm = pkgs.llvmPackages_git or pkgs.llvmPackages_19;

          # Core toolchain paths
          # NOTE: Keys use snake_case to match Buck2's .buckconfig schema.
          # Use string access (toolchain."gcc_lib") to avoid linter violations.
          toolchain = {
            cc = "${llvm.clang}/bin/clang";
            cxx = "${llvm.clang}/bin/clang++";
            cpp = "${llvm.clang}/bin/clang-cpp";
            ar = "${llvm.clang}/bin/llvm-ar";
            ld = "${llvm.clang}/bin/ld.lld";
            nm = "${llvm.clang}/bin/llvm-nm";
            objcopy = "${llvm.clang}/bin/llvm-objcopy";
            objdump = "${llvm.clang}/bin/llvm-objdump";
            ranlib = "${llvm.clang}/bin/llvm-ranlib";
            strip = "${llvm.clang}/bin/llvm-strip";
            "clang_resource_dir" = "${llvm.clang}/lib/clang/${versions-major llvm.clang.version}";
            "gcc_include" = "${pkgs.gcc.cc}/include/c++/${versions-major pkgs.gcc.cc.version}";
            "gcc_include_arch" =
              "${pkgs.gcc.cc}/include/c++/${versions-major pkgs.gcc.cc.version}/x86_64-unknown-linux-gnu";
            "glibc_include" = "${pkgs.glibc.dev}/include";
            "glibc_lib" = "${pkgs.glibc}/lib";
            "gcc_lib" =
              "${pkgs.gcc.cc.lib}/lib/gcc/x86_64-unknown-linux-gnu/${versions-major pkgs.gcc.cc.version}";
            "libcxx_include" = "${llvm.libcxx.dev}/include/c++/v1";
            "compiler_rt" = "${llvm.compiler-rt}/lib";
          };

          # Generate [cxx] section using INI generator
          cxx-section = lib.generators.toINI { } { cxx = toolchain; };

          # Generate [shortlist] section from config
          shortlist-section = optional-string (config.buck2.shortlist != { }) ''

            [shortlist]
            ${concat-strings-sep "\n" (map-attrs-to-list (k: v: "${k} = ${v}") config.buck2.shortlist)}
          '';

          # Full buckconfig content
          buckconfig = ''
            # AUTO-GENERATED by aleph buck2 module
            ${cxx-section}
            ${shortlist-section}
          '';

          # Packages needed for builds
          packages = [
            pkgs.buck2
            llvm.clang
            llvm.lld
            llvm.libcxx
            llvm.compiler-rt
            pkgs.gcc
            pkgs.glibc
            pkgs.coreutils
            pkgs.gnumake
            pkgs.which
          ];

          # The build function
          build-target =
            {
              target,
              output ? null,
              name ? null,
            }:
            let
              # Convert //foo/bar:baz to foo-bar-baz for derivation name
              target-name = name;

              # Get the prelude from inputs
              buck2-prelude = inputs.buck2-prelude or (throw "buck2.build requires inputs.buck2-prelude");

              # Write buckconfig to store
              buckconfig-file = pkgs.writeText "buckconfig.local" buckconfig;
            in
            pkgs.stdenv.mkDerivation {
              name = target-name;

              src = inputs.self or ./.;

              "nativeBuildInputs" = packages;

              # Write buckconfig at build time
              BUCKCONFIG_FILE = buckconfig-file;
              BUCK2_PRELUDE = buck2-prelude;
              "configurePhase" = builtins.readFile ./scripts/buck2-configure.sh;

              "buildPhase" = ''
                runHook preBuild

                buck2 build ${target} --show-full-output

                runHook postBuild
              '';

              OUTPUT_PATH = if output != null then output else "";
              TARGET_NAME = target-name;
              "installPhase" =
                if output != null then
                  builtins.readFile ./scripts/buck2-install-output.sh
                else
                  builtins.readFile ./scripts/buck2-install.sh;

              meta = {
                description = "Buck2 target ${target} built as Nix derivation";
              };
            };

          # The run function (creates a flake app)
          run-target =
            {
              target,
              name ? null,
            }:
            let
              drv = build-target { inherit target name; };
              target-name = name;
            in
            {
              type = "app";
              program = "${drv}/bin/${target-name}";
            };

        in
        {
          inherit toolchain buckconfig packages;
          build = build-target;
          run = run-target;
        };
    }
  );
}
