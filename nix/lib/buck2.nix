# nix/lib/buck2.nix
#
# Buck2 builder library function.
#
# Usage in downstream flakes:
#
#   packages.myapp = aleph.lib.buck2.build pkgs {
#     src = ./.;
#     target = "//src:myapp";
#   };
#
{ inputs, lib }:
let
  # Scripts directory
  scripts-dir = ./scripts;

  # Render Dhall template with env vars (converts attr names to UPPER_SNAKE_CASE)
  render-dhall =
    pkgs: name: src: vars:
    let
      env-vars = lib.mapAttrs' (
        k: v: lib.nameValuePair (lib.toUpper (builtins.replaceStrings [ "-" ] [ "_" ] k)) (toString v)
      ) vars;
    in
    pkgs.runCommand name
      (
        {
          nativeBuildInputs = [ pkgs.haskellPackages.dhall ];
        }
        // env-vars
      )
      ''
        dhall text --file ${src} > $out
      '';

  # Generate .buckconfig.local file using Dhall templates
  mk-buckconfig-file =
    pkgs:
    let
      llvm = pkgs.llvmPackages_git or pkgs.llvmPackages_19;
    in
    render-dhall pkgs "buckconfig-local" (scripts-dir + "/buckconfig.dhall") {
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
      clang_resource_dir = "${llvm.clang}/lib/clang/${lib.versions.major llvm.clang.version}";
      gcc_include = "${pkgs.gcc.cc}/include/c++/${lib.versions.major pkgs.gcc.cc.version}";
      gcc_include_arch = "${pkgs.gcc.cc}/include/c++/${lib.versions.major pkgs.gcc.cc.version}/x86_64-unknown-linux-gnu";
      glibc_include = "${pkgs.glibc.dev}/include";
      glibc_lib = "${pkgs.glibc}/lib";
      gcc_lib = "${pkgs.gcc.cc.lib}/lib/gcc/x86_64-unknown-linux-gnu/${lib.versions.major pkgs.gcc.cc.version}";
      libcxx_include = "${llvm.libcxx.dev}/include/c++/v1";
      compiler_rt = "${llvm.compiler-rt}/lib";
      fmt = "${pkgs.fmt}";
      fmt_dev = "${pkgs.fmt.dev}";
      zlib_ng = "${pkgs.zlib-ng}";
      catch2 = "${pkgs.catch2_3}";
      catch2_dev = "${pkgs.catch2_3.dev or pkgs.catch2_3}";
      spdlog = "${pkgs.spdlog}";
      spdlog_dev = "${pkgs.spdlog.dev or pkgs.spdlog}";
      mdspan = "${pkgs.mdspan}";
      rapidjson = "${pkgs.rapidjson}";
      nlohmann_json = "${pkgs.nlohmann_json}";
      libsodium = "${pkgs.libsodium}";
      libsodium_dev = "${pkgs.libsodium.dev or pkgs.libsodium}";
    };

  # For backwards compatibility: generate buckconfig content string
  mk-buckconfig = pkgs: builtins.readFile (mk-buckconfig-file pkgs);

  # Build packages needed for Buck2
  mk-packages =
    pkgs:
    let
      llvm = pkgs.llvmPackages_git or pkgs.llvmPackages_19;
    in
    [
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

in
{
  # Build a Buck2 target as a Nix derivation
  #
  # Usage:
  #   aleph.lib.buck2.build pkgs {
  #     src = ./.;
  #     target = "//examples/cxx:fmt_test";
  #     # optional:
  #     # name = "my-fmt-test";
  #     # output = "fmt_test";  # binary name in buck-out
  #   }
  #
  build =
    pkgs:
    {
      src,
      target,
      name ? null,
      output ? null,
    }:
    let
      # Convert //foo/bar:baz to foo-bar-baz for derivation name
      raw-name = builtins.replaceStrings [ "//" "/" ":" ] [ "" "-" "-" ] target;
      clean-name = lib.removePrefix "-" (lib.removeSuffix "-" raw-name);
      target-name =
        if name != null then
          name
        else if clean-name == "" then
          "buck2-target"
        else
          clean-name;

      # Get prelude
      prelude = inputs.buck2-prelude or (throw "aleph.lib.buck2.build requires inputs.buck2-prelude");

      buckconfig-file = mk-buckconfig-file pkgs;
      packages = mk-packages pkgs;
      output-name = if output != null then output else target-name;
    in
    pkgs.stdenv.mkDerivation {
      name = target-name;
      inherit src;

      nativeBuildInputs = packages;

      # Environment variables for scripts
      buckconfigFile = buckconfig-file;
      inherit prelude;
      outputName = output-name;
      buck2Target = target;

      configurePhase = builtins.readFile (scripts-dir + "/buck2-configure.bash");
      buildPhase = builtins.readFile (scripts-dir + "/buck2-build.bash");
      installPhase = builtins.readFile (scripts-dir + "/buck2-install.bash");

      meta = {
        description = "Buck2 target ${target} built as Nix derivation";
      };
    };

  # Get the buckconfig file for inspection/debugging
  buckconfigFile = mk-buckconfig-file;

  # Get the buckconfig content for inspection/debugging (backwards compat)
  buckconfig = mk-buckconfig;

  # Get the build packages list
  packages = mk-packages;
}
