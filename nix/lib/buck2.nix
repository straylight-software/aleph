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
  scriptsDir = ./scripts;

  # Generate .buckconfig.local file using replaceVars
  mkBuckconfigFile =
    pkgs:
    let
      llvm = pkgs.llvmPackages_git or pkgs.llvmPackages_19;
    in
    pkgs.replaceVars (scriptsDir + "/buckconfig.template") {
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
  mkBuckconfig = pkgs: builtins.readFile (mkBuckconfigFile pkgs);

  # Build packages needed for Buck2
  mkPackages =
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
      rawName = builtins.replaceStrings [ "//" "/" ":" ] [ "" "-" "-" ] target;
      cleanName = lib.removePrefix "-" (lib.removeSuffix "-" rawName);
      targetName =
        if name != null then
          name
        else if cleanName == "" then
          "buck2-target"
        else
          cleanName;

      # Get prelude
      prelude = inputs.buck2-prelude or (throw "aleph.lib.buck2.build requires inputs.buck2-prelude");

      buckconfigFile = mkBuckconfigFile pkgs;
      packages = mkPackages pkgs;
      outputName = if output != null then output else targetName;
    in
    pkgs.stdenv.mkDerivation {
      name = targetName;
      inherit src;

      nativeBuildInputs = packages;

      # Environment variables for scripts
      inherit buckconfigFile prelude outputName;
      buck2Target = target;

      configurePhase = builtins.readFile (scriptsDir + "/buck2-configure.bash");
      buildPhase = builtins.readFile (scriptsDir + "/buck2-build.bash");
      installPhase = builtins.readFile (scriptsDir + "/buck2-install.bash");

      meta = {
        description = "Buck2 target ${target} built as Nix derivation";
      };
    };

  # Get the buckconfig file for inspection/debugging
  buckconfigFile = mkBuckconfigFile;

  # Get the buckconfig content for inspection/debugging (backwards compat)
  buckconfig = mkBuckconfig;

  # Get the build packages list
  packages = mkPackages;
}
