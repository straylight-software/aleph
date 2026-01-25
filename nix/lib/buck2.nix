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
{ inputs }:
let
  # Import prelude functions directly
  prelude = import ../prelude/functions.nix { lib = inputs.nixpkgs.lib; };

  inherit (prelude)
    map-attrs'
    to-upper
    replace
    join
    head
    to-string
    ;

  # Scripts directory
  scripts-dir = ./scripts;

  read-file = builtins.readFile;
  versions-major = inputs.nixpkgs.lib.versions.major;

  # Render Dhall template with env vars (converts attr names to UPPER_SNAKE_CASE)
  render-dhall =
    pkgs: name: src: vars:
    let
      env-vars = map-attrs' (k: v: {
        name = to-upper (replace [ "-" ] [ "_" ] k);
        value = to-string v;
      }) vars;
    in
    pkgs.aleph.run-command name
      (
        {
          native-build-inputs = [ pkgs.haskellPackages.dhall ];
        }
        // env-vars
      )
      ''
        dhall text --file ${src} > $out
      '';

  # Generate .buckconfig.local file using Dhall templates
  # NOTE: Dhall template expects UPPER_SNAKE_CASE env vars, so we use snake_case keys
  # that get uppercased by render-dhall
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
      clang-resource-dir = "${llvm.clang}/lib/clang/${versions-major llvm.clang.version}";
      gcc-include = "${pkgs.gcc.cc}/include/c++/${versions-major pkgs.gcc.cc.version}";
      gcc-include-arch = "${pkgs.gcc.cc}/include/c++/${versions-major pkgs.gcc.cc.version}/x86_64-unknown-linux-gnu";
      glibc-include = "${pkgs.glibc.dev}/include";
      glibc-lib = "${pkgs.glibc}/lib";
      gcc-lib = "${pkgs.gcc.cc.lib}/lib/gcc/x86_64-unknown-linux-gnu/${versions-major pkgs.gcc.cc.version}";
      libcxx-include = "${llvm.libcxx.dev}/include/c++/v1";
      compiler-rt = "${llvm.compiler-rt}/lib";
      fmt = "${pkgs.fmt}";
      fmt-dev = "${pkgs.fmt.dev}";
      zlib-ng = "${pkgs.zlib-ng}";
      catch2 = "${pkgs.catch2_3}";
      catch2-dev = "${pkgs.catch2_3.dev or pkgs.catch2_3}";
      spdlog = "${pkgs.spdlog}";
      spdlog-dev = "${pkgs.spdlog.dev or pkgs.spdlog}";
      mdspan = "${pkgs.mdspan}";
      rapidjson = "${pkgs.rapidjson}";
      nlohmann-json = "${pkgs.nlohmann_json}";
      libsodium = "${pkgs.libsodium}";
      libsodium-dev = "${pkgs.libsodium.dev or pkgs.libsodium}";
    };

  # For backwards compatibility: generate buckconfig content string
  mk-buckconfig = pkgs: read-file (mk-buckconfig-file pkgs);

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
      raw-name = replace [ "//" "/" ":" ] [ "" "-" "-" ] target;
      # Remove leading/trailing dashes
      clean-name =
        let
          s1 = if prelude.starts-with "-" raw-name then builtins.substring 1 (-1) raw-name else raw-name;
          len = builtins.stringLength s1;
        in
        if prelude.ends-with "-" s1 then builtins.substring 0 (len - 1) s1 else s1;

      target-name =
        if name != null then
          name
        else if clean-name == "" then
          "buck2-target"
        else
          clean-name;

      # Get prelude
      buck2-prelude =
        inputs.buck2-prelude or (throw "aleph.lib.buck2.build requires inputs.buck2-prelude");

      buckconfig-file = mk-buckconfig-file pkgs;
      packages = mk-packages pkgs;
      output-name = if output != null then output else target-name;
    in
    pkgs.aleph.stdenv.default {
      name = target-name;
      inherit src;

      native-build-inputs = packages;

      configure-phase = read-file (scripts-dir + "/buck2-configure.bash");
      build-phase = read-file (scripts-dir + "/buck2-build.bash");
      install-phase = read-file (scripts-dir + "/buck2-install.bash");

      # Environment variables for scripts (passed through as-is)
      inherit buck2-prelude;
      inherit buckconfig-file;
      inherit output-name;
      buck2-target = target;

      meta = {
        description = "Buck2 target ${target} built as Nix derivation";
      };
    };

  # Get the buckconfig file for inspection/debugging
  buckconfig-file = mk-buckconfig-file;

  # Get the buckconfig content for inspection/debugging (backwards compat)
  buckconfig = mk-buckconfig;

  # Get the build packages list
  packages = mk-packages;
}
