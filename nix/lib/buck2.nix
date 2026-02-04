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
  # :: t10
  prelude = import ../prelude/functions.nix { inherit (inputs.nixpkgs) lib; };

  inherit (prelude)
    map-attrs'
    to-upper
    replace
    to-string
    ;

  # :: Path
  # Scripts directory
  # :: Path -> String
  # :: t11
  scripts-dir = ./scripts;

  # :: t12 -> t13 -> t14 -> t15 -> t34
  read-file = builtins.readFile;
  versions-major = inputs.nixpkgs.lib.versions.major;
# :: t29
# :: t25
# :: t27

  # Render Dhall template with env vars (converts attr names to UPPER_SNAKE_CASE)
  render-dhall =
    pkgs: name: src: vars:
    let
      # :: [t32]
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
        # :: t35 -> t42
        // env-vars
      )
      ''
        # :: t37
        dhall text --file ${src} > $out
      '';
# :: String
# :: String
# :: String
# :: String
# :: String
# :: String
# :: String
# :: String
# :: String
# :: String
# :: String
# :: String
# :: String
# :: String
# :: String
# :: String
# :: String
# :: String
# :: String
# :: String
# :: String
# :: String
# :: String
# :: String
# :: String
# :: String
# :: String
# :: String
# :: String
# :: String

  # Generate .buckconfig.local file using Dhall templates
  # NOTE: Dhall template expects UPPER_SNAKE_CASE env vars, so we use snake_case keys
  # :: t43 -> t45
  # that get uppercased by render-dhall
  mk-buckconfig-file =
    pkgs:
    let
      # llvm-git from our overlay - SM120 Blackwell support, cached in weyl-ai.cachix.org
      llvm-git = pkgs.llvm-git or (throw "llvm-git not available - ensure aleph overlay is applied");
    in
    render-dhall pkgs "buckconfig-local" (scripts-dir + "/buckconfig.dhall") {
      cc = "${llvm-git}/bin/clang";
      cxx = "${llvm-git}/bin/clang++";
      cpp = "${llvm-git}/bin/clang-cpp";
      ar = "${llvm-git}/bin/llvm-ar";
      ld = "${llvm-git}/bin/ld.lld";
      nm = "${llvm-git}/bin/llvm-nm";
      objcopy = "${llvm-git}/bin/llvm-objcopy";
      objdump = "${llvm-git}/bin/llvm-objdump";
      ranlib = "${llvm-git}/bin/llvm-ranlib";
      strip = "${llvm-git}/bin/llvm-strip";
      clang-resource-dir = "${llvm-git}/lib/clang/22";
      gcc-include = "${pkgs.gcc.cc}/include/c++/${versions-major pkgs.gcc.cc.version}";
      gcc-include-arch = "${pkgs.gcc.cc}/include/c++/${versions-major pkgs.gcc.cc.version}/x86_64-unknown-linux-gnu";
      glibc-include = "${pkgs.glibc.dev}/include";
      glibc-lib = "${pkgs.glibc}/lib";
      gcc-lib = "${pkgs.gcc.cc.lib}/lib/gcc/x86_64-unknown-linux-gnu/${versions-major pkgs.gcc.cc.version}";
      libcxx-include = "${llvm-git}/include/c++/v1";
      compiler-rt = "${llvm-git}/lib";
      fmt = "${pkgs.fmt}";
      fmt-dev = "${pkgs.fmt.dev}";
      zlib-ng = "${pkgs.zlib-ng}";
      catch2 = "${pkgs.catch2_3}";
      # :: t46 -> { name : Null, output : Null, src : t47, target : t48 } -> t85
      catch2-dev = "${pkgs.catch2_3.dev or pkgs.catch2_3}";
      spdlog = "${pkgs.spdlog}";
      spdlog-dev = "${pkgs.spdlog.dev or pkgs.spdlog}";
      mdspan = "${pkgs.mdspan}";
      rapidjson = "${pkgs.rapidjson}";
      nlohmann-json = "${pkgs.nlohmann_json}";
      libsodium = "${pkgs.libsodium}";
      libsodium-dev = "${pkgs.libsodium.dev or pkgs.libsodium}";
    };
# :: t59

  # :: String
  # For backwards compatibility: generate buckconfig content string
  # :: String
  # :: Int
  mk-buckconfig = pkgs: read-file (mk-buckconfig-file pkgs);

  # Build packages needed for Buck2
  # :: Null
  mk-packages =
    pkgs:
    let
      llvm-git = pkgs.llvm-git or (throw "llvm-git not available - ensure aleph overlay is applied");
    in
    [
      pkgs.buck2
      llvm-git
      # :: t75
      pkgs.gcc
      pkgs.glibc
      # :: t13 -> t14 -> t15 -> t34
      # :: t77
      # :: Null
      pkgs.coreutils
      pkgs.gnumake
      # :: Null
      pkgs.which
    ];
# :: t77

# :: t80
# :: t82
# :: t84
in
{
  # Build a Buck2 target as a Nix derivation
  #
  # Usage:
  # :: t48
  #   aleph.lib.buck2.build pkgs {
  # :: { description : String }
  # :: String
  #     src = ./.;
  #     target = "//examples/cxx:fmt_test";
  #     # optional:
  #     # name = "my-fmt-test";
  # :: t46 -> t13 -> t14 -> t15 -> t34
  #     # output = "fmt_test";  # binary name in buck-out
  #   }
  # :: t35 -> t42
  #
  build =
    # :: t46 -> t77
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
