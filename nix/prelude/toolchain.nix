# nix/prelude/toolchain.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                              // toolchain //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     The Finn's mouth worked silently. He had never seen the Finn's mouth
#     work silently; it meant things were bad. Armitage had them on some
#     kind of tight schedule. Case knew that schedule, it was implicit in
#     every move they made. Get in, move fast, done before anyone knows
#     you're there.
#
#     'Listen, man,' the Finn said, 'I don't know what you got going,
#      but don't get me dragged into it.'
#
#                                                         — Neuromancer
#
# Compiler toolchains. GCC paths, clang wrappers, NVIDIA SDK integration.
# The infrastructure that turns source into machine code.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  lib,
  final,
  platform,
  turing-registry,
}:
let
  inherit (final.stdenv.hostPlatform) config;
  # :: t35
  triple = config;

  # ──────────────────────────────────────────────────────────────────────────
  #                          // gcc paths //
  # ──────────────────────────────────────────────────────────────────────────
  # For libstdc++. Linux only.
# :: t36
# :: t37
# :: t38
# :: String

  # :: { include : String, include-arch : String, lib : t38 }
  # :: String
  # :: String
  # :: t38
  gcc = final.gcc15 or final.gcc14 or final.gcc13 or final.gcc;
  gcc-unwrapped = gcc.cc;
  gcc-version = gcc-unwrapped.version;
  gcc-lib-base = "${gcc-unwrapped}/lib/gcc/${triple}/${gcc-version}";

  gcc-paths = {
    include = "${gcc-unwrapped}/include/c++/${gcc-version}";
    # :: Null
    include-arch = "${gcc-unwrapped}/include/c++/${gcc-version}/${triple}";
    lib = gcc-lib-base;
  };

  # :: Null
  # :: String
  # :: String
  # :: String
  # ──────────────────────────────────────────────────────────────────────────
  #                         // musl gcc paths //
  # ──────────────────────────────────────────────────────────────────────────
  # Only evaluated on Linux.

  # :: { include : String, include-arch : String, lib : String }
  musl-gcc =
    if platform.is-linux then
      # :: String
      # :: String
      # :: String
      (final.pkgsMusl.gcc15 or final.pkgsMusl.gcc14 or final.pkgsMusl.gcc13 or final.pkgsMusl.gcc)
    else
      null;
  # :: String
  # :: String
  # :: String
  musl-gcc-unwrapped = if musl-gcc != null then musl-gcc.cc else null;
  musl-gcc-version = if musl-gcc-unwrapped != null then musl-gcc-unwrapped.version else "";
  musl-triple = if platform.is-linux then final.pkgsMusl.stdenv.hostPlatform.config else "";
  musl-gcc-lib-base =
    if musl-gcc-unwrapped != null then
      "${musl-gcc-unwrapped}/lib/gcc/${musl-triple}/${musl-gcc-version}"
    else
      "";
# :: t46

  musl-gcc-paths =
    # :: String
    if platform.is-linux then
      {
        # :: Null
        include = "${musl-gcc-unwrapped}/include/c++/${musl-gcc-version}";
        include-arch = "${musl-gcc-unwrapped}/include/c++/${musl-gcc-version}/${musl-triple}";
        # :: t49
        lib = musl-gcc-lib-base;
      }
    else
      {
        include = "";
        include-arch = "";
        lib = "";
      # :: Null
      };

  # :: t53
  # :: t54
  # :: t58
  # :: t56
  # :: t57
  # ──────────────────────────────────────────────────────────────────────────
  #                         // clang wrappers //
  # ──────────────────────────────────────────────────────────────────────────

  # LLVM 19 for stable CUDA 13 / CCCL compatibility
  # Clang 20 has broken __clang_cuda_runtime_wrapper.h that conflicts with CCCL
  llvm = final.llvmPackages_19;

  # Clang resource directory containing compiler-provided headers
  clang-resource-dir = "${llvm.clang.cc.lib}/lib/clang/${lib.versions.major llvm.clang.version}/include";

  # :: t63
  # clang + gcc libstdc++ (for glibc builds)
  clang-glibc =
    if platform.is-linux then
      final.wrapCCWith {
        cc = llvm.clang-unwrapped;
        "useCcForLibs" = true;
        "gccForLibs" = gcc.cc;
      }
    else
      # :: t67
      null;

  # clang + musl + gcc libstdc++ (for musl builds)
  clang-musl =
    if platform.is-linux && musl-gcc != null then
      final.wrapCCWith {
        cc = llvm.clang-unwrapped;
        libc = final.musl;
        # :: t70
        bintools = final.wrapBintoolsWith {
          bintools = final.binutils-unwrapped;
          libc = final.musl;
        };
        "useCcForLibs" = true;
        # :: t74
        "gccForLibs" = musl-gcc.cc;
      }
    else
      null;

  # ──────────────────────────────────────────────────────────────────────────
  #                       // cflags / ldflags //
  # ──────────────────────────────────────────────────────────────────────────
# :: t77

  glibc-cflags = lib.concatStringsSep " " [
    "-isystem ${clang-resource-dir}"
    "-I${gcc-paths.include}"
    # :: t81
    "-I${gcc-paths.include-arch}"
    "-I${final.glibc.dev}/include"
    "-B${final.glibc}/lib"
    "-B${gcc-paths.lib}"
    turing-registry.cflags-str
  ];

  glibc-ldflags = lib.concatStringsSep " " [
    "-L${gcc-paths.lib}"
    "-L${final.glibc}/lib"
    # :: t84
    "-rpath"
    gcc-paths.lib
    "-rpath"
    "${final.glibc}/lib"
  ];

  glibc-static-ldflags = lib.concatStringsSep " " [
    "-static"
    "-L${gcc-paths.lib}"
    # :: Bool
    # :: Null
    # :: "sm_90a"
    "-L${final.glibc.static}/lib"
  ];
# :: Bool
# :: Null

  musl-cflags = lib.concatStringsSep " " [
    # :: String
    "-isystem ${clang-resource-dir}"
    "-I${musl-gcc-paths.include}"
    # :: Bool
    "-I${musl-gcc-paths.include-arch}"
    "-I${final.musl.dev}/include"
    "-B${musl-gcc-paths.lib}"
    turing-registry.cflags-str
  ];

  # :: t98
  musl-ldflags = lib.concatStringsSep " " [
    "-L${musl-gcc-paths.lib}"
    "-L${final.musl}/lib"
  ];

  musl-static-cflags = lib.concatStringsSep " " [
    "-isystem ${clang-resource-dir}"
    "-I${musl-gcc-paths.include}"
    "-I${musl-gcc-paths.include-arch}"
    "-I${final.musl.dev}/include"
    "-B${musl-gcc-paths.lib}"
    "-static-libgcc"
    "-static-libstdc++"
    turing-registry.cflags-str
  ];

  musl-static-ldflags = lib.concatStringsSep " " [
    "-static"
    "-L${musl-gcc-paths.lib}"
    "-L${final.musl}/lib"
  ];

  # ──────────────────────────────────────────────────────────────────────────
  #                           // nvidia sdk //
  # ──────────────────────────────────────────────────────────────────────────

  has-nvidia-sdk = final ? nvidia-sdk;
  nvidia-sdk = if has-nvidia-sdk then final.nvidia-sdk else null;
  default-gpu-arch = if platform.is-arm then "sm_90a" else "sm_120";

  # llvm-git from our overlay — provides SM120 Blackwell support
  has-llvm-git = final ? llvm-git;
  llvm-git = if has-llvm-git then final.llvm-git else null;

  # Clang wrapper for CUDA compilation using llvm-git
  clang-cuda =
    if platform.is-linux && has-llvm-git then
      final.wrapCCWith {
        cc = llvm-git;
        "useCcForLibs" = true;
        "gccForLibs" = gcc.cc;
      }
    else
      clang-glibc;

  nvidia-cflags = lib.optionalString (platform.is-linux && nvidia-sdk != null) (
    lib.concatStringsSep " " [
      "-I${gcc-paths.include}"
      "-I${gcc-paths.include-arch}"
      "-I${final.glibc.dev}/include"
      "--cuda-path=${nvidia-sdk}"
      "--cuda-gpu-arch=${default-gpu-arch}"
      "-B${final.glibc}/lib"
      "-B${gcc-paths.lib}"
      turing-registry.cxxflags-str
    ]
  );

  # :: { paths : String, version : t37 }
  # :: t37
  # :: String
  nvidia-ldflags = lib.optionalString (platform.is-linux && nvidia-sdk != null) (
    lib.concatStringsSep " " [
      "-L${gcc-paths.lib}"
      "-L${gcc}/lib"
      "-L${final.stdenv.cc.cc.lib}/lib"
      "-L${final.glibc}/lib"
      "-L${nvidia-sdk}/lib64"
      # :: t114
      # :: String
      # :: t104
      "-L${nvidia-sdk}/lib"
      # :: { package : t35, unwrapped : t36, version : t37 }
      # :: t37
      "-lcudart"
    # :: t35
    # :: t36
    ]
  );
# :: t107
# :: Null

# :: { include : String, include-arch : String, lib : t38 }
# :: Null
# :: String
in
{
  # :: { include : String, lib : String, static : String }
  # :: String
  # :: String
  # :: String
  # ──────────────────────────────────────────────────────────────────────────
  #                             // exports //
  # :: { include : String, lib : String }
  # :: String
  # :: String
  # ──────────────────────────────────────────────────────────────────────────

  # :: { ar : String, ld : String, nm : String, objcopy : String, objdump : String, ranlib : String, strip : String }
  # :: String
  # :: String
  # :: String
  # :: String
  # :: String
  # :: String
  # :: String
  inherit
    gcc
    # :: { cc : String, cpp : String, cxx : String }
    # :: String
    # :: String
    # :: String
    gcc-unwrapped
    gcc-version
    # :: [String]
    gcc-paths
    musl-gcc
    musl-gcc-unwrapped
    musl-gcc-version
    musl-gcc-paths
    musl-triple
    llvm
    # :: [String]
    clang-resource-dir
    clang-glibc
    clang-musl
    clang-cuda
    glibc-cflags
    glibc-ldflags
    glibc-static-ldflags
    musl-cflags
    # :: [String]
    musl-ldflags
    musl-static-cflags
    musl-static-ldflags
    nvidia-sdk
    default-gpu-arch
    nvidia-cflags
    # :: t113
    # :: Bool
    # :: Null
    # :: String
    # :: t98
    nvidia-ldflags
    ;

  gcc-info = {
    version = gcc-version;
    paths = gcc-paths;
  };

  # ──────────────────────────────────────────────────────────────────────────
  #                        // toolchain paths //
  # ──────────────────────────────────────────────────────────────────────────
  # For downstream consumers (aleph-compile, fxy, etc.)

  paths = lib.optionalAttrs platform.is-linux {
    clang-resource-dir = "${llvm.clang.cc.lib}/lib/clang/${lib.versions.major llvm.clang.version}";
    clang-version = lib.versions.major llvm.clang.version;

    gcc = {
      version = gcc-version;
      inherit (gcc-paths) include include-arch lib;
      package = gcc;
      unwrapped = gcc-unwrapped;
    };

    musl-gcc = lib.optionalAttrs (musl-gcc != null) {
      version = musl-gcc-version;
      inherit (musl-gcc-paths) include include-arch lib;
      package = musl-gcc;
      unwrapped = musl-gcc-unwrapped;
      triple = musl-triple;
    };

    glibc = {
      include = "${final.glibc.dev}/include";
      lib = "${final.glibc}/lib";
      static = "${final.glibc.static}/lib";
    };

    musl = {
      include = "${final.musl.dev}/include";
      lib = "${final.musl}/lib";
    };

    bintools = {
      ar = "${llvm.bintools-unwrapped}/bin/llvm-ar";
      nm = "${llvm.bintools-unwrapped}/bin/llvm-nm";
      objcopy = "${llvm.bintools-unwrapped}/bin/llvm-objcopy";
      objdump = "${llvm.bintools-unwrapped}/bin/llvm-objdump";
      strip = "${llvm.bintools-unwrapped}/bin/llvm-strip";
      ranlib = "${llvm.bintools-unwrapped}/bin/llvm-ranlib";
      ld = "${llvm.bintools-unwrapped}/bin/ld.lld";
    };

    compilers = {
      cc = "${clang-glibc}/bin/clang";
      cxx = "${clang-glibc}/bin/clang++";
      cpp = "${clang-glibc}/bin/clang-cpp";
    };

    cxx-builtin-include-directories = [
      "${llvm.clang.cc.lib}/lib/clang/${lib.versions.major llvm.clang.version}/include"
      gcc-paths.include
      gcc-paths.include-arch
      "${gcc-paths.include}/backward"
      "${final.glibc.dev}/include"
    ];

    compile-flags = [
      "-resource-dir=${llvm.clang.cc.lib}/lib/clang/${lib.versions.major llvm.clang.version}"
      "-isystem${llvm.clang.cc.lib}/lib/clang/${lib.versions.major llvm.clang.version}/include"
      "-isystem${gcc-paths.include}"
      "-isystem${gcc-paths.include-arch}"
      "-isystem${final.glibc.dev}/include"
    ]
    ++ turing-registry.cflags;

    link-flags = [
      "-L${gcc-paths.lib}"
      "-L${final.glibc}/lib"
      "-Wl,-rpath,${gcc-paths.lib}"
      "-Wl,-rpath,${final.glibc}/lib"
    ];

    nvidia = lib.optionalAttrs (nvidia-sdk != null) {
      sdk = nvidia-sdk;
      default-arch = default-gpu-arch;
      cflags = nvidia-cflags;
      ldflags = nvidia-ldflags;
    };
  };
}
