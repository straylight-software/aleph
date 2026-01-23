# Auto-generated from Dhall wheel templates
# DO NOT EDIT - regenerate with: dhall text < generate-nix.dhall

final: prev:
let
  inherit (prev) lib fetchurl;
in
{
  nvidia-nccl = prev.stdenv.mkDerivation {
    pname = "nvidia-nccl";
    version = "2.28.9";

    src = fetchurl {
      url = "https://pypi.nvidia.com/nvidia-nccl-cu13/nvidia_nccl_cu13-2.28.9-py3-none-manylinux_2_18_x86_64.whl";
      hash = "sha256-5FU6MPNBlfP6HaAqbaPWM30o8gA5Q6oKPSR7vCX+/EI=";
    };

    nativeBuildInputs = with final; [
      autoPatchelfHook
      unzip
    ];

    buildInputs = with final; [
      stdenv.cc.cc.lib
      zlib
    ];

    autoPatchelfIgnoreMissingDeps = [
      "libcuda.so.1"
      "libnvidia-ml.so.1"
    ];

    dontConfigure = true;
    dontBuild = true;
    dontUnpack = true;

    installPhase = ''
      runHook preInstall
      unzip -q $src/ -d $TMPDIR/unpacked
      mkdir -p $out/lib
      cp -r $TMPDIR/unpacked/nvidia/nccl/lib/. $out/lib/
      mkdir -p $out/include
      cp -r $TMPDIR/unpacked/nvidia/nccl/include/. $out/include/
      ln -s $out/lib $out/lib64
      runHook postInstall
    '';

    meta = {
      description = "NVIDIA NCCL 2.28.9 (from PyPI)";
      homepage = "https://developer.nvidia.com/nccl";
      license = lib.licenses.unfree;
      platforms = [ "x86_64-linux" ];
    };
  };

  nvidia-cudnn = prev.stdenv.mkDerivation {
    pname = "nvidia-cudnn";
    version = "9.17.0.29";

    src = fetchurl {
      url = "https://pypi.nvidia.com/nvidia-cudnn-cu13/nvidia_cudnn_cu13-9.17.0.29-py3-none-manylinux_2_27_x86_64.whl";
      hash = "sha256-Q0Uu8Jj0Q890hyvHj8zZon4af4NWelF5m/rsVgvP4Vo=";
    };

    nativeBuildInputs = with final; [
      autoPatchelfHook
      unzip
    ];

    buildInputs = with final; [
      stdenv.cc.cc.lib
      zlib
    ];

    autoPatchelfIgnoreMissingDeps = [
      "libcuda.so.1"
      "libnvidia-ml.so.1"
    ];

    dontConfigure = true;
    dontBuild = true;
    dontUnpack = true;

    installPhase = ''
      runHook preInstall
      unzip -q $src/ -d $TMPDIR/unpacked
      mkdir -p $out/lib
      cp -r $TMPDIR/unpacked/nvidia/cudnn/lib/. $out/lib/
      mkdir -p $out/include
      cp -r $TMPDIR/unpacked/nvidia/cudnn/include/. $out/include/
      ln -s $out/lib $out/lib64
      runHook postInstall
    '';

    meta = {
      description = "NVIDIA cuDNN 9.17.0.29 (from PyPI)";
      homepage = "https://developer.nvidia.com/cudnn";
      license = lib.licenses.unfree;
      platforms = [ "x86_64-linux" ];
    };
  };

  nvidia-tensorrt = prev.stdenv.mkDerivation {
    pname = "nvidia-tensorrt";
    version = "10.14.1.48";

    src = fetchurl {
      url = "https://pypi.nvidia.com/tensorrt-cu13-libs/tensorrt_cu13_libs-10.14.1.48-py2.py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256-k8SI67WjD/g+pTYD54GAFN5bkyj7JJZZY9I2gUB2UHY=";
    };

    nativeBuildInputs = with final; [
      autoPatchelfHook
      unzip
    ];

    buildInputs = with final; [
      stdenv.cc.cc.lib
      zlib
    ];

    autoPatchelfIgnoreMissingDeps = [
      "libcuda.so.1"
      "libnvidia-ml.so.1"
    ];

    dontConfigure = true;
    dontBuild = true;
    dontUnpack = true;

    installPhase = ''
      runHook preInstall
      unzip -q $src/ -d $TMPDIR/unpacked
      mkdir -p $out/lib
      cp -r $TMPDIR/unpacked/tensorrt_libs/. $out/lib/
      ln -s $out/lib $out/lib64
      runHook postInstall
    '';

    meta = {
      description = "NVIDIA TensorRT 10.14.1.48 (from PyPI)";
      homepage = "https://developer.nvidia.com/tensorrt";
      license = lib.licenses.unfree;
      platforms = [ "x86_64-linux" ];
    };
  };

  nvidia-cutensor = prev.stdenv.mkDerivation {
    pname = "nvidia-cutensor";
    version = "2.4.1";

    src = fetchurl {
      url = "https://pypi.nvidia.com/cutensor-cu13/cutensor_cu13-2.4.1-py3-none-manylinux2014_x86_64.whl";
      hash = "sha256-Hz1oTgSVOuRJI7ZzotQVbdmaghQAxC/ocqqF+PFmtyg=";
    };

    nativeBuildInputs = with final; [
      autoPatchelfHook
      unzip
    ];

    buildInputs = with final; [
      stdenv.cc.cc.lib
      zlib
    ];

    autoPatchelfIgnoreMissingDeps = [
      "libcuda.so.1"
      "libnvidia-ml.so.1"
    ];

    dontConfigure = true;
    dontBuild = true;
    dontUnpack = true;

    installPhase = ''
      runHook preInstall
      unzip -q $src/ -d $TMPDIR/unpacked
      mkdir -p $out/lib
      cp -r $TMPDIR/unpacked/cutensor/lib/. $out/lib/
      mkdir -p $out/include
      cp -r $TMPDIR/unpacked/cutensor/include/. $out/include/
      ln -s $out/lib $out/lib64
      runHook postInstall
    '';

    meta = {
      description = "NVIDIA cuTensor 2.4.1 (from PyPI)";
      homepage = "https://developer.nvidia.com/cutensor";
      license = lib.licenses.unfree;
      platforms = [ "x86_64-linux" ];
    };
  };

  nvidia-cusparselt = prev.stdenv.mkDerivation {
    pname = "nvidia-cusparselt";
    version = "0.8.1";

    src = fetchurl {
      url = "https://pypi.nvidia.com/nvidia-cusparselt-cu13/nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_x86_64.whl";
      hash = "sha256-eGzodWjDA/rbWvzHEC1FTNMEDXX2+GJvXbRg0YcfTdA=";
    };

    nativeBuildInputs = with final; [
      autoPatchelfHook
      unzip
    ];

    buildInputs = with final; [
      stdenv.cc.cc.lib
      zlib
    ];

    autoPatchelfIgnoreMissingDeps = [
      "libcuda.so.1"
      "libnvidia-ml.so.1"
    ];

    dontConfigure = true;
    dontBuild = true;
    dontUnpack = true;

    installPhase = ''
      runHook preInstall
      unzip -q $src/ -d $TMPDIR/unpacked
      mkdir -p $out/lib
      cp -r $TMPDIR/unpacked/nvidia/cusparselt/lib/. $out/lib/
      mkdir -p $out/include
      cp -r $TMPDIR/unpacked/nvidia/cusparselt/include/. $out/include/
      ln -s $out/lib $out/lib64
      runHook postInstall
    '';

    meta = {
      description = "NVIDIA cuSPARSELt 0.8.1 (from PyPI)";
      homepage = "https://developer.nvidia.com/cusparselt";
      license = lib.licenses.unfree;
      platforms = [ "x86_64-linux" ];
    };
  };

}
