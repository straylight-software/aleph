# nvidia-sdk version pins
#
# Sources prioritized in order:
#   1. PyPI wheels (nvidia.com) - no redistribution issues
#   2. Official tarballs - redistribution issues
#   3. NGC containers - redistribution issues
#
# NGC 25.11 blessed for Blackwell (sm_120)
rec {
  # ════════════════════════════════════════════════════════════════════════════
  # PyPI wheels from pypi.nvidia.com (preferred - no redistribution issues)
  # ════════════════════════════════════════════════════════════════════════════

  wheels = {
    # NCCL - Multi-GPU communication
    nccl = {
      version = "2.28.9";
      "x86_64-linux" = {
        url = "https://pypi.nvidia.com/nvidia-nccl-cu13/nvidia_nccl_cu13-2.28.9-py3-none-manylinux_2_18_x86_64.whl";
        hash = "sha256-5FU6MPNBlfP6HaAqbaPWM30o8gA5Q6oKPSR7vCX+/EI=";
        # Paths inside wheel
        lib-path = "nvidia/nccl/lib";
        include-path = "nvidia/nccl/include";
      };
      "aarch64-linux" = {
        url = "";
        hash = "";
      };
    };

    # cuDNN - Deep learning primitives
    cudnn = {
      version = "9.17.0.29";
      "x86_64-linux" = {
        url = "https://pypi.nvidia.com/nvidia-cudnn-cu13/nvidia_cudnn_cu13-9.17.0.29-py3-none-manylinux_2_27_x86_64.whl";
        hash = "sha256-Q0Uu8Jj0Q890hyvHj8zZon4af4NWelF5m/rsVgvP4Vo=";
        lib-path = "nvidia/cudnn/lib";
        include-path = "nvidia/cudnn/include";
      };
      "aarch64-linux" = {
        url = "";
        hash = "";
      };
    };

    # TensorRT - Inference optimization
    tensorrt = {
      version = "10.14.1.48";
      "x86_64-linux" = {
        url = "https://pypi.nvidia.com/tensorrt-cu13-libs/tensorrt_cu13_libs-10.14.1.48-py2.py3-none-manylinux_2_28_x86_64.whl";
        hash = "sha256-k8SI67WjD/g+pTYD54GAFN5bkyj7JJZZY9I2gUB2UHY=";
        lib-path = "tensorrt_libs";
        include-path = null; # libs-only wheel
      };
      "aarch64-linux" = {
        url = "";
        hash = "";
      };
    };

    # cuSPARSELt - Sparse matrix operations
    cusparselt = {
      version = "0.8.1";
      "x86_64-linux" = {
        url = "https://pypi.nvidia.com/nvidia-cusparselt-cu13/nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_x86_64.whl";
        hash = "sha256-eGzodWjDA/rbWvzHEC1FTNMEDXX2+GJvXbRg0YcfTdA=";
        lib-path = "nvidia/cusparselt/lib";
        include-path = "nvidia/cusparselt/include";
      };
      "aarch64-linux" = {
        url = "";
        hash = "";
      };
    };

    # cuTensor - Tensor operations
    cutensor = {
      version = "2.4.1";
      "x86_64-linux" = {
        url = "https://pypi.nvidia.com/cutensor-cu13/cutensor_cu13-2.4.1-py3-none-manylinux2014_x86_64.whl";
        hash = "sha256-Hz1oTgSVOuRJI7ZzotQVbdmaghQAxC/ocqqF+PFmtyg=";
        lib-path = "cutensor/lib";
        include-path = "cutensor/include";
      };
      "aarch64-linux" = {
        url = "";
        hash = "";
      };
    };

    # nvCOMP - GPU compression
    nvcomp = {
      version = "5.1.0.21";
      "x86_64-linux" = {
        url = "https://pypi.nvidia.com/nvidia-nvcomp-cu13/nvidia_nvcomp_cu13-5.1.0.21-py3-none-manylinux_2_28_x86_64.whl";
        hash = "sha256-uLifFENVKbdQ8vq2HDVlXiNGEYB+CFfWBsd8QYB+XVg=";
        # This is a Python module, not a C library
        python-path = "nvidia/nvcomp";
      };
      "aarch64-linux" = {
        url = "";
        hash = "";
      };
    };
  };

  # ════════════════════════════════════════════════════════════════════════════
  # Container sources (for full SDK extraction)
  # ════════════════════════════════════════════════════════════════════════════

  container = {
    # Tritonserver has everything: CUDA, cuDNN, NCCL, TensorRT, TensorRT-LLM
    tritonserver = {
      version = "25.11";
      "x86_64-linux" = {
        ref = "nvcr.io/nvidia/tritonserver:25.11-py3";
        hash = "sha256-yrTbMURSSc5kx4KTegTErpDjCWcjb9Ehp7pOUtP34pM=";
      };
      "aarch64-linux" = {
        ref = "nvcr.io/nvidia/tritonserver:25.11-py3-igpu";
        hash = "";
      };
    };

    # CUDA devel has full toolkit but no ML libs
    cuda-devel = {
      version = "13.0.1";
      "x86_64-linux" = {
        ref = "nvidia/cuda:13.0.1-devel-ubuntu22.04";
        hash = "";
      };
      "aarch64-linux" = {
        ref = "nvidia/cuda:13.0.1-devel-ubuntu22.04";
        hash = "";
      };
    };
  };

  # ════════════════════════════════════════════════════════════════════════════
  # CUTLASS (header-only, from GitHub)
  # ════════════════════════════════════════════════════════════════════════════

  cutlass = {
    version = "4.3.3";
    url = "https://github.com/NVIDIA/cutlass/archive/refs/tags/v4.3.3.zip";
    hash = "sha256-JGSBZqPafqpbIeF3VfxjiZW9B1snmi0Q13fk+HrpN6w=";
  };

  # ════════════════════════════════════════════════════════════════════════════
  # Expected versions (informational)
  # ════════════════════════════════════════════════════════════════════════════

  cuda-version = "13.0.1";
  cudnn-version = "9.17.0";
  nccl-version = "2.28.9";
  tensorrt-version = "10.14.1";
  cutensor-version = "2.4.1";

  # SM architectures
  sm = {
    volta = "sm_70";
    turing = "sm_75";
    ampere = "sm_80";
    ada = "sm_89";
    hopper = "sm_90";
    blackwell = "sm_100";
  };
}
