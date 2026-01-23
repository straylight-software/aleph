--| wheel-template.dhall: Template for NVIDIA wheel packages
--|
--| This is a reusable template that generates typed wheel package definitions.
--| The template captures the common pattern for all nvidia-* wheel packages.

let Prelude = https://prelude.dhall-lang.org/v23.1.0/package.dhall
    sha256:931cbfae9d746c4611b07633ab1e547637ab4ba138b16bf65ef1b9ad66a60b7f

let Script = ../prelude/Script.dhall

-- =============================================================================
-- Wheel Package Template
-- =============================================================================

let WheelSrc =
      { url : Text
      , sha256 : Text
      }

let WheelPaths =
      { libPath : Text        -- e.g., "nvidia/nccl/lib"
      , includePath : Optional Text  -- e.g., Some "nvidia/nccl/include"
      }

let WheelPackage =
      { Type =
          { pname : Text
          , version : Text
          , src : WheelSrc
          , paths : WheelPaths
          , description : Text
          , homepage : Text
          }
      }

-- | Generate the install script for a wheel package
let mkWheelInstallScript
    : WheelPaths -> Script.Script
    = \(paths : WheelPaths) ->
        let includeCommands =
              merge
                { Some = \(incPath : Text) ->
                    [ Script.mkdir "include"
                    , Script.Command.Copy
                        { src = Script.Path.Tmp "unpacked/${incPath}/."
                        , dst = Script.Path.Out "include/"
                        , recursive = True
                        }
                    ]
                , None = [] : List Script.Command
                }
                paths.includePath
        in
          [ -- Unzip the wheel
            Script.Command.Unzip
              { archive = Script.Path.Src ""
              , dest = Script.Path.Tmp "unpacked"
              }
          
          -- Create lib directory
          , Script.mkdir "lib"
          
          -- Copy libraries
          , Script.Command.Copy
              { src = Script.Path.Tmp "unpacked/${paths.libPath}/."
              , dst = Script.Path.Out "lib/"
              , recursive = True
              }
          ]
          # includeCommands
          # [ -- Create lib64 symlink for compatibility
              Script.symlink "lib" "lib64"
            ]

-- | Standard dependencies for wheel packages
let wheelNativeBuildInputs : List Text =
      [ "autoPatchelfHook"
      , "unzip"
      ]

let wheelBuildInputs : List Text =
      [ "stdenv.cc.cc.lib"  -- libstdc++
      , "zlib"
      ]

-- =============================================================================
-- Concrete Packages
-- =============================================================================

let nccl : WheelPackage.Type =
      { pname = "nvidia-nccl"
      , version = "2.28.9"
      , src =
          { url = "https://pypi.nvidia.com/nvidia-nccl-cu13/nvidia_nccl_cu13-2.28.9-py3-none-manylinux_2_18_x86_64.whl"
          , sha256 = "sha256-5FU6MPNBlfP6HaAqbaPWM30o8gA5Q6oKPSR7vCX+/EI="
          }
      , paths =
          { libPath = "nvidia/nccl/lib"
          , includePath = Some "nvidia/nccl/include"
          }
      , description = "NVIDIA NCCL 2.28.9 (from PyPI)"
      , homepage = "https://developer.nvidia.com/nccl"
      }

let cudnn : WheelPackage.Type =
      { pname = "nvidia-cudnn"
      , version = "9.17.0.29"
      , src =
          { url = "https://pypi.nvidia.com/nvidia-cudnn-cu13/nvidia_cudnn_cu13-9.17.0.29-py3-none-manylinux_2_27_x86_64.whl"
          , sha256 = "sha256-Q0Uu8Jj0Q890hyvHj8zZon4af4NWelF5m/rsVgvP4Vo="
          }
      , paths =
          { libPath = "nvidia/cudnn/lib"
          , includePath = Some "nvidia/cudnn/include"
          }
      , description = "NVIDIA cuDNN 9.17.0.29 (from PyPI)"
      , homepage = "https://developer.nvidia.com/cudnn"
      }

let tensorrt : WheelPackage.Type =
      { pname = "nvidia-tensorrt"
      , version = "10.14.1.48"
      , src =
          { url = "https://pypi.nvidia.com/tensorrt-cu13-libs/tensorrt_cu13_libs-10.14.1.48-py2.py3-none-manylinux_2_28_x86_64.whl"
          , sha256 = "sha256-k8SI67WjD/g+pTYD54GAFN5bkyj7JJZZY9I2gUB2UHY="
          }
      , paths =
          { libPath = "tensorrt_libs"
          , includePath = None Text
          }
      , description = "NVIDIA TensorRT 10.14.1.48 (from PyPI)"
      , homepage = "https://developer.nvidia.com/tensorrt"
      }

let cutensor : WheelPackage.Type =
      { pname = "nvidia-cutensor"
      , version = "2.4.1"
      , src =
          { url = "https://pypi.nvidia.com/cutensor-cu13/cutensor_cu13-2.4.1-py3-none-manylinux2014_x86_64.whl"
          , sha256 = "sha256-Hz1oTgSVOuRJI7ZzotQVbdmaghQAxC/ocqqF+PFmtyg="
          }
      , paths =
          { libPath = "cutensor/lib"
          , includePath = Some "cutensor/include"
          }
      , description = "NVIDIA cuTensor 2.4.1 (from PyPI)"
      , homepage = "https://developer.nvidia.com/cutensor"
      }

let cusparselt : WheelPackage.Type =
      { pname = "nvidia-cusparselt"
      , version = "0.8.1"
      , src =
          { url = "https://pypi.nvidia.com/nvidia-cusparselt-cu13/nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_x86_64.whl"
          , sha256 = "sha256-eGzodWjDA/rbWvzHEC1FTNMEDXX2+GJvXbRg0YcfTdA="
          }
      , paths =
          { libPath = "nvidia/cusparselt/lib"
          , includePath = Some "nvidia/cusparselt/include"
          }
      , description = "NVIDIA cuSPARSELt 0.8.1 (from PyPI)"
      , homepage = "https://developer.nvidia.com/cusparselt"
      }

-- =============================================================================
-- Exports
-- =============================================================================

in  { WheelSrc
    , WheelPaths
    , WheelPackage
    , mkWheelInstallScript
    , wheelNativeBuildInputs
    , wheelBuildInputs
    -- Concrete packages
    , nccl
    , cudnn
    , tensorrt
    , cutensor
    , cusparselt
    }
