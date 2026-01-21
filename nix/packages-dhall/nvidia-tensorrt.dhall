-- nvidia-tensorrt: NVIDIA TensorRT inference library
-- Redistrib package: custom builder for unpack + patch

let Drv = ../Drv/Prelude.dhall

in  \(host : Drv.Triple) ->
        Drv.defaults
      //  { name = "nvidia-tensorrt"
          , version = "10.9.0.34"
          , src =
              Drv.url
                "https://developer.download.nvidia.com/compute/tensorrt/redist/tensorrt/linux-x86_64/tensorrt-linux-x86_64-10.9.0.34-cuda-12.8.tar.gz"
                "sha256-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX="
          , deps = [ "nvidia-cudnn", "nvidia-cublas", "patchelf" ]
          , build = Drv.custom "nvidia/tensorrt.hs"
          , host
          , checks = [ "std/has-file lib/libnvinfer.so" ]
          }
