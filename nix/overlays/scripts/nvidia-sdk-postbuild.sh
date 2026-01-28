# lib64 -> lib symlink (NVIDIA tools expect lib64)
if [ ! -e $out/lib64 ]; then
  ln -s lib $out/lib64
fi

# CUDA 13 compat: texture_fetch_functions.h was renamed/removed
# clang's __clang_cuda_runtime_wrapper.h still expects it
if [ ! -e $out/include/texture_fetch_functions.h ] && [ -e $out/include/texture_indirect_functions.h ]; then
  ln -s texture_indirect_functions.h $out/include/texture_fetch_functions.h
fi

# CCCL compat: CUTLASS 4.x expects cccl/cuda/std/ but cuda_cccl provides cuda/std/
if [ ! -e $out/include/cccl ] && [ -e $out/include/cuda/std ]; then
  mkdir -p $out/include/cccl
  ln -s ../cuda $out/include/cccl/cuda
  ln -s ../cub $out/include/cccl/cub
  ln -s ../thrust $out/include/cccl/thrust
  ln -s ../nv $out/include/cccl/nv
fi
