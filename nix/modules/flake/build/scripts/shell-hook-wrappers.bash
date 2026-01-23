# Generate bin/ wrappers for Buck2 toolchains
mkdir -p bin
@haskellWrappers@
@leanWrappers@
@cxxWrappers@
echo "Generated bin/ wrappers for Buck2 toolchains"
