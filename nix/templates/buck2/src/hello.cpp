// C++ Hello World
// Built with LLVM 22 from Nix via Buck2
//
// Build: buck2 build //src:hello
// Run:   buck2 run //src:hello

#include <cstdio>

int main() {
  std::printf("Hello from Buck2 + LLVM 22!\n");
  std::printf("  Compiler: clang %d.%d.%d\n", __clang_major__, __clang_minor__,
              __clang_patchlevel__);
  std::printf("  Standard: C++%ld\n", __cplusplus / 100 % 100);
  return 0;
}
