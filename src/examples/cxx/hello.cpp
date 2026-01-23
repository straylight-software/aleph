// examples/cxx/hello.cpp
// Simple C++23 test to verify toolchain.

#include <cstdio>
#include <string_view>

namespace straylight::examples {

constexpr std::string_view greeting = "straylight toolchain operational";

auto main_impl() -> int {
  std::printf("%.*s\n", static_cast<int>(greeting.size()), greeting.data());
  return 0;
}

} // namespace straylight::examples

auto main() -> int { return straylight::examples::main_impl(); }
