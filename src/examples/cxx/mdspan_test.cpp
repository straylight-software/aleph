// examples/cxx/mdspan_test.cpp
//
// c++23 std::mdspan verification - the whole point of gcc15
//
// verifies:
//   - std::mdspan compiles and works
//   - std::extents, std::dextents work
//   - layout policies work
//   - submdspan works (c++26 but gcc15 has it)

#include <array>
#include <cstdio>
#include <mdspan>
#include <numeric>
#include <span>

namespace straylight::examples {

// ════════════════════════════════════════════════════════════════════════════════
// basic mdspan test - 2d matrix view
// ════════════════════════════════════════════════════════════════════════════════

auto test_basic_mdspan() -> bool {
  // 3x4 matrix stored in row-major order
  std::array<float, 12> data{};
  std::iota(data.begin(), data.end(), 0.0f); // 0, 1, 2, ..., 11

  // create mdspan view
  std::mdspan matrix{data.data(), std::extents<std::size_t, 3, 4>{}};

  // verify dimensions
  if (matrix.extent(0) != 3 || matrix.extent(1) != 4) {
    std::printf("mdspan: extent mismatch\n");
    return false;
  }

  // verify element access - row 1, col 2 should be 1*4 + 2 = 6
  if (matrix[1, 2] != 6.0f) {
    std::printf("mdspan: element access failed, got %f expected 6.0\n",
                static_cast<double>(matrix[1, 2]));
    return false;
  }

  // verify we can modify through the view
  matrix[2, 3] = 99.0f;
  if (data[11] != 99.0f) {
    std::printf("mdspan: modification failed\n");
    return false;
  }

  return true;
}

// ════════════════════════════════════════════════════════════════════════════════
// dynamic extents test
// ════════════════════════════════════════════════════════════════════════════════

auto test_dynamic_extents() -> bool {
  std::array<int, 24> data{};
  std::iota(data.begin(), data.end(), 0);

  // 2x3x4 tensor with all dynamic extents
  std::mdspan tensor{data.data(), std::dextents<std::size_t, 3>{2, 3, 4}};

  if (tensor.extent(0) != 2 || tensor.extent(1) != 3 || tensor.extent(2) != 4) {
    std::printf("mdspan: dynamic extent mismatch\n");
    return false;
  }

  // element [1][2][3] = 1*12 + 2*4 + 3 = 23
  if (tensor[1, 2, 3] != 23) {
    std::printf("mdspan: 3d access failed, got %d expected 23\n",
                tensor[1, 2, 3]);
    return false;
  }

  return true;
}

// ════════════════════════════════════════════════════════════════════════════════
// layout stride test - column major
// ════════════════════════════════════════════════════════════════════════════════

auto test_layout_stride() -> bool {
  std::array<float, 6> data{};
  std::iota(data.begin(), data.end(), 0.0f);

  // 2x3 matrix in column-major order
  using col_major = std::layout_left;
  std::mdspan<float, std::extents<std::size_t, 2, 3>, col_major> matrix{
      data.data()};

  // in column major, [1][0] should be element 1 (second element of first
  // column)
  if (matrix[1, 0] != 1.0f) {
    std::printf("mdspan: column major layout failed\n");
    return false;
  }

  // [0][1] should be element 2 (first element of second column)
  if (matrix[0, 1] != 2.0f) {
    std::printf("mdspan: column major [0,1] failed, got %f\n",
                static_cast<double>(matrix[0, 1]));
    return false;
  }

  return true;
}

// ════════════════════════════════════════════════════════════════════════════════
// mixed static/dynamic extents
// ════════════════════════════════════════════════════════════════════════════════

auto test_mixed_extents() -> bool {
  std::array<double, 32> data{};
  std::iota(data.begin(), data.end(), 0.0);

  // batch of 4x8 matrices where batch size is dynamic
  // extents<size_t, dynamic_extent, 4, 8> means [?, 4, 8]
  using batch_matrix_extents =
      std::extents<std::size_t, std::dynamic_extent, 4, 8>;
  std::mdspan batch{data.data(), batch_matrix_extents{1}}; // 1 batch

  if (batch.extent(0) != 1 || batch.extent(1) != 4 || batch.extent(2) != 8) {
    std::printf("mdspan: mixed extents failed\n");
    return false;
  }

  // [0][2][3] = 0*32 + 2*8 + 3 = 19
  if (batch[0, 2, 3] != 19.0) {
    std::printf("mdspan: mixed extent access failed\n");
    return false;
  }

  return true;
}

// ════════════════════════════════════════════════════════════════════════════════
// submdspan test (c++26 feature, but gcc15 has it)
// ════════════════════════════════════════════════════════════════════════════════

auto test_submdspan() -> bool {
  std::array<int, 12> data{};
  std::iota(data.begin(), data.end(), 0);

  // 3x4 matrix
  std::mdspan matrix{data.data(), std::extents<std::size_t, 3, 4>{}};

  // extract row 1 as a 1d span
  auto row1 = std::submdspan(matrix, 1, std::full_extent);

  if (row1.extent(0) != 4) {
    std::printf("submdspan: row extent wrong\n");
    return false;
  }

  // row 1 starts at element 4
  if (row1[0] != 4 || row1[3] != 7) {
    std::printf("submdspan: row elements wrong\n");
    return false;
  }

  // extract column 2 as strided 1d span
  auto col2 = std::submdspan(matrix, std::full_extent, 2);

  if (col2.extent(0) != 3) {
    std::printf("submdspan: column extent wrong\n");
    return false;
  }

  // column 2 elements: 2, 6, 10
  if (col2[0] != 2 || col2[1] != 6 || col2[2] != 10) {
    std::printf("submdspan: column elements wrong: %d %d %d\n", col2[0],
                col2[1], col2[2]);
    return false;
  }

  return true;
}

auto implementation() -> int {
  int failures = 0;

  std::printf("mdspan tests (gcc15 libstdc++ c++23):\n");

  if (test_basic_mdspan()) {
    std::printf("  basic_mdspan: pass\n");
  } else {
    std::printf("  basic_mdspan: FAIL\n");
    failures++;
  }

  if (test_dynamic_extents()) {
    std::printf("  dynamic_extents: pass\n");
  } else {
    std::printf("  dynamic_extents: FAIL\n");
    failures++;
  }

  if (test_layout_stride()) {
    std::printf("  layout_stride: pass\n");
  } else {
    std::printf("  layout_stride: FAIL\n");
    failures++;
  }

  if (test_mixed_extents()) {
    std::printf("  mixed_extents: pass\n");
  } else {
    std::printf("  mixed_extents: FAIL\n");
    failures++;
  }

  if (test_submdspan()) {
    std::printf("  submdspan: pass\n");
  } else {
    std::printf("  submdspan: FAIL\n");
    failures++;
  }

  if (failures == 0) {
    std::printf("all mdspan tests passed\n");
    return 0;
  } else {
    std::printf("%d mdspan tests FAILED\n", failures);
    return 1;
  }
}

} // namespace straylight::examples

auto main(int argc, char* argv[]) -> int { return straylight::examples::implementation(); }
