#include <mdspan>
#include <vector>
#include <cassert>

int main() {
  // Test std::mdspan is available (not std::experimental::mdspan)
  std::vector<int> data = {1, 2, 3, 4, 5, 6};

  // Create a 2D mdspan view of the data (2x3 matrix)
  std::mdspan<int, std::dextents<size_t, 2>> mat(
    data.data(),
    2, 3
  );

  // Verify dimensions
  assert(mat.extent(0) == 2);
  assert(mat.extent(1) == 3);

  // Verify data access (parentheses needed to avoid comma-in-macro issue)
  assert((mat[0, 0] == 1));
  assert((mat[0, 1] == 2));
  assert((mat[0, 2] == 3));
  assert((mat[1, 0] == 4));
  assert((mat[1, 1] == 5));
  assert((mat[1, 2] == 6));

  return 0;
}
