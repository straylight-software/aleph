#include <stdio.h>
#include <string.h>
#include <zlib.h>

int main(void) {
  const char *input = "Hello from armitage! This will be compressed.";
  char compressed[256];
  char decompressed[256];
  uLongf comp_len = sizeof(compressed);
  uLongf decomp_len = sizeof(decompressed);

  // Compress
  // cppcheck-suppress dangerousTypeCast ; Bytef* is the correct type for zlib API
  if (compress(reinterpret_cast<Bytef *>(compressed), &comp_len,
               reinterpret_cast<const Bytef *>(input), strlen(input) + 1) !=
      Z_OK) {
    fprintf(stderr, "compression failed\n");
    return 1;
  }

  printf("Original:   %zu bytes\n", strlen(input) + 1);
  printf("Compressed: %lu bytes\n", comp_len);

  // Decompress
  // cppcheck-suppress dangerousTypeCast ; Bytef* is the correct type for zlib API
  if (uncompress(reinterpret_cast<Bytef *>(decompressed), &decomp_len,
                 reinterpret_cast<const Bytef *>(compressed), comp_len) !=
      Z_OK) {
    fprintf(stderr, "decompression failed\n");
    return 1;
  }

  printf("Result:     %s\n", decompressed);
  printf("\nzlib version: %s\n", zlibVersion());

  return 0;
}
