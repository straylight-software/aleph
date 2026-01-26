// Test that actually requires network (curl fetch during build)
#include <cstdio>
#include <cstdlib>

int main() {
    // This is a compile-time test - no runtime fetch
    printf("Built successfully!\n");
    return 0;
}
