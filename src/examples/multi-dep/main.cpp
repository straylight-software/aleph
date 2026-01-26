// Test multiple nix dependencies
// Uses: zlib, openssl, curl (curl depends on both)

#include <stdio.h>
#include <string.h>
#include <zlib.h>
#include <openssl/sha.h>
#include <curl/curl.h>

// Callback for curl - just count bytes
static size_t write_callback(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t *total = (size_t *)userp;
    *total += size * nmemb;
    return size * nmemb;
}

int main(void) {
    printf("=== Multi-dependency test ===\n\n");

    // 1. zlib: compress something
    printf("1. zlib %s\n", zlibVersion());
    const char *input = "Hello from multi-dep test!";
    char compressed[256];
    uLongf comp_len = sizeof(compressed);
    if (compress((Bytef *)compressed, &comp_len, (const Bytef *)input, strlen(input) + 1) == Z_OK) {
        printf("   Compressed %zu -> %lu bytes\n", strlen(input) + 1, comp_len);
    }

    // 2. openssl: hash something
    printf("\n2. OpenSSL %s\n", OPENSSL_VERSION_TEXT);
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256((const unsigned char *)input, strlen(input), hash);
    printf("   SHA256: ");
    for (int i = 0; i < 8; i++) printf("%02x", hash[i]);
    printf("...\n");

    // 3. curl: check version (don't actually fetch)
    printf("\n3. curl %s\n", curl_version());
    curl_global_init(CURL_GLOBAL_DEFAULT);
    CURL *curl = curl_easy_init();
    if (curl) {
        printf("   curl_easy_init() OK\n");
        curl_easy_cleanup(curl);
    }
    curl_global_cleanup();

    printf("\n=== All dependencies working ===\n");
    return 0;
}
