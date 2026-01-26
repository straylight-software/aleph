#!/bin/bash
# A build script that fetches from the network
set -e

echo ":: Fetching from network..."
curl -sL https://example.com -o example.html
curl -sL https://httpbin.org/uuid -o uuid.json

echo ":: Downloaded:"
wc -c example.html uuid.json

echo ":: Build complete"
echo "fetched at $(date -u +%Y-%m-%dT%H:%M:%SZ)" > fetch-test
