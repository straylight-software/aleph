# Generate .buckconfig.local with Nix store paths
rm -f .buckconfig.local 2>/dev/null || true
cp @buckconfigLocal@ .buckconfig.local
chmod 644 .buckconfig.local
echo "Generated .buckconfig.local with Nix store paths"
