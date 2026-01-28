runHook preConfigure

# Write .buckconfig.local with Nix store paths
cp "$buckconfigFile" .buckconfig.local
chmod 644 .buckconfig.local

# Link prelude if not present
if [ ! -d "prelude" ] && [ ! -L "prelude" ]; then
  ln -sf "$prelude" prelude
fi

runHook postConfigure
