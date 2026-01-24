runHook preConfigure

# Write .buckconfig.local with Nix store paths
cp "$buckconfigFile" .buckconfig.local

# Link prelude if needed
if [ ! -d "prelude" ] && [ ! -L "prelude" ]; then
	ln -s "$buck2Prelude" prelude
fi

runHook postConfigure
