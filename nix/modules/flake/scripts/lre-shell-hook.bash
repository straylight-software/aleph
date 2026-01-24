# Append RE configuration to .buckconfig.local
# (build.nix shellHook creates .buckconfig.local, we append to it)
if [ -f ".buckconfig.local" ]; then
	# Check if RE config already present
	if ! grep -q "buck2_re_client" .buckconfig.local 2>/dev/null; then
		cat "@buckconfigReFile@" >>.buckconfig.local
		echo "Appended NativeLink RE config to .buckconfig.local"
	fi
else
	echo "Warning: .buckconfig.local not found, LRE config not added"
fi
