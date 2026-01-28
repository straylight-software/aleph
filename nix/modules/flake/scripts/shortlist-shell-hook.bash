# Add shortlist section to .buckconfig.local
if [ -f ".buckconfig.local" ]; then
	if ! grep -q "\\[shortlist\\]" .buckconfig.local 2>/dev/null; then
		cat "@shortlistFile@" >>.buckconfig.local
		echo "Added [shortlist] section to .buckconfig.local"
	fi
fi
