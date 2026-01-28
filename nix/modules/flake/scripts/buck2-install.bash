runHook preInstall

mkdir -p "$out/bin"

# Find and copy the output
if [ -n "$buck2Output" ]; then
  cp "buck-out/v2/gen/$buck2Output" "$out/bin/"
else
  # Auto-detect: copy executables from buck-out
  find buck-out/v2/gen -type f -executable -name "$targetName*" | head -1 | xargs -I{} cp {} "$out/bin/"
fi

runHook postInstall
