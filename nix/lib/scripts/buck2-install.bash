runHook preInstall

mkdir -p "$out/bin"

# Find the output binary
OUTPUT_PATH=$(grep "^$buck2Target" build.log | awk '{print $2}' | head -1)
if [ -n "$OUTPUT_PATH" ] && [ -f "$OUTPUT_PATH" ]; then
  cp "$OUTPUT_PATH" "$out/bin/$outputName"
else
  # Fallback: search buck-out
  find buck-out -type f -executable -name "$outputName*" 2>/dev/null | head -1 | xargs -I{} cp {} "$out/bin/" || true
fi

runHook postInstall
