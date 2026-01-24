mkdir -p $out/simple

echo '<!DOCTYPE html><html><head><title>Simple Index</title></head><body><h1>Simple Index</h1>' >$out/simple/index.html

for whl in @wheelDir@/*.whl; do
	[ -f "$whl" ] || continue

	filename=$(basename "$whl")
	# PEP 503 normalization: lowercase, underscores/dots -> hyphens
	pkg=$(echo "$filename" | sed 's/-[0-9].*//' | tr '[:upper:]_.' '[:lower:]--')

	mkdir -p "$out/simple/$pkg"
	cp "$whl" "$out/simple/$pkg/"

	sha=$(sha256sum "$whl" | cut -d' ' -f1)
	echo "<!DOCTYPE html><html><head><title>$pkg</title></head><body><h1>$pkg</h1>" >"$out/simple/$pkg/index.html"
	echo "<a href=\"$filename#sha256=$sha\">$filename</a><br/>" >>"$out/simple/$pkg/index.html"
	echo '</body></html>' >>"$out/simple/$pkg/index.html"

	echo "<a href=\"$pkg/\">$pkg</a><br/>" >>$out/simple/index.html
done

echo '</body></html>' >>$out/simple/index.html
