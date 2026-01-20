# nix/overlays/container/pep503.nix
#
# PEP 503 simple index generation
#
{ final }:
{
  # Generate a PEP 503 simple index from a wheel directory
  #
  # Example:
  #   mk-simple-index {
  #     name = "pytorch-index";
  #     wheel-dir = ./wheels;
  #   }
  #
  mk-simple-index =
    { name, wheel-dir }:
    final.runCommand name { } ''
      mkdir -p $out/simple

      echo '<!DOCTYPE html><html><head><title>Simple Index</title></head><body><h1>Simple Index</h1>' > $out/simple/index.html

      for whl in ${wheel-dir}/*.whl; do
        [ -f "$whl" ] || continue

        filename=$(basename "$whl")
        # PEP 503 normalization: lowercase, underscores/dots -> hyphens
        pkg=$(echo "$filename" | sed 's/-[0-9].*//' | tr '[:upper:]_.' '[:lower:]--')

        mkdir -p "$out/simple/$pkg"
        cp "$whl" "$out/simple/$pkg/"

        sha=$(sha256sum "$whl" | cut -d' ' -f1)
        echo "<!DOCTYPE html><html><head><title>$pkg</title></head><body><h1>$pkg</h1>" > "$out/simple/$pkg/index.html"
        echo "<a href=\"$filename#sha256=$sha\">$filename</a><br/>" >> "$out/simple/$pkg/index.html"
        echo '</body></html>' >> "$out/simple/$pkg/index.html"

        echo "<a href=\"$pkg/\">$pkg</a><br/>" >> $out/simple/index.html
      done

      echo '</body></html>' >> $out/simple/index.html
    '';
}
