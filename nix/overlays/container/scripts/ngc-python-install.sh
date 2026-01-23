runHook preInstall

mkdir -p $out/{bin,lib}

# Find and copy Python interpreter
PYTHON_BIN=$(find $src -path "*/bin/python@pythonVersion@" -type f | head -1)
if [ -z "$PYTHON_BIN" ]; then
	PYTHON_BIN=$(find $src -path "*/bin/python3" -type f | head -1)
fi

if [ -n "$PYTHON_BIN" ]; then
	cp "$PYTHON_BIN" $out/bin/python3
	chmod +x $out/bin/python3
	ln -s python3 $out/bin/python
fi

# Copy Python standard library
PYTHON_LIB=$(find $src -type d -name "python@pythonVersion@" -path "*/lib/*" | head -1)
if [ -n "$PYTHON_LIB" ]; then
	cp -a "$PYTHON_LIB" $out/lib/
fi

# Copy site-packages from all locations
mkdir -p $out/lib/python@pythonVersion@/site-packages
for site in \
	$src/usr/lib/python3/dist-packages \
	$src/usr/local/lib/python@pythonVersion@/dist-packages \
	$src/usr/lib/python@pythonVersion@/site-packages; do
	if [ -d "$site" ]; then
		cp -a "$site"/* $out/lib/python@pythonVersion@/site-packages/ 2>/dev/null || true
	fi
done

# Copy container's native libs that Python modules depend on
mkdir -p $out/lib/native
for lib_dir in $src/usr/lib/x86_64-linux-gnu $src/usr/local/lib; do
	if [ -d "$lib_dir" ]; then
		find "$lib_dir" -name "*.so*" -type f 2>/dev/null | while read -r f; do
			cp -a "$f" $out/lib/native/ 2>/dev/null || true
		done
	fi
done

runHook postInstall
