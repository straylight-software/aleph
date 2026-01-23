# Link buck2-prelude to nix/build/prelude
if [ ! -e "nix/build/prelude/prelude.bzl" ]; then
	echo "Linking buck2-prelude..."
	rm -rf nix/build/prelude
	mkdir -p nix/build
	ln -sf @preludeSrc@ nix/build/prelude
	echo "Linked @preludeSrc@ -> nix/build/prelude"
fi

# Link toolchains to nix/build/toolchains
if [ ! -e "nix/build/toolchains/cxx.bzl" ]; then
	echo "Linking buck2-toolchains..."
	rm -rf nix/build/toolchains
	mkdir -p nix/build
	ln -sf @toolchainsSrc@ nix/build/toolchains
	echo "Linked @toolchainsSrc@ -> nix/build/toolchains"
fi
