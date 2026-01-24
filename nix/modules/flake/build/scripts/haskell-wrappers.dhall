-- nix/modules/flake/build/scripts/haskell-wrappers.dhall
--
-- Shell hook for copying Haskell wrapper scripts
-- Environment variables are injected by render.dhall-with-vars

let scriptsDir : Text = env:SCRIPTS_DIR as Text

in ''
cp ${scriptsDir}/ghc-wrapper.bash bin/ghc
chmod +x bin/ghc

cp ${scriptsDir}/ghc-pkg-wrapper.bash bin/ghc-pkg
chmod +x bin/ghc-pkg

cp ${scriptsDir}/haddock-wrapper.bash bin/haddock
chmod +x bin/haddock

# Generate hie.yaml for HLS if not exists
if [ ! -e "hie.yaml" ]; then
	cp ${scriptsDir}/hie.yaml.template hie.yaml
	echo "Generated hie.yaml for HLS"
fi
''
