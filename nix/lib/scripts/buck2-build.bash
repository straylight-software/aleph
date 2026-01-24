runHook preBuild

export HOME=$TMPDIR
buck2 build "$buck2Target" --show-full-output 2>&1 | tee build.log

runHook postBuild
