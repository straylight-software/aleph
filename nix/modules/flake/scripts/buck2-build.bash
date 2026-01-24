runHook preBuild

buck2 build "$buck2Target" --show-full-output

runHook postBuild
