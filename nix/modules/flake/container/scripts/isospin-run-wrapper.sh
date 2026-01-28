mkdir -p $out/bin

makeWrapper @isospinRun@/bin/isospin-run $out/bin/isospin-run \
  --set CONFIG_FILE @dhallConfig@ \
  --prefix PATH : @binPath@
