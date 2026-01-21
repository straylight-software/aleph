# nix/packages/lre-start.nix
#
# Script to start NativeLink for Local Remote Execution
#
{
  lib,
  writeShellApplication,
  nativelink,
  procps,
  coreutils,
}:

writeShellApplication {
  name = "lre-start";

  runtimeInputs = [
    nativelink
    procps
    coreutils
  ];

  text = builtins.readFile ../scripts/lre-start.sh;

  meta = {
    description = "Start NativeLink for Local Remote Execution";
    mainProgram = "lre-start";
  };
}
