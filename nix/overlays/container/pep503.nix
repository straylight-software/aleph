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
    let
      to-string = builtins.toString;
      script = builtins.replaceStrings [ "@wheelDir@" ] [ (to-string wheel-dir) ] (
        builtins.readFile ./scripts/pep503-index.sh
      );
    in
    final.runCommand name { } script;
}
