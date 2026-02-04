{ pkgs }:
let
  # This file demonstrates type inference.
  # Run: nix run .#nix-compile -- fmt examples/nix-compile/typed.nix

  # Simple values
  port = 8080;
  host = "localhost";
  debug = true;

  # Paths
  src = ./.;

  # Lists
  packages = [
    pkgs.curl
    pkgs.jq
    pkgs.git
  ];

  # Attribute set
  config = {
    server = {
      inherit port host;
      ssl = false;
    };
    db = {
      path = "/var/lib/db";
      backup = true;
    };
  };

  # Function
  mkServer =
    { name, port }:
    {
      inherit name port;
      type = "server";
    };

  # Function call
  myServer = mkServer {
    name = "production";
    inherit port;
  };
in
{
  inherit config packages myServer;
}
