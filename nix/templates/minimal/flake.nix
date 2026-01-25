{
  description = "Minimal project with aleph nixpkgs";

  inputs = {
    aleph.url = "github:straylight-software/aleph";
    nixpkgs.follows = "aleph/nixpkgs";
    flake-parts.follows = "aleph/flake-parts";
  };

  outputs =
    inputs@{ flake-parts, aleph, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ aleph.modules.flake.nixpkgs ];
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];

      aleph.nixpkgs.allow-unfree = true;

      "perSystem" =
        { pkgs, ... }:
        {
          packages.default = pkgs.hello;
        };
    };
}
