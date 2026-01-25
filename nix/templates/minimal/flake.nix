{
  description = "Minimal project with aleph-naught nixpkgs";

  inputs = {
    aleph-naught.url = "github:straylight-software/aleph-naught";
    nixpkgs.follows = "aleph-naught/nixpkgs";
    flake-parts.follows = "aleph-naught/flake-parts";
  };

  outputs =
    inputs@{ flake-parts, aleph-naught, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ aleph-naught.modules.flake.nixpkgs ];
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];

      aleph-naught.nixpkgs.allow-unfree = true;

      "perSystem" =
        { pkgs, ... }:
        {
          packages.default = pkgs.hello;
        };
    };
}
