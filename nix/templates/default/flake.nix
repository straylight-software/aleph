{
  description = "Project powered by aleph-naught";

  inputs = {
    aleph-naught.url = "github:straylight-software/aleph-naught";
    nixpkgs.follows = "aleph-naught/nixpkgs";
    flake-parts.follows = "aleph-naught/flake-parts";
  };

  outputs =
    inputs@{ flake-parts, aleph-naught, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ aleph-naught.modules.flake.default ];
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];

      aleph-naught = {
        formatter.enable = true;
        devshell.enable = true;
        nixpkgs.allow-unfree = true;
      };

      "perSystem" =
        { config, pkgs, ... }:
        let
          # The Weyl Prelude is available via config.straylight.prelude
          inherit (config.straylight) prelude;
          inherit (prelude)
            stdenv
            fetch
            render
            license
            ;
        in
        {
          packages.default = pkgs.hello;

          # Example using prelude
          # packages.my-tool = stdenv.default {
          #   pname = "my-tool";
          #   version = "0.1.0";
          #
          #   src = fetch.github {
          #     owner = "me";
          #     repo = "my-tool";
          #     rev = "v0.1.0";
          #     hash = "sha256-...";
          #   };
          #
          #   meta = {
          #     description = "My tool";
          #     license = license.mit;
          #   };
          # };
        };
    };
}
