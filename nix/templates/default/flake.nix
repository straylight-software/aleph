{
  description = "Project powered by aleph";

  inputs = {
    aleph.url = "github:straylight-software/aleph";
    nixpkgs.follows = "aleph/nixpkgs";
    flake-parts.follows = "aleph/flake-parts";
  };

  outputs =
    inputs@{ flake-parts, aleph, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ aleph.modules.flake.default ];
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];

      aleph = {
        formatter.enable = true;
        devshell.enable = true;
        nixpkgs.allow-unfree = true;
      };

      "perSystem" =
        { config, pkgs, ... }:
        let
          # The Aleph Prelude is available via config.aleph.prelude
          inherit (config.aleph) prelude;
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
