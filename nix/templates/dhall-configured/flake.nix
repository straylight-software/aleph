{
  description = "Dhall-configured infrastructure (agenix users/machines)";

  inputs = {
    aleph.url = "github:straylight-software/aleph";
    nixpkgs.follows = "aleph/nixpkgs";
    flake-parts.follows = "aleph/flake-parts";
    agenix.url = "github:ryantm/agenix";
  };

  outputs =
    inputs@{
      flake-parts,
      aleph,
      agenix,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        aleph.modules.flake.nixpkgs
        aleph.modules.flake.prelude
      ];

      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      aleph.nixpkgs.allow-unfree = true;

      "perSystem" =
        { prelude, pkgs, ... }:
        let
          # Load and type-check configuration from Dhall
          config = pkgs.dhallToNix (builtins.readFile ./config.dhall);

          # Validate config structure
          validated = prelude.typed.define {
            options.users = prelude.opt.list-of prelude.typed.types.str {
              description = "List of user names";
            };
            options.machines = prelude.opt.attrs-of (prelude.typed.types.attrs prelude.typed.types.str) {
              description = "Machine configurations";
            };
          } config;
        in
        {
          # Generate agenix secrets rules from config
          packages.secrets-rules = prelude.render.json "secrets.json" {
            inherit (validated) users machines;
          };

          # Check that validates config at eval time
          checks.config-valid = pkgs.runCommand "config-valid" { } ''
            echo "Config validated:"
            echo "  Users: ${builtins.toString validated.users}"
            echo "  Machines: ${builtins.toString (builtins.attrNames validated.machines)}"
            touch $out
          '';
        };

      # Export NixOS modules for each machine
      flake."nixosModules" = {
        agenix-secrets = {
          imports = [ agenix.nixosModules.default ];
          # Secrets configuration would go here
        };
      };
    };
}
