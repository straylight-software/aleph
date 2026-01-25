{
  description = "NVIDIA/ML project powered by aleph";

  inputs = {
    aleph.url = "github:straylight-software/aleph";
    nixpkgs.follows = "aleph/nixpkgs";
    flake-parts.follows = "aleph/flake-parts";
    systems.url = "github:nix-systems/default-linux";
  };

  outputs =
    inputs@{ flake-parts, aleph, ... }:
    let
      # Memoized nixpkgs instances - one import per system, not per module.
      # See: https://zimbatm.com/notes/1000-instances-of-nixpkgs
      nixpkgs-instances = inputs.nixpkgs.lib.genAttrs (import inputs.systems) (
        system:
        import inputs.nixpkgs {
          inherit system;
          overlays = [
            aleph.overlays.default
          ];
          config = {
            "cudaSupport" = true;
            "cudaCapabilities" = [ "12.0" ]; # Blackwell
            "allowUnfree" = true;
          };
        }
      );
    in
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ aleph.modules.flake.default ];
      systems = import inputs.systems;

      aleph = {
        formatter.enable = true;
        devshell = {
          enable = true;
          nv.enable = true;
        };
      };

      # Expose memoized instances for downstream
      flake.nixpkgs = nixpkgs-instances;

      "perSystem" =
        { config, system, ... }:
        let
          # Use memoized nixpkgs instance
          pkgs = nixpkgs-instances.${system};

          # The Aleph Prelude is available via config.aleph.prelude
          inherit (config.aleph) prelude;
        in
        {
          # Override _module.args.pkgs with our memoized instance
          _module.args.pkgs = pkgs;
          "legacyPackages" = pkgs;

          packages.default = pkgs.hello;

          # Example NVIDIA kernel using prelude
          # packages.my-kernel = prelude.cpp.nvidia.kernel {
          #   pname = "my-kernel";
          #   version = "0.1.0";
          #   target-gpu = prelude.gpu.sm_120;  # Blackwell
          #
          #   src = prelude.fetch.github {
          #     owner = "me";
          #     repo = "my-kernel";
          #     rev = "v0.1.0";
          #     hash = "sha256-...";
          #   };
          #
          #   meta = {
          #     description = "My NVIDIA kernel";
          #     license = prelude.license.mit;
          #   };
          # };
        };
    };
}
