# nix/modules/nixos/nv-driver.nix
#
# NixOS module for NVIDIA driver configuration (:: NixOSModule)
#
# The directory is the kind signature.
#
{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.aleph.nv;
in
{
  _class = "nixos";

  options.aleph.nv = {
    enable = lib.mkEnableOption "NVIDIA driver via aleph";

    open = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Use open-source kernel modules (Turing+)";
    };

    package = lib.mkOption {
      type = lib.types.package;
      default = config.boot.kernelPackages.nvidiaPackages.beta;
      description = "NVIDIA driver package";
    };
  };

  config = lib.mkIf cfg.enable {

    hardware.nvidia = {
      inherit (cfg) open;
      inherit (cfg) package;
      modesetting.enable = true;
      "powerManagement".enable = false;
      "nvidiaSettings" = false;
    };

    hardware.graphics.enable = true;

    services.xserver."videoDrivers" = [ "nvidia" ];

    environment."systemPackages" = with pkgs; [
      "nvtopPackages".nvidia
      pciutils
    ];
  };
}
