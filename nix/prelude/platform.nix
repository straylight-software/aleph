# nix/prelude/platform.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                               // platform //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     Home was BAMA, the Sprawl, the Boston-Loss Angeles Metropolitan
#     Axis. Program a map to display frequency of data exchange, every
#     thousand megabytes a single pixel on a very large screen. Manhattan
#     and Atlanta burn solid white. Then they start to pulse, the rate
#     of traffic threatening to overload your simulation. Your map is
#     about to go nova. Cool it down. Up your scale. Each pixel a million
#     megabytes. At a hundred million megabytes per second, you begin to
#     make out certain blocks in midtown Manhattan, outlines of hundred-
#     year-old industrial parks ringing the old core of Atlanta...
#
#                                                         — Neuromancer
#
# Platform detection. Where are we running? What can we do here?
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ final, ... }:
let
  inherit (final.stdenv.hostPlatform) system;

  # ──────────────────────────────────────────────────────────────────────────
  #                           // known platforms //
  # ──────────────────────────────────────────────────────────────────────────

  platforms = {
    linux-x86-64 = {
      os = "linux";
      arch = "x86_64";
      system = "x86_64-linux";
    };
    linux-sbsa = {
      os = "linux";
      arch = "aarch64";
      system = "aarch64-linux";
      variant = "sbsa";
    };
    linux-aarch64 = {
      os = "linux";
      arch = "aarch64";
      system = "aarch64-linux";
    };
    darwin-aarch64 = {
      os = "darwin";
      arch = "aarch64";
      system = "aarch64-darwin";
    };
    darwin-x86-64 = {
      os = "darwin";
      arch = "x86_64";
      system = "x86_64-darwin";
    };
  };

  # ──────────────────────────────────────────────────────────────────────────
  #                               // current //
  # ──────────────────────────────────────────────────────────────────────────

  current =
    {
      "x86_64-linux" = platforms.linux-x86-64;
      "aarch64-linux" = platforms.linux-aarch64;
      "aarch64-darwin" = platforms.darwin-aarch64;
      "x86_64-darwin" = platforms.darwin-x86-64;
    }
    .${system} or (throw "unsupported system: ${system}");

in
platforms
// {
  inherit current;
  is-linux = current.os == "linux";
  is-darwin = current.os == "darwin";
  is-x86 = current.arch == "x86_64";
  is-arm = current.arch == "aarch64";
}
