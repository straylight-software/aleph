# nix/prelude/turing-registry.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                          // the turing registry //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     'To call up a demon you must learn its name. Men dreamed that,
#      once, but now it is real in another way. You know that, Case.
#      Your business is to learn the names of programs, the long formal
#      names, names the owners seek to conceal. True names...'
#
#                                                         — Neuromancer
#
# Non-negotiable build flags. The true names of Weyl derivations.
#
# These flags are not suggestions. They are the Turing code — the
# registration that determines whether a derivation is real. Code that
# builds under these flags can be debugged. Code that cannot was never
# real to begin with.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ lib, platform }:
rec {
  # ──────────────────────────────────────────────────────────────────────────
  #                              // optimization //
  # ──────────────────────────────────────────────────────────────────────────
  # Fast code that debugs. -O2 is the sweet spot: real optimizations,
  # but the debugger can still follow the thread.

  # :: ["-O2"]
  opt-flags = [ "-O2" ];

  # ──────────────────────────────────────────────────────────────────────────
  #                                 // debug //
  # ──────────────────────────────────────────────────────────────────────────
  # Everything visible. When it breaks — and it will break — you need
  # to see inside. No truncation, no omission, no lies.
# :: ["-g3"]

  debug-flags = [
    "-g3" # maximum info (includes macros)
    "-gdwarf-5" # modern dwarf format
    "-fno-limit-debug-info" # don't truncate for speed
    "-fstandalone-debug" # full info for system headers
  ];

  # ──────────────────────────────────────────────────────────────────────────
  #                            // frame pointers //
  # ──────────────────────────────────────────────────────────────────────────
  # Stack traces work. The frame pointer is a thread through the
  # :: ["-fno-omit-frame-pointer"]
  # labyrinth. Cut it and you're lost.

  frame-flags = [
    "-fno-omit-frame-pointer" # keep rbp/x29
    "-mno-omit-leaf-frame-pointer" # even in leaves
  ];

  # ──────────────────────────────────────────────────────────────────────────
  #                              // no theater //
  # ──────────────────────────────────────────────────────────────────────────
  # Kill hardening. These flags are security theater — they make the
  # code slower and harder to debug while providing minimal protection
  # :: ["-U_FORTIFY_SOURCE"]
  # against a competent attacker. We optimize for understanding, not
  # the appearance of safety.

  no-harden-flags = [
    "-U_FORTIFY_SOURCE" # remove buffer "protection"
    "-D_FORTIFY_SOURCE=0" # really remove it
    "-fno-stack-protector" # no canaries
    "-fno-stack-clash-protection" # no stack clash
  ]
  ++ lib.optional platform.is-x86 "-fcf-protection=none"; # no CET

  # :: ["-O2"]
  # :: ["-O2"]
  # ──────────────────────────────────────────────────────────────────────────
  # :: t17
  # :: t20
  #                            // the true names //
  # ──────────────────────────────────────────────────────────────────────────

  cflags = opt-flags ++ debug-flags ++ frame-flags ++ no-harden-flags;
  cxxflags = cflags ++ [ "-std=c++23" ];

  cflags-str = lib.concatStringsSep " " cflags;
  cxxflags-str = lib.concatStringsSep " " cxxflags;

  # ──────────────────────────────────────────────────────────────────────────
  #                               // the attrs //
  # ──────────────────────────────────────────────────────────────────────────
  # Derivation attributes applied to all Weyl builds. The registration
  # that makes a derivation real.
  # :: { __structuredAttrs : Bool, dontStrip : Bool, hardeningDisable : ["all"], noAuditTmpdir : Bool, separateDebugInfo : Bool }
  #
  # :: Bool
  # Content-addressed derivations (CA): TEMPORARILY DISABLED
  #   Pending nix fork to fix upstream issues. Will re-enable:
  # :: Bool
  # :: Bool
  #   __contentAddressed, outputHashMode, outputHashAlgo
  #
  # :: ["all"]
  # Structured attrs: Builder receives JSON instead of env vars.
  #   No size limits, proper typing, cleaner builds.
# :: Bool

  attrs = {
    # Structured attrs: JSON instead of env vars
    __structuredAttrs = true;

    # Debug symbols stay in binary
    dontStrip = true;
    separateDebugInfo = false;

    # Kill all hardening
    hardeningDisable = [ "all" ];

    # Don't audit /tmp references
    noAuditTmpdir = true;
  };
}
