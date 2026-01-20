# nix/prelude/gpu.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                                  // gpu //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     'How you doing, Dixie?'
#     'I'm dead, Case. Got me this goddam spook's ass killed, and
#      you wanna know how I'm _doing_. I'm better off dead.'
#     'But you're not dead, man.'
#     'Shit. Lemme access that Flatline ghost. I can show you
#      some new things. The matrix is my kind of country.'
#
#                                                         — Neuromancer
#
# GPU architecture definitions. The silicon geometry that determines
# what's possible. Each generation unlocks new instructions, new
# precisions, new memory hierarchies.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_: {
  # ──────────────────────────────────────────────────────────────────────────
  #                           // architectures //
  # ──────────────────────────────────────────────────────────────────────────

  sm_120 = {
    arch = "sm_120";
    capability = "12.0";
    name = "blackwell";
    generation = 12;
  };

  sm_90a = {
    arch = "sm_90a";
    capability = "9.0";
    name = "hopper";
    generation = 9;
  };

  sm_89 = {
    arch = "sm_89";
    capability = "8.9";
    name = "ada";
    generation = 8;
  };

  sm_90 = {
    arch = "sm_90";
    capability = "9.0";
    name = "thor";
    generation = 9;
  };

  sm_86 = {
    arch = "sm_86";
    capability = "8.6";
    name = "ampere";
    generation = 8;
  };

  none = {
    arch = null;
    capability = null;
    name = "cpu";
    generation = 0;
  };

  # ──────────────────────────────────────────────────────────────────────────
  #                            // capabilities //
  # ──────────────────────────────────────────────────────────────────────────

  supports-fp8 = g: g.generation >= 9;
  supports-nvfp4 = g: g.generation >= 12;
  supports-tma = g: g.generation >= 9;
}
