# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                                                                // translations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#    "Who owned it when Tally taped there?"
#    "Tessier-Ashpool S.A."
#    "I want to know more about Tessier-Ashpool."
#    "Antarctica starts here ."
#    She stared up through the steam at the white circle of the speaker. "What
#    id you just say?"
#    "Antarctica Starts Here  is a two-hour video study of the Tessier-Ashpool
#    amily by Hans Becker, Angie."
#    "Do you have it?"
#    "Of course. David Pope accessed it recently. He was quite impressed."
#    "Really? How recently?"
#    "Last Monday."
#    "I'll see it tonight, then."
#    "Done. Is that all?"
#    "Yes."
#    "Goodbye, Angie."
#
#                                                         — Mona Lisa Overdrive
# Philosophy:
#
#     `lisp-case-as-god-intended` to `camelCaseFromHell` translations. The
#     membrane that lets you write readable attribute names while nixpkgs
#     expects its historical conventions on the rare occasions it honors any
#     logic at all
#
#     N.B !! for use IN THE PRELUDE ONLY !!
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ lib }:
let

  # ══════════════════════════════════════════════════════════════════════════
  #                                                       // translation table
  # ══════════════════════════════════════════════════════════════════════════

  translations = {
    "build-inputs" = "buildInputs";
    "native-build-inputs" = "nativeBuildInputs";
    "propagated-build-inputs" = "propagatedBuildInputs";
    "check-inputs" = "checkInputs";
    "install-check-inputs" = "installCheckInputs";
    "pre-configure" = "preConfigure";
    "configure-flags" = "configureFlags";
    "post-configure" = "postConfigure";
    "configure-phase" = "configurePhase";
    "pre-build" = "preBuild";
    "build-flags" = "buildFlags";
    "post-build" = "postBuild";
    "build-phase" = "buildPhase";
    "pre-install" = "preInstall";
    "install-flags" = "installFlags";
    "post-install" = "postInstall";
    "install-phase" = "installPhase";
    "pre-check" = "preCheck";
    "check-flags" = "checkFlags";
    "post-check" = "postCheck";
    "check-phase" = "checkPhase";
    "pre-fixup" = "preFixup";
    "post-fixup" = "postFixup";
    "dont-unpack" = "dontUnpack";
    "dont-configure" = "dontConfigure";
    "dont-build" = "dontBuild";
    "dont-install" = "dontInstall";
    "do-check" = "doCheck";
    "dont-check" = "dontCheck";
    "dont-fixup" = "dontFixup";
    "make-flags" = "makeFlags";
    "cmake-flags" = "cmakeFlags";
    "meson-flags" = "mesonFlags";
    "source-root" = "sourceRoot";
    "unpack-phase" = "unpackPhase";
    "enable-parallel-building" = "enableParallelBuilding";
    "deps" = "buildInputs";
    "native-deps" = "nativeBuildInputs";

    # Rust-specific
    "cargo-hash" = "cargoHash";
    "cargo-lock" = "cargoLock";
    "cargo-sha256" = "cargoSha256";
    "use-next-est-lock-file" = "useNextestLockFile";

    # buildEnv
    "paths-to-link" = "pathsToLink";
    "extra-outputs-to-install" = "extraOutputsToInstall";
    "ignore-collision" = "ignoreCollisionExtraction";

    # writeShellApplication
    "runtime-inputs" = "runtimeInputs";
    "derivation-args" = "derivationArgs";

    # autoPatchelf
    "auto-patchelf-ignore-missing-deps" = "autoPatchelfIgnoreMissingDeps";

    # Fixed-output derivations (FOD)
    "output-hash-algo" = "outputHashAlgo";
    "output-hash-mode" = "outputHashMode";
    "output-hash" = "outputHash";
    "build-command" = "buildCommand";

    # Patching
    "post-patch" = "postPatch";
  };

  # ──────────────────────────────────────────────────────────────────────────
  #                          // translators //
  # ──────────────────────────────────────────────────────────────────────────

  # TODO[b7r6]: !! define this in terms of the prelude itself !!
  translate-attr = name: translations.${name} or name;

  translate-meta =
    meta:
    lib.mapAttrs' (n: v: {
      name =
        {
          "main-program" = "mainProgram";
          "broken" = "broken";
          "insecure" = "insecure";
        }
        .${n} or n;
      value = v;
    }) meta;

  translate-attrs =
    attrs:
    lib.mapAttrs' (n: v: {
      name = translate-attr n;
      value = if n == "meta" then translate-meta v else v;
    }) attrs;

in
{
  inherit
    translations
    translate-attr
    translate-meta
    translate-attrs
    ;
}
