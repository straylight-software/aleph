# nix/prelude/translations.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                            // translations //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     The dismembered parts of the Dixie Flatline construct were
#     scattered across the table. McCoy Pauley, someone had written
#     on the side of the old Hosaka in white grease pencil.
#
#     'Just so you know,' the Finn said, 'these guys, they don't
#      just hack. They're into all kindsa voodoo shit. Mambos and
#      houngans and such. Not that I believe any of it, mind you,
#      but you gotta be careful when you're messing around with a
#      dead console cowboy's ROM construct.'
#
#                                                         — Neuromancer
#
# Lisp-case to camelCase translations. The membrane that lets you write
# readable attribute names while nixpkgs expects its historical conventions.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ lib }:
let
  # ──────────────────────────────────────────────────────────────────────────
  #                         // translation table //
  # ──────────────────────────────────────────────────────────────────────────

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
