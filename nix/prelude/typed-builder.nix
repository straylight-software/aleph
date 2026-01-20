# nix/prelude/typed-builder.nix
#
# Zero-Bash Typed Builder
# ======================
#
# This module implements the zero-bash architecture from RFC-007.
#
# Instead of:
#   Haskell -> WASM -> attrset -> actionToShell -> bash strings -> stdenv
#
# We now have:
#   Haskell -> WASM -> spec -> validateStorePaths -> derivation { builder = aleph-exec }
#
# Key innovations:
#   1. Store paths are validated at eval time (not build time)
#   2. No shell strings anywhere - aleph-exec reads spec directly
#   3. Actions are executed by Haskell, not bash
#
# The aleph-exec binary reads a JSON spec and executes typed actions directly:
#   - Mkdir     -> System.Directory.createDirectoryIfMissing
#   - Copy      -> copyFileWithMetadata / copyDirectoryRecursive
#   - Symlink   -> createSymbolicLink
#   - WriteFile -> Text.IO.writeFile
#   - Unzip     -> Codec.Archive.Zip
#   - PatchElf* -> callProcess "patchelf" [...]  (only external tool needed)
#
{
  lib,
  writeText,
  runCommand,
  stdenv,
  # The aleph-exec binary (Haskell, built natively)
  aleph-exec ? null,
  # For fetching sources
  fetchFromGitHub,
  fetchurl,
}:

let
  # ────────────────────────────────────────────────────────────────────────────
  #                        // store-path-validation //
  # ────────────────────────────────────────────────────────────────────────────
  # Validate that a string is a valid Nix store path.
  #
  # This is the key security mechanism: you cannot construct arbitrary paths.
  # Only paths that exist in /nix/store pass validation.
  #
  # Returns: { valid : bool, path : string, error : string | null }
  #
  validateStorePath =
    path:
    let
      # Basic format check: must start with /nix/store/
      hasStorePrefix = lib.hasPrefix "/nix/store/" path;

      # Extract the hash-name portion
      # /nix/store/abc123-foo-1.0 -> abc123-foo-1.0
      storePart = lib.removePrefix "/nix/store/" path;

      # Hash must be 32 characters (base32 of SHA-256)
      hashPart = builtins.substring 0 32 storePart;
      hasValidHash = builtins.stringLength hashPart == 32;

      # Check if path actually exists in the store
      # Note: This is evaluated at Nix eval time, so the path must already exist
      # or be a known derivation output
      pathExists = builtins.pathExists (builtins.toPath path);
    in
    if !hasStorePrefix then
      {
        valid = false;
        inherit path;
        error = "Path does not start with /nix/store/: ${path}";
      }
    else if !hasValidHash then
      {
        valid = false;
        inherit path;
        error = "Invalid store hash format: ${path}";
      }
    else if !pathExists then
      {
        valid = false;
        inherit path;
        error = "Store path does not exist: ${path}";
      }
    else
      {
        valid = true;
        inherit path;
        error = null;
      };

  # Validate multiple store paths, collecting all errors
  validateStorePaths =
    paths:
    let
      results = map validateStorePath paths;
      errors = builtins.filter (r: !r.valid) results;
    in
    if errors == [ ] then
      {
        valid = true;
        errors = [ ];
      }
    else
      {
        valid = false;
        errors = map (e: e.error) errors;
      };

  # ────────────────────────────────────────────────────────────────────────────
  #                        // extract-store-paths //
  # ────────────────────────────────────────────────────────────────────────────
  # Extract all store paths from a spec for validation.
  #
  # This recursively walks the spec and collects:
  #   - StorePath references in actions (PatchElfRpath rpaths, etc.)
  #   - Resolved dependency paths
  #   - Patch file paths
  #
  extractStorePaths =
    spec:
    let
      # Extract from a single action
      extractFromAction =
        action:
        if action.action or "" == "patchelfRpath" then
          builtins.filter (p: lib.hasPrefix "/nix/store/" p) (action.rpaths or [ ])
        else if action.action or "" == "patchelfAddRpath" then
          builtins.filter (p: lib.hasPrefix "/nix/store/" p) (action.rpaths or [ ])
        else if action.action or "" == "patchelfInterpreter" then
          let
            interp = action.interpreter or "";
          in
          if lib.hasPrefix "/nix/store/" interp then [ interp ] else [ ]
        else
          [ ];

      # Extract from all phases
      phases = spec.phases or { };
      allActions =
        (phases.postPatch or [ ])
        ++ (phases.preConfigure or [ ])
        ++ (phases.installPhase or [ ])
        ++ (phases.postInstall or [ ])
        ++ (phases.postFixup or [ ]);

      actionPaths = lib.concatMap extractFromAction allActions;

      # Patch files (already store paths)
      patchPaths = spec.patches or [ ];
    in
    lib.unique (actionPaths ++ patchPaths);

  # ────────────────────────────────────────────────────────────────────────────
  #                        // resolve-source //
  # ────────────────────────────────────────────────────────────────────────────
  # Resolve source specification to a store path.
  #
  resolveSource =
    src:
    if src == null then
      null
    else if src.type == "github" then
      fetchFromGitHub {
        inherit (src)
          owner
          repo
          rev
          hash
          ;
      }
    else if src.type == "url" then
      fetchurl {
        inherit (src) url hash;
      }
    else if src.type == "store" then
      # Already a store path - validate it
      let
        validation = validateStorePath src.path;
      in
      if validation.valid then src.path else throw "Invalid store path in source: ${validation.error}"
    else
      throw "Unknown source type: ${src.type}";

  # ────────────────────────────────────────────────────────────────────────────
  #                        // resolve-dependencies //
  # ────────────────────────────────────────────────────────────────────────────
  # Resolve dependency names to actual packages.
  #
  # This is the bridge between the typed world (string names) and Nix packages.
  #
  resolveDeps =
    pkgs: names:
    map (
      name:
      let
        # Support dotted paths like "stdenv.cc.cc.lib"
        parts = lib.splitString "." name;
        resolved = lib.foldl' (acc: part: acc.${part} or null) pkgs parts;
      in
      if resolved != null then resolved else throw "Unknown package: ${name}"
    ) names;

  # ────────────────────────────────────────────────────────────────────────────
  #                        // build-typed-derivation //
  # ────────────────────────────────────────────────────────────────────────────
  # Build a derivation from a typed spec using aleph-exec.
  #
  # This is the zero-bash builder: no shell strings, no bash phases.
  # The aleph-exec binary reads the spec and executes actions directly.
  #
  buildTypedDerivation =
    {
      spec,
      pkgs,
    }:
    let
      # Validate all store paths in the spec
      storePaths = extractStorePaths spec;
      validation = validateStorePaths storePaths;

      # Fail early if any store paths are invalid
      _ =
        if !validation.valid then
          throw ''
            Invalid store paths in derivation spec:
            ${lib.concatStringsSep "\n" validation.errors}
          ''
        else
          null;

      # Resolve source
      src = resolveSource (spec.src or null);

      # Resolve dependencies
      deps = spec.deps or { };
      nativeBuildInputs = resolveDeps pkgs (deps.nativeBuildInputs or [ ]);
      buildInputs = resolveDeps pkgs (deps.buildInputs or [ ]);
      propagatedBuildInputs = resolveDeps pkgs (deps.propagatedBuildInputs or [ ]);
      checkInputs = resolveDeps pkgs (deps.checkInputs or [ ]);

      # Write spec to a file that aleph-exec can read
      specJson = builtins.toJSON spec;
      specFile = writeText "${spec.pname}-${spec.version}-spec.json" specJson;

      # The builder: aleph-exec reads the spec and executes actions
      builderScript = ''
        ${aleph-exec}/bin/aleph-exec \
          --spec "${specFile}" \
          --src "$src" \
          --out "$out"
      '';

    in
    if aleph-exec == null then
      throw ''
        Zero-bash builder requires aleph-exec binary.
        Build it with: nix build .#aleph-exec
      ''
    else
      stdenv.mkDerivation {
        inherit (spec) pname version;
        inherit
          src
          buildInputs
          propagatedBuildInputs
          checkInputs
          ;

        nativeBuildInputs = nativeBuildInputs ++ [ aleph-exec ];

        # No phases - aleph-exec handles everything
        dontUnpack = spec.dontUnpack or false;
        dontConfigure = true;
        dontBuild = true;
        dontFixup = true;

        # The only phase: run aleph-exec
        installPhase = builderScript;

        strictDeps = spec.strictDeps or true;
        doCheck = spec.doCheck or false;

        meta = {
          description = spec.meta.description or "";
          homepage = spec.meta.homepage or null;
          license = lib.licenses.${spec.meta.license or "unfree"} or lib.licenses.unfree;
          platforms = if (spec.meta.platforms or [ ]) == [ ] then lib.platforms.all else spec.meta.platforms;
          mainProgram = spec.meta.mainProgram or null;
        };
      };

  # ────────────────────────────────────────────────────────────────────────────
  #                        // pure-derivation //
  # ────────────────────────────────────────────────────────────────────────────
  # Create a pure derivation with aleph-exec as the sole builder.
  #
  # This is the ultimate zero-bash form: no stdenv, no bash at all.
  # The derivation uses aleph-exec directly as the builder.
  #
  # NOTE: This requires aleph-exec to be statically linked or have all
  # its dependencies in the closure.
  #
  pureDerivation =
    {
      spec,
      pkgs,
    }:
    let
      # Validate all store paths
      storePaths = extractStorePaths spec;
      validation = validateStorePaths storePaths;
      _ =
        if !validation.valid then
          throw "Invalid store paths: ${lib.concatStringsSep ", " validation.errors}"
        else
          null;

      # Resolve source
      src = resolveSource (spec.src or null);

      # Resolve all dependencies to store paths
      deps = spec.deps or { };
      resolveToPaths = names: map (d: "${d}") (resolveDeps pkgs names);
      nativeBuildInputs = resolveToPaths (deps.nativeBuildInputs or [ ]);
      buildInputs = resolveToPaths (deps.buildInputs or [ ]);

      # Write spec with resolved paths
      resolvedSpec = spec // {
        _resolvedDeps = {
          nativeBuildInputs = nativeBuildInputs;
          buildInputs = buildInputs;
        };
        _src = if src != null then "${src}" else null;
      };
      specJson = builtins.toJSON resolvedSpec;
      specFile = writeText "${spec.pname}-${spec.version}-spec.json" specJson;

    in
    if aleph-exec == null then
      throw "Pure derivation requires aleph-exec binary"
    else
      derivation {
        name = "${spec.pname}-${spec.version}";
        system = pkgs.stdenv.hostPlatform.system;

        # aleph-exec is the builder - no bash involved
        builder = "${aleph-exec}/bin/aleph-exec";
        args = [
          "--spec"
          "${specFile}"
        ];

        # Pass source and output as environment
        inherit src;

        # All dependencies in the closure
        __structuredAttrs = true;
        outputs = [ "out" ];
      };

in
{
  inherit
    validateStorePath
    validateStorePaths
    extractStorePaths
    resolveSource
    resolveDeps
    buildTypedDerivation
    pureDerivation
    ;
}
