# nix/build/from-dhall.nix
#
# buildFromDhall: Build packages from Dhall specs
#
# This is the Nix executor for Aleph-1. It:
# 1. Reads a Package.dhall spec (or receives it as attrset)
# 2. Fetches the source
# 3. Resolves dep names to store paths
# 4. Generates BuildContext.dhall
# 5. Invokes the appropriate builder
#
# No stdenv. No bash. Just Dhall -> Nix -> Haskell.
#
{
  lib,
  runCommand,
  writeText,
  fetchFromGitHub,
  fetchurl,
  dhall,
  dhall-json,
}:

let
  # ════════════════════════════════════════════════════════════════════════════
  # DHALL SCHEMA PATHS
  # ════════════════════════════════════════════════════════════════════════════

  drvDir = ../Drv;

  # ════════════════════════════════════════════════════════════════════════════
  # SOURCE FETCHERS
  # ════════════════════════════════════════════════════════════════════════════

  # Fetch source based on Src type from Dhall
  fetchSource =
    src:
    if src.type or null == "GitHub" then
      fetchFromGitHub {
        owner = src.owner;
        repo = src.repo;
        rev = src.rev;
        hash = src.hash;
      }
    else if src.type or null == "Url" then
      fetchurl {
        url = src.url;
        hash = src.hash;
      }
    else if src.type or null == "Local" then
      src.path
    else if src.type or null == "None" then
      null
    else
      throw "Unknown source type: ${builtins.toJSON src}";

  # ════════════════════════════════════════════════════════════════════════════
  # TRIPLE CONVERSION
  # ════════════════════════════════════════════════════════════════════════════

  # Convert Nix system string to Dhall Triple
  systemToTriple =
    system:
    let
      parts = lib.splitString "-" system;
      arch = builtins.head parts;
      os = builtins.elemAt parts 1;
    in
    {
      arch =
        {
          "x86_64" = "Arch.X86_64";
          "aarch64" = "Arch.AArch64";
          "armv7l" = "Arch.ARMv7";
          "riscv64" = "Arch.RISCV64";
          "wasm32" = "Arch.WASM32";
          "powerpc64le" = "Arch.PowerPC64LE";
        }
        .${arch} or (throw "Unknown arch: ${arch}");
      vendor = "Vendor.Unknown";
      os =
        {
          "linux" = "OS.Linux";
          "darwin" = "OS.Darwin";
          "windows" = "OS.Windows";
          "wasi" = "OS.WASI";
        }
        .${os} or (throw "Unknown OS: ${os}");
      abi = if os == "darwin" then "ABI.NoABI" else "ABI.GNU";
    };

  # Emit Dhall Triple literal
  tripleToDhall =
    triple:
    "{ arch = ${triple.arch}, vendor = ${triple.vendor}, os = ${triple.os}, abi = ${triple.abi} }";

  # ════════════════════════════════════════════════════════════════════════════
  # DEP RESOLUTION
  # ════════════════════════════════════════════════════════════════════════════

  # Resolve a dependency name to a package
  # This is the key integration point - maps Dhall dep names to Nix packages
  resolveDep =
    depRegistry: name:
    if builtins.hasAttr name depRegistry then
      depRegistry.${name}
    else
      throw "Unknown dependency: ${name}. Add it to depRegistry.";

  # Build deps map for BuildContext
  buildDepsMap =
    depRegistry: depNames:
    lib.listToAttrs (
      map (name: {
        inherit name;
        value = resolveDep depRegistry name;
      }) depNames
    );

  # ════════════════════════════════════════════════════════════════════════════
  # BUILD CONTEXT GENERATION
  # ════════════════════════════════════════════════════════════════════════════

  # Generate BuildContext.dhall content
  mkBuildContext =
    {
      out, # Output path (placeholder)
      src, # Source path
      host, # Host triple (Dhall)
      target, # Target triple (Optional, Dhall)
      cores, # Number of cores
      deps, # Map of name -> path
      specName, # Package name
      specVersion, # Package version
    }:
    ''
      let Triple = ${drvDir}/Triple.dhall

      in  { out = "${out}"
          , src = "${src}"
          , host = ${tripleToDhall host}
          , target = ${if target == null then "None Triple.Triple" else "Some ${tripleToDhall target}"}
          , cores = ${toString cores}
          , deps = [${
            lib.concatMapStringsSep ", " (n: ''{ mapKey = "${n}", mapValue = "${deps.${n}}" }'') (
              builtins.attrNames deps
            )
          }]
          , specName = "${specName}"
          , specVersion = "${specVersion}"
          }
    '';

  # ════════════════════════════════════════════════════════════════════════════
  # BUILD FROM DHALL
  # ════════════════════════════════════════════════════════════════════════════

  # The main function: build a package from a Dhall spec
  #
  # Arguments:
  #   spec: Either a path to a .dhall file or an attrset with package info
  #   depRegistry: Map of dep names to Nix packages (cmake, ninja, gcc, etc.)
  #   builders: Map of builder programs (aleph-build, or individual builders)
  #   system: The target system (e.g., "x86_64-linux")
  #
  buildFromDhall =
    {
      spec,
      depRegistry,
      builders,
      system ? builtins.currentSystem,
      cores ? 8,
    }:
    let
      # If spec is a path, we need to evaluate it with dhall-to-json
      # For now, assume spec is an attrset (the Dhall has been pre-evaluated)
      pkg =
        if builtins.isPath spec then
          throw "Direct .dhall file evaluation not yet implemented. Pass evaluated spec."
        else
          spec;

      # Fetch source
      src = fetchSource pkg.src;

      # Resolve dependencies
      resolvedDeps = buildDepsMap depRegistry pkg.deps;

      # Host triple from system
      hostTriple = systemToTriple system;

      # Target triple (for cross-compilation)
      targetTriple = pkg.target or null;

      # Select builder based on build type
      builderBin =
        let
          build = pkg.build;
        in
        if build.type or null == "CMake" then
          builders.cmake or (throw "No cmake builder provided")
        else if build.type or null == "Autotools" then
          builders.autotools or (throw "No autotools builder provided")
        else if build.type or null == "Meson" then
          builders.meson or (throw "No meson builder provided")
        else if build.type or null == "HeaderOnly" then
          builders.headerOnly or (throw "No header-only builder provided")
        else if build.type or null == "Custom" then
          builders.custom or (throw "No custom builder provided")
        else
          throw "Unknown build type: ${builtins.toJSON build}";

      # Generate BuildContext.dhall
      buildContextContent = mkBuildContext {
        out = builtins.placeholder "out";
        src = toString src;
        host = hostTriple;
        target = targetTriple;
        inherit cores;
        deps = lib.mapAttrs (_: toString) resolvedDeps;
        specName = pkg.name;
        specVersion = pkg.version;
      };

      buildContext = writeText "${pkg.name}-context.dhall" buildContextContent;

    in
    runCommand "${pkg.name}-${pkg.version}"
      {
        inherit src;
        nativeBuildInputs = [ builderBin ] ++ (builtins.attrValues resolvedDeps);

        # Pass the context file path to the builder
        ALEPH_CONTEXT = buildContext;

        passthru = {
          inherit pkg buildContext resolvedDeps;
        };
      }
      ''
        # The builder reads ALEPH_CONTEXT and does everything
        ${builderBin}/bin/${builderBin.meta.mainProgram or (builtins.baseNameOf (toString builderBin))}
      '';

in
{
  inherit
    buildFromDhall
    fetchSource
    systemToTriple
    tripleToDhall
    resolveDep
    buildDepsMap
    mkBuildContext
    ;
}
