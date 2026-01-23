--| generate-nix.dhall: Generate Nix expressions from typed Dhall packages
--|
--| Run with: dhall text < generate-nix.dhall > nvidia-packages.nix

let Prelude = https://prelude.dhall-lang.org/v23.1.0/package.dhall
    sha256:931cbfae9d746c4611b07633ab1e547637ab4ba138b16bf65ef1b9ad66a60b7f

let Text/concatSep = Prelude.Text.concatSep
let List/map = Prelude.List.map

let Script = ../prelude/Script.dhall
let Wheel = ./wheel-template.dhall

-- =============================================================================
-- Script to Bash Conversion (for installPhase)
-- =============================================================================

let pathToBash : Script.Path -> Text =
      \(p : Script.Path) ->
        merge
          { Src = \(t : Text) -> "\$src/${t}"
          , Out = \(t : Text) -> "\$out/${t}"
          , Dep = \(d : { dep : Text, path : Text }) -> "\${${d.dep}}/${d.path}"
          , Tmp = \(t : Text) -> "\$TMPDIR/${t}"
          , Abs = \(t : Text) -> t
          }
          p

let commandToBash : Script.Command -> Text =
      \(cmd : Script.Command) ->
        merge
          { Mkdir = \(m : { path : Script.Path, parents : Bool }) ->
              "mkdir ${if m.parents then "-p " else ""}${pathToBash m.path}"
          
          , Copy = \(c : { src : Script.Path, dst : Script.Path, recursive : Bool }) ->
              "cp ${if c.recursive then "-r " else ""}${pathToBash c.src} ${pathToBash c.dst}"
          
          , Move = \(m : { src : Script.Path, dst : Script.Path }) ->
              "mv ${pathToBash m.src} ${pathToBash m.dst}"
          
          , Remove = \(r : { path : Script.Path, recursive : Bool, force : Bool }) ->
              "rm ${if r.recursive then "-r " else ""}${if r.force then "-f " else ""}${pathToBash r.path}"
          
          , Symlink = \(s : { target : Script.Path, link : Script.Path }) ->
              "ln -s ${pathToBash s.target} ${pathToBash s.link}"
          
          , Chmod = \(_ : { path : Script.Path, mode : Script.Mode }) ->
              "# chmod (TODO)"
          
          , Touch = \(t : { path : Script.Path }) ->
              "touch ${pathToBash t.path}"
          
          , Write = \(_ : { path : Script.Path, content : Script.Interp }) ->
              "# write (TODO)"
          
          , Append = \(_ : { path : Script.Path, content : Script.Interp }) ->
              "# append (TODO)"
          
          , Substitute = \(_ : { file : Script.Path, replacements : List { from : Text, to : Script.Interp } }) ->
              "# substitute (TODO)"
          
          , Untar = \(u : { archive : Script.Path, dest : Script.Path, strip : Natural }) ->
              "tar xf ${pathToBash u.archive} -C ${pathToBash u.dest}"
          
          , Unzip = \(u : { archive : Script.Path, dest : Script.Path }) ->
              "unzip -q ${pathToBash u.archive} -d ${pathToBash u.dest}"
          
          , Tar = \(_ : { files : List Script.Path, archive : Script.Path, compression : < None | Gzip | Xz | Zstd > }) ->
              "# tar (TODO)"
          
          , Patch = \(p : { patch : Script.Path, strip : Natural }) ->
              "patch -p${Natural/show p.strip} -i ${pathToBash p.patch}"
          
          , PatchElf = \(_ : { binary : Script.Path, action : < SetRpath : List Script.Path | AddRpath : List Script.Path | SetInterpreter : Script.Path > }) ->
              "# patchelf (TODO)"
          
          , Configure = \(_ : { flags : List Script.Interp, workdir : Optional Script.Path }) ->
              "./configure"
          
          , Make = \(m : { targets : List Text, flags : List Script.Interp, jobs : Optional Natural }) ->
              "make ${Text/concatSep " " m.targets}"
          
          , CMake = \(_ : { srcdir : Script.Path, builddir : Script.Path, flags : List Script.Interp }) ->
              "# cmake (TODO)"
          
          , Meson = \(_ : { srcdir : Script.Path, builddir : Script.Path, flags : List Script.Interp }) ->
              "# meson (TODO)"
          
          , Cargo = \(_ : { command : < Build | Test | Install >, flags : List Script.Interp }) ->
              "# cargo (TODO)"
          
          , Cabal = \(_ : { command : < Build | Test | Install >, flags : List Script.Interp }) ->
              "# cabal (TODO)"
          
          , InstallBin = \(_ : { src : Script.Path, name : Optional Text }) ->
              "# installBin (TODO)"
          
          , InstallLib = \(_ : { src : Script.Path, name : Optional Text }) ->
              "# installLib (TODO)"
          
          , InstallHeader = \(_ : { src : Script.Path, name : Optional Text }) ->
              "# installHeader (TODO)"
          
          , InstallMan = \(_ : { src : Script.Path, section : Natural }) ->
              "# installMan (TODO)"
          
          , InstallDoc = \(_ : { src : Script.Path }) ->
              "# installDoc (TODO)"
          
          , Run = \(r : { cmd : Text, args : List Script.Interp, env : List { name : Text, value : Script.Interp } }) ->
              r.cmd
          
          , Shell = \(s : Text) ->
              s
          }
          cmd

let scriptToBash : Script.Script -> Text =
      \(script : Script.Script) ->
        Text/concatSep "\n      " (List/map Script.Command Text commandToBash script)

-- =============================================================================
-- Generate Nix Expression for a Wheel Package
-- =============================================================================

let wheelToNix : Wheel.WheelPackage.Type -> Text =
      \(pkg : Wheel.WheelPackage.Type) ->
        let installScript = Wheel.mkWheelInstallScript pkg.paths
        in
        ''
  ${pkg.pname} = prev.stdenv.mkDerivation {
    pname = "${pkg.pname}";
    version = "${pkg.version}";

    src = fetchurl {
      url = "${pkg.src.url}";
      hash = "${pkg.src.sha256}";
    };

    nativeBuildInputs = with final; [
      autoPatchelfHook
      unzip
    ];

    buildInputs = with final; [
      stdenv.cc.cc.lib
      zlib
    ];

    autoPatchelfIgnoreMissingDeps = [
      "libcuda.so.1"
      "libnvidia-ml.so.1"
    ];

    dontConfigure = true;
    dontBuild = true;
    dontUnpack = true;

    installPhase = '''
      runHook preInstall
      ${scriptToBash installScript}
      runHook postInstall
    ''';

    meta = {
      description = "${pkg.description}";
      homepage = "${pkg.homepage}";
      license = lib.licenses.unfree;
      platforms = [ "x86_64-linux" ];
    };
  };
''

-- =============================================================================
-- Generate Full Overlay
-- =============================================================================

let overlay =
      ''
# Auto-generated from Dhall wheel templates
# DO NOT EDIT - regenerate with: dhall text < generate-nix.dhall

final: prev:
let
  inherit (prev) lib fetchurl;
in
{
${wheelToNix Wheel.nccl}
${wheelToNix Wheel.cudnn}
${wheelToNix Wheel.tensorrt}
${wheelToNix Wheel.cutensor}
${wheelToNix Wheel.cusparselt}
}
''

in overlay
