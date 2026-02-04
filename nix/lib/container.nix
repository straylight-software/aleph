# lib/container.nix — Container, Namespace, and Firecracker Utilities
#
# Pure functions for working with OCI images, Linux namespaces, and Firecracker VMs.
# No pkgs dependency - these are library functions.
#
# Philosophy:
#   - Namespaces, not daemons (bwrap, not Docker)
#   - Presentation, not mutation (bind mounts, not patchelf)
#   - VM isolation for network builds (Firecracker, not sandbox escape)
#
{ lib }:
let
  # Lisp-case aliases for lib.* functions
  # :: t15
  # :: t16
  # :: t17
  # :: t18
  # :: t19
  # :: t20
  # :: t21
  # :: t22
  # :: t23
  # :: t24
  # :: t25
  # :: t26
  concat-map = lib.concatMap;
  concat-map-strings-sep = lib.concatMapStringsSep;
  # :: Path -> String
  # :: t27
  concat-strings-sep = lib.concatStringsSep;
  elem-at = lib.elemAt;
  has-infix = lib.hasInfix;
  map-attrs-to-list = lib.mapAttrsToList;
  optional-attrs = lib.optionalAttrs;
  optional-string = lib.optionalString;
  remove-suffix = lib.removeSuffix;
  replace-strings = lib.replaceStrings;
  # :: { images : { alpine : t62 -> String, debian : t61 -> String, ngc-cuda-devel : t67 -> String, ngc-cuda-runtime : t68 -> String, ngc-pytorch : t64 -> String, ngc-tensorrt : t66 -> String, ngc-triton : t65 -> String, python : t63 -> String, ubuntu : t60 -> String }, ref-to-name : t54 -> t59 }
  split-string = lib.splitString;
  to-lower = lib.toLower;

  # Lisp-case aliases for builtins.* functions
  read-file = builtins.readFile;
  to-json = builtins.toJSON;

  inherit (lib) head tail length;
in
{
  # :: t29 -> {}
  # ════════════════════════════════════════════════════════════════════════════
  # OCI IMAGE UTILITIES
  # :: t38
  # :: Bool
  # :: Any
  # :: t29
  # :: [Any]
  # :: Any
  # :: "latest"
  # ════════════════════════════════════════════════════════════════════════════

  oci =
    let
      # Parse image reference: registry/repo:tag -> { registry, repo, tag }
      #
      # Examples:
      #   parse-ref "ubuntu:24.04"
      #   => { registry = "docker.io"; repo = "ubuntu"; tag = "24.04"; }
      #
      #   parse-ref "nvcr.io/nvidia/pytorch:25.01-py3"
      #   => { registry = "nvcr.io"; repo = "nvidia/pytorch"; tag = "25.01-py3"; }
      #
      parse-ref =
        # :: t54 -> t59
        ref:
        let
          # :: {}
          parts = split-string "/" ref;
          has-registry = length parts > 2 || (length parts == 2 && has-infix "." (head parts));
          registry = if has-registry then head parts else "docker.io";
          repo-with-tag = if has-registry then concat-strings-sep "/" (tail parts) else ref;
          tag-parts = split-string ":" repo-with-tag;
          repo = head tag-parts;
          tag = if length tag-parts > 1 then elem-at tag-parts 1 else "latest";
        in
        {
          inherit registry repo tag;
        };
    in
    {
      inherit parse-ref;

      # :: { alpine : t62 -> String, debian : t61 -> String, ngc-cuda-devel : t67 -> String, ngc-cuda-runtime : t68 -> String, ngc-pytorch : t64 -> String, ngc-tensorrt : t66 -> String, ngc-triton : t65 -> String, python : t63 -> String, ubuntu : t60 -> String }
      # :: t60 -> String
      # :: t61 -> String
      # :: t62 -> String
      # :: t63 -> String
      # Convert ref to nix store-safe name
      #
      # :: t64 -> String
      # :: t65 -> String
      # :: t66 -> String
      # :: t67 -> String
      # :: t68 -> String
      # Example:
      #   ref-to-name "nvcr.io/nvidia/pytorch:25.01-py3"
      #   => "nvidia-pytorch-25-01-py3"
      #
      ref-to-name =
        ref:
        let
          # :: { base-flags : ["--dev-bind /dev /dev"], dri-flags : ["--dev-bind /dev/dri /dev/dri"], fhs-lib-flags : t69 -> [String], gpu-flags : ["--dev-bind /dev/nvidia0 /dev/nvidia0"], network-flags : ["--share-net"], user-flags : ["--bind $HOME $HOME"] }
          parsed = parse-ref ref;
        # :: ["--dev-bind /dev /dev"]
        in
        replace-strings
          [
            "/"
            ":"
            "."
          ]
          [
            "-"
            "-"
            "-"
          ]
          # :: t69 -> [String]
          "${parsed.repo}-${parsed.tag}";

      # Common base images (convenience functions)
      images = {
        ubuntu = version: "ubuntu:${version}";
        debian = version: "debian:${version}";
        alpine = version: "alpine:${version}";
        # :: ["--dev-bind /dev/nvidia0 /dev/nvidia0"]
        python = version: "python:${version}";

        # NGC images
        ngc-pytorch = version: "nvcr.io/nvidia/pytorch:${version}-py3";
        ngc-triton = version: "nvcr.io/nvidia/tritonserver:${version}-py3";
        ngc-tensorrt = version: "nvcr.io/nvidia/tensorrt:${version}-py3";
        ngc-cuda-devel = version: "nvcr.io/nvidia/cuda:${version}-devel-ubuntu24.04";
        ngc-cuda-runtime = version: "nvcr.io/nvidia/cuda:${version}-runtime-ubuntu24.04";
      # :: ["--dev-bind /dev/dri /dev/dri"]
      };
    };

  # ════════════════════════════════════════════════════════════════════════════
  # :: ["--share-net"]
  # LINUX NAMESPACE UTILITIES
  # ════════════════════════════════════════════════════════════════════════════

  namespace = {
    # Common bwrap flags for all namespace environments
    base-flags = [
      "--dev-bind /dev /dev"
      # :: ["--bind $HOME $HOME"]
      "--proc /proc"
      "--tmpfs /tmp"
      "--die-with-parent"
    ];

    # FHS mount flags - present a library path as standard FHS locations
    #
    # Example:
    #   fhs-lib-flags "/nix/store/xxx-libs"
    #   => [ "--ro-bind /nix/store/xxx-libs /usr/lib" ... ]
    # :: { boot-args : { minimal : "console=ttyS0 reboot=k panic=1 pci=off root=/dev/vda rw init=/init", quiet : "quiet loglevel=0 reboot=k panic=1 pci=off root=/dev/vda rw init=/init", with-serial : "console=ttyS0 earlyprintk=serial reboot=k panic=1 pci=off root=/dev/vda rw init=/init" }, init-script : { build-cmd : Null, env : {}, with-network : Bool } -> t104, mk-config : { boot-args : "console=ttyS0 reboot=k panic=1 pci=off root=/dev/vda rw init=/init", cpus : Int, drives : [{}], kernel-path : t70, mem-mib : Int, network-interfaces : [t74], rootfs-path : t71 } -> t77, mk-network-interface : { guest-mac : "AA:FC:00:00:00:01", host-dev-name : "isospin-tap0", iface-id : "eth0" } -> {}, sizes : { builder : { cpus : Int, mem-mib : Int }, large : { cpus : Int, mem-mib : Int }, medium : { cpus : Int, mem-mib : Int }, small : { cpus : Int, mem-mib : Int }, xlarge : { cpus : Int, mem-mib : Int } } }
    #
    fhs-lib-flags = lib-path: [
      "--ro-bind ${lib-path} /usr/lib"
      "--ro-bind ${lib-path} /usr/lib64"
      "--ro-bind ${lib-path} /lib"
      "--ro-bind ${lib-path} /lib64"
    ];

    # GPU device bind flags (NVIDIA)
    gpu-flags = [
      "--dev-bind /dev/nvidia0 /dev/nvidia0"
      # :: { boot-args : "console=ttyS0 reboot=k panic=1 pci=off root=/dev/vda rw init=/init", cpus : Int, drives : [{}], kernel-path : t70, mem-mib : Int, network-interfaces : [t74], rootfs-path : t71 } -> t77
      "--dev-bind /dev/nvidiactl /dev/nvidiactl"
      "--dev-bind /dev/nvidia-uvm /dev/nvidia-uvm"
      "--dev-bind /dev/nvidia-uvm-tools /dev/nvidia-uvm-tools"
      "--dev-bind /dev/nvidia-modeset /dev/nvidia-modeset"
    ];

    # DRI device bind flags (for OpenGL/Vulkan without full GPU)
    dri-flags = [
      "--dev-bind /dev/dri /dev/dri"
    ];

    # :: {}
    # Network namespace flags
    network-flags = [
      "--share-net"
      # :: [{}]
      "--ro-bind /etc/resolv.conf /etc/resolv.conf"
      "--ro-bind /etc/ssl /etc/ssl"
      "--ro-bind /etc/hosts /etc/hosts"
    ];

    # Home and CWD bind flags
    user-flags = [
      "--bind $HOME $HOME"
      # :: {}
      "--bind $(pwd) $(pwd)"
      "--chdir $(pwd)"
    ];
  };

  # ════════════════════════════════════════════════════════════════════════════
  # FIRECRACKER UTILITIES
  # ════════════════════════════════════════════════════════════════════════════

  firecracker = {
    # Generate Firecracker config JSON
    # :: { guest-mac : "AA:FC:00:00:00:01", host-dev-name : "isospin-tap0", iface-id : "eth0" } -> {}
    #
    # Example:
    #   mk-config {
    #     kernel-path = "/path/to/vmlinux";
    #     rootfs-path = "/path/to/rootfs.ext4";
    #     cpus = 8;
    #     mem-mib = 8192;
    #   }
    #
    # NOTE: Firecracker JSON schema attributes are quoted - external API
    mk-config =
      {
        kernel-path,
        rootfs-path,
        cpus ? 4,
        mem-mib ? 4096,
        boot-args ? "console=ttyS0 reboot=k panic=1 pci=off root=/dev/vda rw init=/init",
        network-interfaces ? [ ],
        drives ? [ ],
      }:
      # :: { build-cmd : Null, env : {}, with-network : Bool } -> t104
      to-json (
        {
          boot-source = {
            "kernel_image_path" = kernel-path;
            "boot_args" = boot-args;
          };
          # :: t88
          # :: t91
          # :: t97
          drives = [
            {
              "drive_id" = "rootfs";
              "path_on_host" = rootfs-path;
              "is_root_device" = true;
              # :: t99
              # :: String
              "is_read_only" = false;
            }
          ]
          ++ drives;
          machine-config = {
            "vcpu_count" = cpus;
            "mem_size_mib" = mem-mib;
          };
        }
        // optional-attrs (network-interfaces != [ ]) {
          inherit network-interfaces;
        }
      );
# :: { minimal : "console=ttyS0 reboot=k panic=1 pci=off root=/dev/vda rw init=/init", quiet : "quiet loglevel=0 reboot=k panic=1 pci=off root=/dev/vda rw init=/init", with-serial : "console=ttyS0 earlyprintk=serial reboot=k panic=1 pci=off root=/dev/vda rw init=/init" }
# :: "console=ttyS0 reboot=k panic=1 pci=off root=/dev/vda rw init=/init"
# :: "console=ttyS0 earlyprintk=serial reboot=k panic=1 pci=off root=/dev/vda rw init=/init"
# :: "quiet loglevel=0 reboot=k panic=1 pci=off root=/dev/vda rw init=/init"

    # Network interface config
    # NOTE: Firecracker JSON schema attributes are quoted - external API
    # :: { builder : { cpus : Int, mem-mib : Int }, large : { cpus : Int, mem-mib : Int }, medium : { cpus : Int, mem-mib : Int }, small : { cpus : Int, mem-mib : Int }, xlarge : { cpus : Int, mem-mib : Int } }
    # :: { cpus : Int, mem-mib : Int }
    # :: Int
    # :: Int
    mk-network-interface =
      # :: { cpus : Int, mem-mib : Int }
      # :: Int
      # :: Int
      {
        # :: { cpus : Int, mem-mib : Int }
        # :: Int
        # :: Int
        iface-id ? "eth0",
        # :: { cpus : Int, mem-mib : Int }
        # :: Int
        # :: Int
        guest-mac ? "AA:FC:00:00:00:01",
        # :: { cpus : Int, mem-mib : Int }
        # :: Int
        # :: Int
        host-dev-name ? "isospin-tap0",
      }:
      {
        "iface_id" = iface-id;
        "guest_mac" = guest-mac;
        "host_dev_name" = host-dev-name;
      };

    # :: { mk-rpath : t105 -> String, patch-commands : { interpreter-path : Null, rpath : t113 } -> t114 -> String, patch-commands-preserve : { interpreter-path : Null, out : String, rpath : String } -> t124 }
    # Minimal init script template for Firecracker VMs
    #
    # This init script:
    #   1. Mounts essential filesystems
    #   2. Optionally sets up networking
    #   3. Runs a build command (if provided)
    # :: t105 -> String
    #   4. Either exits (build mode) or drops to shell (interactive mode)
    #
    # Template loaded from ./scripts/isospin-init.sh.in to comply with ALEPH-W003.
    init-script =
      {
        # :: t109
        with-network ? true,
        build-cmd ? null,
        env ? { },
      }:
      let
        env-exports = concat-strings-sep "\n" (map-attrs-to-list (k: v: "export ${k}=\"${v}\"") env);
        network-setup = optional-string with-network (read-file ./scripts/isospin-init-network.sh);
        build-section = optional-string (build-cmd != null) (
          read-file ./scripts/isospin-init-build.sh
          + "\n"
          + build-cmd
          + "\nEXIT=$?\necho \":: Exit code: $EXIT\"\necho o > /proc/sysrq-trigger"
        );
        # :: { interpreter-path : Null, rpath : t113 } -> t114 -> String
        interactive-section = optional-string (build-cmd == null) "exec setsid cttyhack /bin/bash -l";
        template = read-file ./scripts/isospin-init.sh.in;
      in
      replace-strings
        [ "@env-exports@" "@base-init@" "@network-setup@" "@build-section@" "@interactive-section@" ]
        [
          env-exports
          (read-file ./scripts/isospin-init-base.sh)
          network-setup
          build-section
          interactive-section
        ]
        template;

    # Default boot args for different use cases
    boot-args = {
      minimal = "console=ttyS0 reboot=k panic=1 pci=off root=/dev/vda rw init=/init";
      with-serial = "console=ttyS0 earlyprintk=serial reboot=k panic=1 pci=off root=/dev/vda rw init=/init";
      quiet = "quiet loglevel=0 reboot=k panic=1 pci=off root=/dev/vda rw init=/init";
    # :: { interpreter-path : Null, out : String, rpath : String } -> t124
    };

    # Common VM sizes
    sizes = {
      small = {
        cpus = 2;
        # :: String
        mem-mib = 2048;
      };
      medium = {
        cpus = 4;
        # :: String
        mem-mib = 4096;
      };
      large = {
        cpus = 8;
        mem-mib = 8192;
      };
      xlarge = {
        cpus = 16;
        mem-mib = 16384;
      # :: { mk-package-index : { name : t147, wheels : t154 } -> String, mk-root-index : t154 -> String, normalize-name : String -> t129, parse-wheel-name : t130 -> { abi : "latest", name : Any, platform : "latest", python : "latest", version : "latest" } }
      };
      builder = {
        # :: String -> t129
        cpus = 32;
        mem-mib = 65536;
      };
    };
  # :: t130 -> { abi : "latest", name : Any, platform : "latest", python : "latest", version : "latest" }
  };

  # :: t134
  # :: [Any]
  # ════════════════════════════════════════════════════════════════════════════
  # ELF UTILITIES
  # ════════════════════════════════════════════════════════════════════════════
# :: Any
# :: "latest"
# :: "latest"
# :: "latest"
# :: "latest"

  elf = {
    # Build rpath from list of packages
    #
    # Example:
    # :: { name : t147, wheels : t148 } -> String
    #   mk-rpath [ pkgs.zlib pkgs.openssl ]
    #   => "/nix/store/xxx-zlib/lib:/nix/store/xxx-zlib/lib64:..."
    # :: t153
    #
    mk-rpath =
      packages:
      concat-strings-sep ":" (
        concat-map (
          pkg:
          let
            p = pkg.lib or pkg.out or pkg;
          in
          [
            "${p}/lib"
            "${p}/lib64"
          ]
        ) packages
      );
# :: t154 -> String

    # Generate patchelf commands for a directory
    # :: t159
    #
    # Example:
    #   patch-commands { rpath = mk-rpath deps; interpreter-path = "${glibc}/lib/ld-linux-x86-64.so.2"; } "$out"
    #
    patch-commands =
      {
        rpath,
        interpreter-path ? null,
      }:
      dir: ''
        find ${dir} -type f \( -executable -o -name "*.so*" \) 2>/dev/null | while read -r f; do
          [ -L "$f" ] && continue
          file "$f" | grep -q ELF || continue
          ${optional-string (interpreter-path != null) ''
            if file "$f" | grep -q "executable"; then
              patchelf --set-interpreter "${interpreter-path}" "$f" 2>/dev/null || true
            fi
          ''}
          patchelf --set-rpath "${rpath}" "$f" 2>/dev/null || true
        done
      '';

    # Patch ELF with existing rpath preserved
    # Template loaded from ./scripts/patch-elf-preserve.sh.in to comply with ALEPH-W003.
    patch-commands-preserve =
      {
        rpath,
        interpreter-path ? null,
        out,
      }:
      let
        interpreter-patch = optional-string (interpreter-path != null) ''
          if file "$f" | grep -q "executable"; then
            patchelf --set-interpreter "${interpreter-path}" "$f" 2>/dev/null || true
          fi
        '';
        template = read-file ./scripts/patch-elf-preserve.sh.in;
      in
      replace-strings [ "@out@" "@rpath@" "@interpreter-patch@" ] [ out rpath interpreter-patch ]
        template;
  };

  # ════════════════════════════════════════════════════════════════════════════
  # PEP 503 UTILITIES (Python Simple Repository API)
  # ════════════════════════════════════════════════════════════════════════════

  pep503 = {
    # Normalize package name per PEP 503
    # "Foo_Bar" -> "foo-bar"
    normalize-name = name: to-lower (replace-strings [ "_" "." ] [ "-" "-" ] name);

    # Parse wheel filename
    # "numpy-1.24.0-cp311-cp311-linux_x86_64.whl"
    # => { name = "numpy"; version = "1.24.0"; python = "cp311"; abi = "cp311"; platform = "linux_x86_64"; }
    parse-wheel-name =
      filename:
      let
        base = remove-suffix ".whl" filename;
        parts = split-string "-" base;
      in
      if length parts >= 5 then
        {
          name = head parts;
          version = elem-at parts 1;
          python = elem-at parts 2;
          abi = elem-at parts 3;
          platform = elem-at parts 4;
        }
      else
        null;

    # Generate index.html for a package
    mk-package-index =
      { name, wheels }:
      let
        links = concat-map-strings-sep "\n" (
          w: ''<a href="${w.filename}#sha256=${w.hash}">${w.filename}</a><br/>''
        ) wheels;
      in
      ''
        <!DOCTYPE html>
        <html>
        <head><title>${name}</title></head>
        <body>
        <h1>${name}</h1>
        ${links}
        </body>
        </html>
      '';

    # Generate root index.html
    mk-root-index =
      packages:
      let
        links = concat-map-strings-sep "\n" (p: ''<a href="${p}/">${p}</a><br/>'') packages;
      in
      ''
        <!DOCTYPE html>
        <html>
        <head><title>Simple Index</title></head>
        <body>
        <h1>Simple Index</h1>
        ${links}
        </body>
        </html>
      '';
  };
}
