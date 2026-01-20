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
rec {
  # ════════════════════════════════════════════════════════════════════════════
  # OCI IMAGE UTILITIES
  # ════════════════════════════════════════════════════════════════════════════

  oci = rec {
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
      ref:
      let
        parts = lib.splitString "/" ref;
        has-registry = lib.length parts > 2 || (lib.length parts == 2 && lib.hasInfix "." (lib.head parts));
        registry = if has-registry then lib.head parts else "docker.io";
        repo-with-tag = if has-registry then lib.concatStringsSep "/" (lib.tail parts) else ref;
        tag-parts = lib.splitString ":" repo-with-tag;
        repo = lib.head tag-parts;
        tag = if lib.length tag-parts > 1 then lib.elemAt tag-parts 1 else "latest";
      in
      {
        inherit registry repo tag;
      };

    # Convert ref to nix store-safe name
    #
    # Example:
    #   ref-to-name "nvcr.io/nvidia/pytorch:25.01-py3"
    #   => "nvidia-pytorch-25-01-py3"
    #
    ref-to-name =
      ref:
      let
        parsed = parse-ref ref;
      in
      lib.replaceStrings
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
        "${parsed.repo}-${parsed.tag}";

    # Common base images (convenience functions)
    images = {
      ubuntu = version: "ubuntu:${version}";
      debian = version: "debian:${version}";
      alpine = version: "alpine:${version}";
      python = version: "python:${version}";

      # NGC images
      ngc-pytorch = version: "nvcr.io/nvidia/pytorch:${version}-py3";
      ngc-triton = version: "nvcr.io/nvidia/tritonserver:${version}-py3";
      ngc-tensorrt = version: "nvcr.io/nvidia/tensorrt:${version}-py3";
      ngc-cuda-devel = version: "nvcr.io/nvidia/cuda:${version}-devel-ubuntu24.04";
      ngc-cuda-runtime = version: "nvcr.io/nvidia/cuda:${version}-runtime-ubuntu24.04";
    };
  };

  # ════════════════════════════════════════════════════════════════════════════
  # LINUX NAMESPACE UTILITIES
  # ════════════════════════════════════════════════════════════════════════════

  namespace = rec {
    # Common bwrap flags for all namespace environments
    base-flags = [
      "--dev-bind /dev /dev"
      "--proc /proc"
      "--tmpfs /tmp"
      "--die-with-parent"
    ];

    # FHS mount flags - present a library path as standard FHS locations
    #
    # Example:
    #   fhs-lib-flags "/nix/store/xxx-libs"
    #   => [ "--ro-bind /nix/store/xxx-libs /usr/lib" ... ]
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
      "--dev-bind /dev/nvidiactl /dev/nvidiactl"
      "--dev-bind /dev/nvidia-uvm /dev/nvidia-uvm"
      "--dev-bind /dev/nvidia-uvm-tools /dev/nvidia-uvm-tools"
      "--dev-bind /dev/nvidia-modeset /dev/nvidia-modeset"
    ];

    # DRI device bind flags (for OpenGL/Vulkan without full GPU)
    dri-flags = [
      "--dev-bind /dev/dri /dev/dri"
    ];

    # Network namespace flags
    network-flags = [
      "--share-net"
      "--ro-bind /etc/resolv.conf /etc/resolv.conf"
      "--ro-bind /etc/ssl /etc/ssl"
      "--ro-bind /etc/hosts /etc/hosts"
    ];

    # Home and CWD bind flags
    user-flags = [
      "--bind $HOME $HOME"
      "--bind $(pwd) $(pwd)"
      "--chdir $(pwd)"
    ];
  };

  # ════════════════════════════════════════════════════════════════════════════
  # FIRECRACKER UTILITIES
  # ════════════════════════════════════════════════════════════════════════════

  firecracker = rec {
    # Generate Firecracker config JSON
    #
    # Example:
    #   mk-config {
    #     kernel-path = "/path/to/vmlinux";
    #     rootfs-path = "/path/to/rootfs.ext4";
    #     cpus = 8;
    #     mem-mib = 8192;
    #   }
    #
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
      builtins.toJSON (
        {
          boot-source = {
            kernel_image_path = kernel-path;
            boot_args = boot-args;
          };
          drives = [
            {
              drive_id = "rootfs";
              path_on_host = rootfs-path;
              is_root_device = true;
              is_read_only = false;
            }
          ]
          ++ drives;
          machine-config = {
            vcpu_count = cpus;
            mem_size_mib = mem-mib;
          };
        }
        // lib.optionalAttrs (network-interfaces != [ ]) {
          inherit network-interfaces;
        }
      );

    # Network interface config
    mk-network-interface =
      {
        iface-id ? "eth0",
        guest-mac ? "AA:FC:00:00:00:01",
        host-dev-name ? "fc-tap0",
      }:
      {
        iface_id = iface-id;
        guest_mac = guest-mac;
        host_dev_name = host-dev-name;
      };

    # Minimal init script template for Firecracker VMs
    #
    # This init script:
    #   1. Mounts essential filesystems
    #   2. Optionally sets up networking
    #   3. Runs a build command (if provided)
    #   4. Either exits (build mode) or drops to shell (interactive mode)
    #
    init-script =
      {
        with-network ? true,
        build-cmd ? null,
        env ? { },
      }:
      let
        env-exports = lib.concatStringsSep "\n" (lib.mapAttrsToList (k: v: "export ${k}=\"${v}\"") env);
      in
      ''
        #!/bin/sh
        set -e

        export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
        ${env-exports}

        # Mount essential filesystems
        mount -t proc proc /proc
        mount -t sysfs sys /sys
        mount -t devtmpfs dev /dev 2>/dev/null || true
        mkdir -p /dev/pts /dev/shm
        mount -t devpts devpts /dev/pts 2>/dev/null || true
        mount -t tmpfs tmpfs /tmp 2>/dev/null || true
        mount -t tmpfs tmpfs /run 2>/dev/null || true

        hostname builder

        ${lib.optionalString with-network ''
          # Set up networking
          ip link set lo up 2>/dev/null || true
          if [ -e /sys/class/net/eth0 ]; then
            ip link set eth0 up
            ip addr add 172.16.0.2/24 dev eth0
            ip route add default via 172.16.0.1
            echo "nameserver 8.8.8.8" > /etc/resolv.conf
            echo "nameserver 1.1.1.1" >> /etc/resolv.conf
          fi
        ''}

        ${lib.optionalString (build-cmd != null) ''
          # Run build command
          echo ":: Running build command..."
          ${build-cmd}
          EXIT=$?
          echo ":: Exit code: $EXIT"

          # Trigger clean shutdown
          echo o > /proc/sysrq-trigger
        ''}

        # Drop to interactive shell if no build command
        ${lib.optionalString (build-cmd == null) ''
          exec setsid cttyhack /bin/bash -l
        ''}
      '';

    # Default boot args for different use cases
    boot-args = {
      minimal = "console=ttyS0 reboot=k panic=1 pci=off root=/dev/vda rw init=/init";
      with-serial = "console=ttyS0 earlyprintk=serial reboot=k panic=1 pci=off root=/dev/vda rw init=/init";
      quiet = "quiet loglevel=0 reboot=k panic=1 pci=off root=/dev/vda rw init=/init";
    };

    # Common VM sizes
    sizes = {
      small = {
        cpus = 2;
        mem-mib = 2048;
      };
      medium = {
        cpus = 4;
        mem-mib = 4096;
      };
      large = {
        cpus = 8;
        mem-mib = 8192;
      };
      xlarge = {
        cpus = 16;
        mem-mib = 16384;
      };
      builder = {
        cpus = 32;
        mem-mib = 65536;
      };
    };
  };

  # ════════════════════════════════════════════════════════════════════════════
  # ELF UTILITIES
  # ════════════════════════════════════════════════════════════════════════════

  elf = rec {
    # Build rpath from list of packages
    #
    # Example:
    #   mk-rpath [ pkgs.zlib pkgs.openssl ]
    #   => "/nix/store/xxx-zlib/lib:/nix/store/xxx-zlib/lib64:..."
    #
    mk-rpath =
      packages:
      lib.concatStringsSep ":" (
        lib.concatMap (
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

    # Generate patchelf commands for a directory
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
          ${lib.optionalString (interpreter-path != null) ''
            if file "$f" | grep -q "executable"; then
              patchelf --set-interpreter "${interpreter-path}" "$f" 2>/dev/null || true
            fi
          ''}
          patchelf --set-rpath "${rpath}" "$f" 2>/dev/null || true
        done
      '';

    # Patch ELF with existing rpath preserved
    patch-commands-preserve =
      {
        rpath,
        interpreter-path ? null,
        out,
      }:
      ''
        find ${out} -type f \( -executable -o -name "*.so*" \) 2>/dev/null | while read -r f; do
          [ -L "$f" ] && continue
          file "$f" | grep -q ELF || continue
          ${lib.optionalString (interpreter-path != null) ''
            if file "$f" | grep -q "executable"; then
              patchelf --set-interpreter "${interpreter-path}" "$f" 2>/dev/null || true
            fi
          ''}
          existing=$(patchelf --print-rpath "$f" 2>/dev/null || echo "")
          combined="${rpath}:${out}/lib:${out}/lib64''${existing:+:$existing}"
          patchelf --set-rpath "$combined" "$f" 2>/dev/null || true
        done
      '';
  };

  # ════════════════════════════════════════════════════════════════════════════
  # PEP 503 UTILITIES (Python Simple Repository API)
  # ════════════════════════════════════════════════════════════════════════════

  pep503 = rec {
    # Normalize package name per PEP 503
    # "Foo_Bar" -> "foo-bar"
    normalize-name = name: lib.toLower (lib.replaceStrings [ "_" "." ] [ "-" "-" ] name);

    # Parse wheel filename
    # "numpy-1.24.0-cp311-cp311-linux_x86_64.whl"
    # => { name = "numpy"; version = "1.24.0"; python = "cp311"; abi = "cp311"; platform = "linux_x86_64"; }
    parse-wheel-name =
      filename:
      let
        base = lib.removeSuffix ".whl" filename;
        parts = lib.splitString "-" base;
      in
      if lib.length parts >= 5 then
        {
          name = lib.head parts;
          version = lib.elemAt parts 1;
          python = lib.elemAt parts 2;
          abi = lib.elemAt parts 3;
          platform = lib.elemAt parts 4;
        }
      else
        null;

    # Generate index.html for a package
    mk-package-index =
      { name, wheels }:
      let
        links = lib.concatMapStringsSep "\n" (
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
        links = lib.concatMapStringsSep "\n" (p: ''<a href="${p}/">${p}</a><br/>'') packages;
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
