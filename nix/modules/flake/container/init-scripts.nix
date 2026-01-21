# nix/modules/flake/container/init-scripts.nix
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                              // vm init scripts //
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#     "Behind the deck, the other construct flickered in and out of focus."
#
#                                                         — Neuromancer
#
# VM init scripts for Firecracker and Cloud Hypervisor.
# These are Nix strings that get written via pkgs.writeText.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  # ────────────────────────────────────────────────────────────────────────────
  # // fc-run init (interactive shell) //
  # ────────────────────────────────────────────────────────────────────────────

  fc-run-init = ''
    #!/bin/sh
    export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    mount -t proc proc /proc
    mount -t sysfs sys /sys
    mount -t devtmpfs dev /dev 2>/dev/null || true
    mkdir -p /dev/pts /dev/shm
    mount -t devpts devpts /dev/pts 2>/dev/null || true
    mount -t tmpfs tmpfs /tmp 2>/dev/null || true
    mount -t tmpfs tmpfs /run 2>/dev/null || true
    hostname builder

    ip link set lo up 2>/dev/null || true
    if [ -e /sys/class/net/eth0 ]; then
      ip link set eth0 up 2>/dev/null || true
      ip addr add 172.16.0.2/24 dev eth0 2>/dev/null || true
      ip route add default via 172.16.0.1 2>/dev/null || true
      echo "nameserver 8.8.8.8" > /etc/resolv.conf
    fi

    if [ -f /build-cmd ]; then
      chmod +x /build-cmd
      /build-cmd
      EXIT=$?
      echo ":: Exit code: $EXIT"
      echo o > /proc/sysrq-trigger
    fi

    clear
    echo "════════════════════════════════════════════════════════"
    echo " Firecracker VM -"
    echo " CPUs: $(nproc)  RAM: $(free -h 2>/dev/null | awk '/Mem:/{print $2}' || echo '?')"
    echo "════════════════════════════════════════════════════════"
    echo ""
    # Launch interactive shell - prefer bash if available, fall back to sh
    if [ -x /bin/bash ]; then
      exec setsid cttyhack /bin/bash -l
    else
      exec setsid cttyhack /bin/sh
    fi
  '';

  # ────────────────────────────────────────────────────────────────────────────
  # // fc-build init (run command and exit) //
  # ────────────────────────────────────────────────────────────────────────────

  fc-build-init = ''
    #!/bin/sh
    export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    mount -t proc proc /proc
    mount -t sysfs sys /sys
    mount -t devtmpfs dev /dev 2>/dev/null || true
    mount -t tmpfs tmpfs /tmp 2>/dev/null || true
    hostname builder

    ip link set lo up 2>/dev/null || true
    if [ -e /sys/class/net/eth0 ]; then
      ip link set eth0 up
      ip addr add 172.16.0.2/24 dev eth0
      ip route add default via 172.16.0.1
      echo "nameserver 8.8.8.8" > /etc/resolv.conf
    fi

    if [ -f /build-cmd ]; then
      chmod +x /build-cmd
      /build-cmd
      EXIT=$?
      echo ":: Build exit code: $EXIT"
      echo o > /proc/sysrq-trigger
    fi
    # Fall back to sh if bash not available
    if [ -x /bin/bash ]; then
      exec /bin/bash
    else
      exec /bin/sh
    fi
  '';

  # ────────────────────────────────────────────────────────────────────────────
  # // ch-run init (interactive cloud hypervisor) //
  # ────────────────────────────────────────────────────────────────────────────

  ch-run-init = ''
    #!/bin/sh
    export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    mount -t proc proc /proc
    mount -t sysfs sys /sys
    mount -t devtmpfs dev /dev 2>/dev/null || true
    mkdir -p /dev/pts /dev/shm
    mount -t devpts devpts /dev/pts 2>/dev/null || true
    mount -t tmpfs tmpfs /tmp 2>/dev/null || true
    mount -t tmpfs tmpfs /run 2>/dev/null || true
    hostname cloud-vm

    ip link set lo up 2>/dev/null || true
    if [ -e /sys/class/net/eth0 ]; then
      ip link set eth0 up
      udhcpc -i eth0 2>/dev/null || (
        ip addr add 172.16.0.2/24 dev eth0
        ip route add default via 172.16.0.1
        echo "nameserver 8.8.8.8" > /etc/resolv.conf
      )
    fi

    clear
    echo "════════════════════════════════════════════════════════"
    echo " Cloud Hypervisor VM"
    echo " CPUs: $(nproc)  RAM: $(free -h 2>/dev/null | awk '/Mem:/{print $2}' || echo '?')"
    echo "════════════════════════════════════════════════════════"
    # Launch interactive shell - prefer bash if available, fall back to sh
    if [ -x /bin/bash ]; then
      exec setsid cttyhack /bin/bash -l
    else
      exec setsid cttyhack /bin/sh
    fi
  '';

  # ────────────────────────────────────────────────────────────────────────────
  # // ch-gpu init (cloud hypervisor with gpu passthrough) //
  # ────────────────────────────────────────────────────────────────────────────

  ch-gpu-init = ''
    #!/bin/sh
    export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    mount -t proc proc /proc
    mount -t sysfs sys /sys
    mount -t devtmpfs dev /dev 2>/dev/null || true
    mkdir -p /dev/pts /dev/shm
    mount -t devpts devpts /dev/pts 2>/dev/null || true
    mount -t tmpfs tmpfs /tmp 2>/dev/null || true
    mount -t tmpfs tmpfs /run 2>/dev/null || true
    hostname ch-gpu

    ip link set lo up 2>/dev/null || true
    if [ -e /sys/class/net/eth0 ]; then
      ip link set eth0 up 2>/dev/null || true
      ip addr add 172.16.0.2/24 dev eth0 2>/dev/null || true
      ip route add default via 172.16.0.1 2>/dev/null || true
      echo "nameserver 8.8.8.8" > /etc/resolv.conf
    fi

    echo ":: Waiting for GPU..."
    for i in $(seq 1 30); do
      if [ -e /dev/nvidia0 ]; then
        echo ":: GPU ready"
        break
      fi
      sleep 0.1
    done

    modprobe nvidia 2>/dev/null || true
    modprobe nvidia_uvm 2>/dev/null || true

    if [ -f /build-cmd ]; then
      chmod +x /build-cmd
      echo ":: Running command"
      /build-cmd
      EXIT=$?
      echo ":: Exit code: $EXIT"
      echo o > /proc/sysrq-trigger
    fi

    # Launch interactive shell - prefer bash if available, fall back to sh
    if [ -x /bin/bash ]; then
      exec setsid cttyhack /bin/bash -l
    else
      exec setsid cttyhack /bin/sh
    fi
  '';

  # ────────────────────────────────────────────────────────────────────────────
  # // lre-worker init (NativeLink worker for remote execution) //
  # ────────────────────────────────────────────────────────────────────────────

  lre-worker-init = ''
    #!/bin/sh
    export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/nix/store/bin
    mount -t proc proc /proc
    mount -t sysfs sys /sys
    mount -t devtmpfs dev /dev 2>/dev/null || true
    mkdir -p /dev/pts /dev/shm
    mount -t devpts devpts /dev/pts 2>/dev/null || true
    mount -t tmpfs tmpfs /tmp 2>/dev/null || true
    mount -t tmpfs tmpfs /run 2>/dev/null || true
    hostname lre-worker

    # Network setup
    ip link set lo up 2>/dev/null || true
    if [ -e /sys/class/net/eth0 ]; then
      ip link set eth0 up
      ip addr add 172.16.0.2/24 dev eth0
      ip route add default via 172.16.0.1
      echo "nameserver 8.8.8.8" > /etc/resolv.conf
      echo "nameserver 1.1.1.1" >> /etc/resolv.conf
    fi

    # Mount /nix/store from virtiofs if available
    if grep -q virtiofs /proc/filesystems 2>/dev/null; then
      mkdir -p /nix/store
      mount -t virtiofs nix-store /nix/store -o ro 2>/dev/null || true
    fi

    # Create work directory for NativeLink
    mkdir -p /tmp/nativelink

    echo "════════════════════════════════════════════════════════"
    echo " LRE Worker - NativeLink Remote Execution"
    echo " CPUs: $(nproc)  RAM: $(free -h 2>/dev/null | awk '/Mem:/{print $2}' || echo '?')"
    echo "════════════════════════════════════════════════════════"

    # Start NativeLink worker if config exists
    if [ -f /etc/nativelink/worker.json ]; then
      echo ":: Starting NativeLink worker..."
      exec /nix/store/bin/nativelink /etc/nativelink/worker.json
    else
      echo ":: No worker config found, dropping to shell"
      if [ -x /bin/bash ]; then
        exec setsid cttyhack /bin/bash -l
      else
        exec setsid cttyhack /bin/sh
      fi
    fi
  '';
}
