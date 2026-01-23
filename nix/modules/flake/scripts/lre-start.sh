#!/usr/bin/env bash
set -euo pipefail

# lre-start: Start NativeLink for local remote execution with Buck2
#
# Usage:
#   lre-start [--port=PORT] [--status] [--stop]

PORT="${LRE_DEFAULT_PORT:-50051}"
CONFIG_DIR="${XDG_RUNTIME_DIR:-/tmp}/nativelink"
NATIVELINK="@nativelink@"

show_help() {
	echo "lre-start: Start NativeLink for local remote execution"
	echo ""
	echo "Usage: lre-start [OPTIONS]"
	echo ""
	echo "Options:"
	echo "  --port=PORT    Port to listen on (default: 50051)"
	echo "  --status       Show status of running instance"
	echo "  --stop         Stop running instance"
	echo "  --help         Show this help"
}

show_status() {
	if [[ -f "$CONFIG_DIR/nativelink.pid" ]]; then
		PID=$(cat "$CONFIG_DIR/nativelink.pid")
		if kill -0 "$PID" 2>/dev/null; then
			echo "NativeLink running (PID: $PID)"
			echo "  Port: $PORT"
			echo "  Log: $CONFIG_DIR/nativelink.log"
			return 0
		fi
	fi
	echo "NativeLink not running"
	return 1
}

stop_instance() {
	if [[ -f "$CONFIG_DIR/nativelink.pid" ]]; then
		PID=$(cat "$CONFIG_DIR/nativelink.pid")
		if kill -0 "$PID" 2>/dev/null; then
			echo "Stopping NativeLink (PID: $PID)..."
			kill "$PID"
			rm -f "$CONFIG_DIR/nativelink.pid"
			echo "Stopped"
			return 0
		fi
	fi
	echo "NativeLink not running"
	return 1
}

# Parse arguments
for arg in "$@"; do
	case $arg in
	--port=*)
		PORT="${arg#*=}"
		;;
	--status)
		show_status
		exit $?
		;;
	--stop)
		stop_instance
		exit $?
		;;
	--help | -h)
		show_help
		exit 0
		;;
	*)
		echo "Unknown option: $arg"
		show_help
		exit 1
		;;
	esac
done

# Check if already running
if [[ -f "$CONFIG_DIR/nativelink.pid" ]]; then
	PID=$(cat "$CONFIG_DIR/nativelink.pid")
	if kill -0 "$PID" 2>/dev/null; then
		echo "NativeLink already running (PID: $PID)"
		echo "Use --stop to stop it first"
		exit 1
	fi
fi

# Create config directory
mkdir -p "$CONFIG_DIR"

# Write config file
cat >"$CONFIG_DIR/config.json" <<EOF
{
  "stores": [
    {
      "name": "CAS_MAIN_STORE",
      "compression": {
        "compression_algorithm": { "lz4": {} },
        "backend": {
          "filesystem": {
            "content_path": "$CONFIG_DIR/cas/content",
            "temp_path": "$CONFIG_DIR/cas/tmp",
            "eviction_policy": { "max_bytes": 10737418240 }
          }
        }
      }
    },
    {
      "name": "AC_MAIN_STORE",
      "filesystem": {
        "content_path": "$CONFIG_DIR/ac/content",
        "temp_path": "$CONFIG_DIR/ac/tmp",
        "eviction_policy": { "max_bytes": 536870912 }
      }
    }
  ],
  "schedulers": [
    {
      "name": "MAIN_SCHEDULER",
      "simple": {
        "supported_platform_properties": {
          "cpu_count": "minimum",
          "memory_kb": "minimum",
          "OSFamily": "priority",
          "container-image": "priority"
        }
      }
    }
  ],
  "servers": [
    {
      "listener": {
        "http": { "socket_address": "0.0.0.0:$PORT" }
      },
      "services": {
        "cas": [{ "cas_store": "CAS_MAIN_STORE" }],
        "ac": [{ "ac_store": "AC_MAIN_STORE" }],
        "execution": [{ "cas_store": "CAS_MAIN_STORE", "scheduler": "MAIN_SCHEDULER" }],
        "capabilities": [{ "remote_execution": { "scheduler": "MAIN_SCHEDULER" } }],
        "bytestream": { "cas_stores": { "": "CAS_MAIN_STORE" } },
        "health": {}
      }
    }
  ],
  "global": { "max_open_files": 65536 }
}
EOF

# Create storage directories
mkdir -p "$CONFIG_DIR/cas/content" "$CONFIG_DIR/cas/tmp"
mkdir -p "$CONFIG_DIR/ac/content" "$CONFIG_DIR/ac/tmp"

# Start NativeLink
echo "Starting NativeLink on port $PORT..."
$NATIVELINK "$CONFIG_DIR/config.json" >"$CONFIG_DIR/nativelink.log" 2>&1 &
PID=$!
echo "$PID" >"$CONFIG_DIR/nativelink.pid"

# Wait a moment and check if it started
sleep 1
if kill -0 "$PID" 2>/dev/null; then
	echo "NativeLink started successfully"
	echo "  Port: $PORT"
	echo "  PID: $PID"
	echo ""
	echo "Usage:"
	echo "  buck2 build --prefer-remote //..."
	echo "  buck2 build --remote-only //..."
else
	echo "Failed to start NativeLink"
	cat "$CONFIG_DIR/nativelink.log"
	exit 1
fi
