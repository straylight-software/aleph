#!/usr/bin/env bash
# lre-start: Start NativeLink for local remote execution with Buck2
set -euo pipefail

WORKERS="@defaultWorkers@"
PORT="@defaultPort@"
CONFIG_DIR="${XDG_RUNTIME_DIR:-/tmp}/nativelink"
PID_FILE="$CONFIG_DIR/nativelink.pid"
LOG_FILE="$CONFIG_DIR/nativelink.log"
NATIVELINK="@nativelink@"

usage() {
  cat <<USAGE
Usage: lre-start [OPTIONS]

Start NativeLink local remote execution server.

Options:
  --workers=N    Number of worker processes (default: $WORKERS)
  --port=N       Port for CAS/scheduler (default: $PORT)
  --status       Show status and exit
  --stop         Stop running server
  --help         Show this help
USAGE
}

status() {
  if [[ -f $PID_FILE ]]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
      echo "NativeLink running (PID: $PID)"
      echo "  Port: $PORT"
      echo "  Log: $LOG_FILE"
      return 0
    else
      echo "NativeLink not running (stale PID file)"
      rm -f "$PID_FILE"
      return 1
    fi
  else
    echo "NativeLink not running"
    return 1
  fi
}

stop_server() {
  if [[ -f $PID_FILE ]]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
      echo "Stopping NativeLink (PID: $PID)..."
      kill "$PID"
      rm -f "$PID_FILE"
      echo "Stopped."
    else
      echo "NativeLink not running (removing stale PID file)"
      rm -f "$PID_FILE"
    fi
  else
    echo "NativeLink not running"
  fi
}

write_config() {
  cat >"$CONFIG_DIR/config.json" <<CONFIG
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
CONFIG
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
  --workers=*)
    WORKERS="${1#*=}"
    shift
    ;;
  --port=*)
    PORT="${1#*=}"
    shift
    ;;
  --status)
    status
    exit $?
    ;;
  --stop)
    stop_server
    exit 0
    ;;
  --help | -h)
    usage
    exit 0
    ;;
  *)
    echo "Unknown option: $1"
    usage
    exit 1
    ;;
  esac
done

# Create config directory and storage
mkdir -p "$CONFIG_DIR/cas/content" "$CONFIG_DIR/cas/tmp"
mkdir -p "$CONFIG_DIR/ac/content" "$CONFIG_DIR/ac/tmp"

# Write config
write_config

# Check if already running
if [[ -f $PID_FILE ]]; then
  PID=$(cat "$PID_FILE")
  if kill -0 "$PID" 2>/dev/null; then
    echo "NativeLink already running (PID: $PID)"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

# Start nativelink
echo "Starting NativeLink on port $PORT..."
"$NATIVELINK" "$CONFIG_DIR/config.json" >"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"

# Wait for startup
sleep 1
if status >/dev/null 2>&1; then
  echo "NativeLink started successfully"
  echo "  Port: $PORT"
  echo "  PID: $(cat "$PID_FILE")"
  echo ""
  echo "Usage:"
  echo "  buck2 build --prefer-remote //..."
  echo "  buck2 build --remote-only //..."
else
  echo "Failed to start NativeLink. Check $LOG_FILE"
  exit 1
fi
