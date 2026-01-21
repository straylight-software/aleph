#!/usr/bin/env bash
# lre-start.sh - Start NativeLink for Local Remote Execution
#
# THE GUARANTEE: First command in dev shell = that command in build
#
# This script starts:
#   1. NativeLink CAS (content-addressable storage) on port 50051
#   2. NativeLink Scheduler on port 50052
#   3. NativeLink Worker(s)
#
# Usage:
#   lre-start              # Start all components
#   lre-start --workers=4  # Start with 4 workers
#   lre-start --stop       # Stop all components

set -euo pipefail

# Configuration
LRE_DIR="${LRE_DIR:-$HOME/.cache/lre}"
WORKERS="${WORKERS:-4}"
CAS_PORT="${CAS_PORT:-50051}"
SCHEDULER_PORT="${SCHEDULER_PORT:-50052}"
WORKER_BASE_PORT="${WORKER_BASE_PORT:-50061}"

# Parse arguments
while [[ $# -gt 0 ]]; do
	case $1 in
	--workers=*)
		WORKERS="${1#*=}"
		shift
		;;
	--stop)
		echo "Stopping LRE services..."
		pkill -f "nativelink.*lre" 2>/dev/null || true
		echo "Stopped."
		exit 0
		;;
	--status)
		echo "LRE Status:"
		pgrep -af "nativelink.*lre" || echo "No LRE processes running"
		exit 0
		;;
	*)
		echo "Unknown option: $1"
		exit 1
		;;
	esac
done

# Create directories
mkdir -p "$LRE_DIR"/{cas,scheduler,workers}
mkdir -p "$LRE_DIR/config"

# Generate CAS config
cat >"$LRE_DIR/config/cas.json" <<EOF
{
  "stores": {
    "CAS_MAIN_STORE": {
      "filesystem": {
        "content_path": "$LRE_DIR/cas/content",
        "temp_path": "$LRE_DIR/cas/temp",
        "eviction_policy": {
          "max_bytes": 10737418240
        }
      }
    },
    "AC_MAIN_STORE": {
      "filesystem": {
        "content_path": "$LRE_DIR/cas/ac",
        "temp_path": "$LRE_DIR/cas/ac-temp",
        "eviction_policy": {
          "max_bytes": 1073741824
        }
      }
    }
  },
  "servers": [{
    "listener": {
      "http": {
        "socket_address": "0.0.0.0:$CAS_PORT"
      }
    },
    "services": {
      "cas": {
        "main": {
          "cas_store": "CAS_MAIN_STORE"
        }
      },
      "ac": {
        "main": {
          "ac_store": "AC_MAIN_STORE"
        }
      },
      "bytestream": {
        "cas_stores": {
          "main": "CAS_MAIN_STORE"
        }
      },
      "capabilities": {}
    }
  }]
}
EOF

# Generate Scheduler config
cat >"$LRE_DIR/config/scheduler.json" <<EOF
{
  "stores": {
    "GRPC_LOCAL_STORE": {
      "grpc": {
        "instance_name": "main",
        "endpoints": [{
          "address": "grpc://127.0.0.1:$CAS_PORT"
        }],
        "connections_per_endpoint": 4
      }
    }
  },
  "servers": [{
    "listener": {
      "http": {
        "socket_address": "0.0.0.0:$SCHEDULER_PORT"
      }
    },
    "services": {
      "execution": {
        "main": {
          "cas_store": "GRPC_LOCAL_STORE",
          "scheduler": "MAIN_SCHEDULER"
        }
      },
      "capabilities": {}
    }
  }],
  "schedulers": {
    "MAIN_SCHEDULER": {
      "simple": {
        "supported_platform_properties": {
          "cpu_count": {
            "minimum": 1
          },
          "OSFamily": {
            "exact": "linux"
          }
        }
      }
    }
  }
}
EOF

# Generate Worker config template
generate_worker_config() {
	local worker_id=$1
	local worker_port=$((WORKER_BASE_PORT + worker_id))

	cat >"$LRE_DIR/config/worker-$worker_id.json" <<EOF
{
  "stores": {
    "GRPC_LOCAL_STORE": {
      "grpc": {
        "instance_name": "main",
        "endpoints": [{
          "address": "grpc://127.0.0.1:$CAS_PORT"
        }],
        "connections_per_endpoint": 2
      }
    },
    "LOCAL_WORKER_FAST_SLOW": {
      "fast_slow": {
        "fast": {
          "memory": {
            "eviction_policy": {
              "max_bytes": 1073741824
            }
          }
        },
        "slow": "GRPC_LOCAL_STORE"
      }
    }
  },
  "workers": [{
    "local": {
      "worker_api_endpoint": {
        "uri": "grpc://127.0.0.1:$SCHEDULER_PORT"
      },
      "cas_fast_slow_store": "LOCAL_WORKER_FAST_SLOW",
      "upload_action_result": {
        "ac_store": "GRPC_LOCAL_STORE"
      },
      "work_directory": "$LRE_DIR/workers/$worker_id",
      "platform_properties": {
        "cpu_count": "4",
        "OSFamily": "linux",
        "container-image": "nix-lre"
      }
    }
  }]
}
EOF
}

echo "Starting LRE (Local Remote Execution)..."
echo "  CAS: grpc://127.0.0.1:$CAS_PORT"
echo "  Scheduler: grpc://127.0.0.1:$SCHEDULER_PORT"
echo "  Workers: $WORKERS"
echo ""

# Start CAS
echo "Starting CAS..."
nativelink "$LRE_DIR/config/cas.json" &
CAS_PID=$!
echo "  CAS PID: $CAS_PID"

# Wait for CAS to be ready
sleep 1

# Start Scheduler
echo "Starting Scheduler..."
nativelink "$LRE_DIR/config/scheduler.json" &
SCHEDULER_PID=$!
echo "  Scheduler PID: $SCHEDULER_PID"

# Wait for Scheduler to be ready
sleep 1

# Start Workers
echo "Starting $WORKERS workers..."
for i in $(seq 0 $((WORKERS - 1))); do
	generate_worker_config $i
	mkdir -p "$LRE_DIR/workers/$i"
	nativelink "$LRE_DIR/config/worker-$i.json" &
	echo "  Worker $i PID: $!"
done

echo ""
echo "LRE is running. Use 'lre-start --stop' to stop."
echo ""
echo "Buck2 configuration:"
echo "  buck2 build --config=lre //:target"
echo ""
echo "Or set in .buckconfig.local:"
echo "  [buck2_re_client]"
echo "  engine_address = grpc://127.0.0.1:$SCHEDULER_PORT"
echo "  cas_address = grpc://127.0.0.1:$CAS_PORT"

# Wait for all background processes
wait
