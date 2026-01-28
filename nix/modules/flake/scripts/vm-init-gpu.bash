echo ":: Waiting for GPU..."
for _ in $(seq 1 30); do
  if [ -e /dev/nvidia0 ]; then
    echo ":: GPU ready"
    break
  fi
  sleep 0.1
done
modprobe nvidia 2>/dev/null || true
modprobe nvidia_uvm 2>/dev/null || true
