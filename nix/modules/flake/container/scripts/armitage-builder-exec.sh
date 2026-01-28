# Armitage builder exec-into script
#
# Sets up Armitage proxy environment and runs build command.
# Uses @ARMITAGE_PROXY_BIN@ placeholder for proxy binary path.

# Create directories for armitage
mkdir -p /var/cache/armitage /var/log/armitage /etc/ssl/armitage

# Start Armitage proxy in background
echo ":: Starting Armitage proxy..."
PROXY_PORT=8888 \
  PROXY_CACHE_DIR=/var/cache/armitage \
  PROXY_LOG_DIR=/var/log/armitage \
  PROXY_CERT_DIR=/etc/ssl/armitage \
  @ARMITAGE_PROXY_BIN@ &
ARMITAGE_PID=$!

# Wait for Armitage CA cert to be generated
echo ":: Waiting for Armitage CA cert..."
for i in $(seq 1 30); do
  [ -f /etc/ssl/armitage/ca.pem ] && break
  sleep 0.5
done

if [ ! -f /etc/ssl/armitage/ca.pem ]; then
  echo ":: ERROR: Armitage CA cert not found after 15s, dropping to shell"
  exec setsid cttyhack /bin/sh
fi

echo ":: Armitage proxy ready (PID $ARMITAGE_PID)"

# Export proxy env for all builds
export HTTP_PROXY="http://127.0.0.1:8888"
export HTTPS_PROXY="http://127.0.0.1:8888"
export http_proxy="http://127.0.0.1:8888"
export https_proxy="http://127.0.0.1:8888"
export SSL_CERT_FILE="/etc/ssl/armitage/ca.pem"

# Run build if /build-cmd exists, otherwise shell
if [ -f /build-cmd ]; then
  chmod +x /build-cmd
  /build-cmd
  EXIT_CODE=$?
  echo ":: Build exit: $EXIT_CODE"

  # Copy attestations to output if available
  if [ -d /var/log/armitage ] && [ -d /output ]; then
    cp -r /var/log/armitage /output/attestations
  fi

  # Shutdown VM
  echo o >/proc/sysrq-trigger
else
  echo ":: No /build-cmd, dropping to shell"
  exec setsid cttyhack /bin/sh
fi
