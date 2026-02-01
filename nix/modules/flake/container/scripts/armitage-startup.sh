# Create directories for armitage
mkdir -p /var/cache/armitage /var/log/armitage /etc/ssl/armitage

# Set environment for armitage proxy
export PROXY_PORT="8888"
export PROXY_CACHE_DIR="/var/cache/armitage"
export PROXY_LOG_DIR="/var/log/armitage"
export PROXY_CERT_DIR="/etc/ssl/armitage"

# Start Armitage proxy
echo ":: Starting Armitage proxy on :8888..."
exec @armitageProxy@/bin/armitage-proxy
