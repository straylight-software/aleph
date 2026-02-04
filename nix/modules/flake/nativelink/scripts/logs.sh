set -euo pipefail
SERVICE="${1:-all}"
case "$SERVICE" in
scheduler)
  flyctl logs -a "@appPrefix@-scheduler" --no-tail | tail -50
  ;;
cas)
  flyctl logs -a "@appPrefix@-cas" --no-tail | tail -50
  ;;
worker)
  flyctl logs -a "@appPrefix@-worker" --no-tail | tail -50
  ;;
all | *)
  echo "=== Scheduler logs ===" && flyctl logs -a "@appPrefix@-scheduler" --no-tail 2>&1 | tail -20
  echo "" && echo "=== CAS logs ===" && flyctl logs -a "@appPrefix@-cas" --no-tail 2>&1 | tail -20
  echo "" && echo "=== Worker logs ===" && flyctl logs -a "@appPrefix@-worker" --no-tail 2>&1 | tail -20
  ;;
esac
