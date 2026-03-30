#!/usr/bin/env bash
set -euo pipefail

# Manage the hexchess training cluster on k3s.
# Usage: ./k8s/run.sh <command> [args]

NAMESPACE="hexchess"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NFS_NODE="root@46.224.129.243"  # master node with NFS export

usage() {
  cat <<EOF
Usage: $(basename "$0") <command>

Commands:
  deploy              Apply all k8s manifests (namespace, NFS PV/PVC, deployments, cronjobs)
  restart             Rolling restart to pick up new image
  stop                Scale deployments to 0
  start               Scale deployments back up
  status              Show pods and training status
  logs [trainer|worker|elo]  Tail logs (default: trainer)
  elo                 Trigger an Elo ranking run now (outside cron schedule)
  clean               Stop deployments and wipe all training data on NFS
EOF
  exit 1
}

cmd_deploy() {
  echo "==> Applying manifests..."
  kubectl apply -f "$SCRIPT_DIR/nfs-pv.yaml"
  kubectl apply -f "$SCRIPT_DIR/trainer.yaml"
  kubectl apply -f "$SCRIPT_DIR/worker.yaml"
  kubectl apply -f "$SCRIPT_DIR/elo-service.yaml"
  echo "==> Done. Use '$(basename "$0") status' to check pods."
}

cmd_restart() {
  echo "==> Restarting deployments (will pull latest image)..."
  kubectl rollout restart deployment/hexchess-trainer -n "$NAMESPACE"
  kubectl rollout restart deployment/hexchess-worker -n "$NAMESPACE"
  kubectl rollout status deployment/hexchess-trainer -n "$NAMESPACE" --timeout=120s
  kubectl rollout status deployment/hexchess-worker -n "$NAMESPACE" --timeout=120s
  echo "==> Restarted."
}

cmd_stop() {
  echo "==> Scaling to 0..."
  kubectl scale deployment/hexchess-trainer deployment/hexchess-worker \
    deployment/hexchess-elo-service --replicas=0 -n "$NAMESPACE"
  echo "==> Stopped."
}

cmd_start() {
  echo "==> Scaling up..."
  kubectl scale deployment/hexchess-trainer --replicas=1 -n "$NAMESPACE"
  kubectl scale deployment/hexchess-worker --replicas=1 -n "$NAMESPACE"
  kubectl scale deployment/hexchess-elo-service --replicas=1 -n "$NAMESPACE"
  echo "==> Started. Use '$(basename "$0") status' to check pods."
}

cmd_status() {
  echo "==> Pods:"
  kubectl get pods -n "$NAMESPACE" -o wide
  echo ""
  echo "==> Deployments:"
  kubectl get deployments -n "$NAMESPACE"
  echo ""
  # Try to get training status from the trainer pod
  trainer_pod=$(kubectl get pods -n "$NAMESPACE" -l role=trainer \
    --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
  if [[ -n "$trainer_pod" ]]; then
    echo "==> Training status:"
    kubectl exec "$trainer_pod" -n "$NAMESPACE" -c trainer -- \
      python -m training status 2>/dev/null || echo "  (could not fetch status)"
  fi
}

cmd_logs() {
  local target="${1:-trainer}"
  case "$target" in
    trainer)
      kubectl logs -f deployment/hexchess-trainer -c trainer -n "$NAMESPACE"
      ;;
    worker)
      # Follow the standalone worker deployment
      kubectl logs -f deployment/hexchess-worker -n "$NAMESPACE"
      ;;
    elo)
      kubectl logs -f deployment/hexchess-elo-service -n "$NAMESPACE"
      ;;
    *)
      echo "Unknown log target: $target (use 'trainer', 'worker', or 'elo')"
      exit 1
      ;;
  esac
}

cmd_elo() {
  echo "==> Restarting Elo service..."
  kubectl rollout restart deployment/hexchess-elo-service -n "$NAMESPACE"
  echo "==> Restarted. Use '$(basename "$0") logs elo' to follow output."
}

cmd_clean() {
  read -rp "This will DELETE all training data on NFS. Are you sure? [y/N] " confirm
  if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted."
    exit 0
  fi
  echo "==> Stopping deployments first..."
  cmd_stop
  echo "==> Wiping NFS data..."
  ssh "$NFS_NODE" "rm -rf /srv/hexchess-data/{models,training_data,logs,gen*}"
  echo "==> Clean. Use 'deploy' or 'start' to resume."
}

[[ $# -lt 1 ]] && usage

case "$1" in
  deploy)  cmd_deploy ;;
  restart) cmd_restart ;;
  stop)    cmd_stop ;;
  start)   cmd_start ;;
  status)  cmd_status ;;
  logs)    cmd_logs "${2:-trainer}" ;;
  elo)     cmd_elo ;;
  clean)   cmd_clean ;;
  *)       usage ;;
esac
