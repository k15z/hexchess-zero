#!/usr/bin/env bash
set -euo pipefail

# Build the hexchess training image and import it into k3s on all nodes.
# Usage: ./k8s/push.sh [tag]

IMAGE="hexchess-training"
TAG="${1:-latest}"
TARBALL="/tmp/${IMAGE}.tar.gz"

NODES=(
  "root@46.224.129.243"   # master
  "root@46.225.158.88"    # worker1
  "root@46.224.160.54"    # worker2
)

echo "==> Building Docker image (linux/amd64)..."
docker buildx build --platform linux/amd64 -t "${IMAGE}:${TAG}" -f Dockerfile --load .

echo "==> Saving image to tarball..."
docker save "${IMAGE}:${TAG}" | gzip > "${TARBALL}"
echo "    $(du -h "${TARBALL}" | cut -f1)"

pids=()
for node in "${NODES[@]}"; do
  echo "==> Uploading and importing on ${node}..."
  (scp -q "${TARBALL}" "${node}:/tmp/" \
    && ssh "${node}" "gunzip -c /tmp/${IMAGE}.tar.gz | k3s ctr images import - && rm /tmp/${IMAGE}.tar.gz" \
    && echo "    ${node} done.") &
  pids+=($!)
done

for pid in "${pids[@]}"; do
  wait "${pid}" || { echo "ERROR: a node failed"; exit 1; }
done

rm -f "${TARBALL}"
echo "==> Done! Image ${IMAGE}:${TAG} available on all nodes."
