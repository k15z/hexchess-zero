# syntax=docker/dockerfile:1.7
# Stage 1: Build Rust PyO3 bindings
FROM python:3.13-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential pkg-config libssl-dev \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"
ENV CARGO_HOME=/usr/local/cargo

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install maturin

WORKDIR /build
COPY Cargo.toml Cargo.lock README.md LICENSE ./
COPY engine/ engine/
COPY bindings/ bindings/

# Cache cargo registry/git + target dir across builds. With buildx remote
# cache export, incremental rebuilds only recompile changed crates instead of
# the whole dependency graph each time.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/build/target,sharing=locked \
    cd bindings/python && maturin build --release -o /build/wheels

# Stage 2: Runtime
FROM python:3.13-slim

# TORCH_VARIANT: "cpu" for CPU-only (smaller image), "cu126" for CUDA 12.6
ARG TORCH_VARIANT=cpu
ARG TORCH_VERSION=2.11.0

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

COPY pyproject.toml ./

# Install third-party dependencies before copying app code. Most image builds
# change training/ only, and keeping dependencies in their own layer avoids
# rebuilding and re-exporting the large Torch/CUDA layer on those merges.
RUN --mount=type=cache,target=/root/.cache/pip <<'SH'
set -e
python - <<'PY' > /tmp/runtime-requirements.txt
import tomllib
import os

with open("pyproject.toml", "rb") as f:
    dependencies = tomllib.load(f)["project"]["dependencies"]

torch_requirement = f"torch=={os.environ['TORCH_VERSION']}+{os.environ['TORCH_VARIANT']}"

for dependency in dependencies:
    name = dependency.split(";", 1)[0].strip().lower()
    if name.startswith("hexchess-zero"):
        continue
    if name.startswith("torch"):
        print(torch_requirement)
        continue
    print(dependency)
PY
pip install -r /tmp/runtime-requirements.txt \
    --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}" \
    --extra-index-url https://pypi.org/simple
SH

COPY --from=builder /build/wheels/*.whl /tmp/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps /tmp/*.whl && rm /tmp/*.whl

COPY training/ training/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps .

ENTRYPOINT ["python", "-m", "training"]
