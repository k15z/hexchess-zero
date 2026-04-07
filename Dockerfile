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

# Cache cargo registry/git + target dir across builds. With buildx + the
# gha cache exporter (mode=max) these mounts persist between CI runs, so
# incremental rebuilds only recompile changed crates instead of the whole
# dep graph each time.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/build/target,sharing=locked \
    cd bindings/python && maturin build --release -o /build/wheels

# Stage 2: Runtime
FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml ./
COPY training/ training/

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install . \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

COPY --from=builder /build/wheels/*.whl /tmp/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install /tmp/*.whl && rm /tmp/*.whl

ENTRYPOINT ["python", "-m", "training"]
