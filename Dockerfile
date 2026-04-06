# Stage 1: Build Rust PyO3 bindings
FROM python:3.13-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential pkg-config libssl-dev \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip install --no-cache-dir maturin

WORKDIR /build
COPY Cargo.toml Cargo.lock README.md LICENSE ./
COPY engine/ engine/
COPY bindings/ bindings/

RUN cd bindings/python && maturin build --release -o /build/wheels

# Stage 2: Runtime
FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml ./
COPY training/ training/

RUN pip install --no-cache-dir . \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

COPY --from=builder /build/wheels/*.whl /tmp/
RUN pip install /tmp/*.whl && rm /tmp/*.whl

ENTRYPOINT ["python", "-m", "training"]
