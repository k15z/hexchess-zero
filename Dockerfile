# Stage 1: Build Rust PyO3 bindings
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential pkg-config libssl-dev \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip install --no-cache-dir maturin

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY engine/ engine/
COPY bindings/ bindings/

# Build the Python extension
RUN cd bindings/python && maturin build --release -o /build/wheels

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Install Python deps
COPY training/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

# Install the hexchess wheel from stage 1
COPY --from=builder /build/wheels/*.whl /tmp/
RUN pip install /tmp/*.whl && rm /tmp/*.whl

# Copy training code
COPY training/ training/

ENTRYPOINT ["python", "-m", "training"]
