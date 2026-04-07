# Multi-stage build for all Expert Rust services.
# Build: docker build --build-arg SERVICE=<name> -t expert-<name> .

ARG SERVICE

FROM rust:1.94-bookworm AS builder

WORKDIR /app

# Copy workspace files
COPY Cargo.toml Cargo.lock* ./
COPY crates/ crates/
COPY services/ services/

# Build the target service in release mode
ARG SERVICE
RUN cargo build --release -p ${SERVICE}

# Runtime image
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG SERVICE
COPY --from=builder /app/target/release/${SERVICE} /usr/local/bin/service

ENTRYPOINT ["/usr/local/bin/service"]
