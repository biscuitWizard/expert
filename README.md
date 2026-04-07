# Expert

Semantic attention system for live event-driven AI agents. Determines **when** an LLM should be invoked on a continuous event stream, **what context** it should receive, and **how** invocation quality improves over time without human retraining.

## Architecture

8 Rust microservices + a Python CLI, connected via Redis Streams. No synchronous service-to-service calls in the data path.

| Service | Role |
|---|---|
| **stream-ingestion** | Bidirectional adapter: ingest events, execute outbound domain actions |
| **encoder** | Shared singleton: text → embedding vectors (Qwen3-Embedding-8B) |
| **orchestrator** | Control plane: activity lifecycle, goal CRUD, worker assignment, fire queue |
| **ssm-worker** | SSM recurrence, scoring, debounce, adaptive thresholds |
| **context-builder** | Assemble natural-language context packages for LLM invocation |
| **llm-gateway** | llamacpp integration, tool-call loop, feedback routing |
| **rag-service** | Graph DB API: episodes, patterns, goals, session history |
| **training-service** | Training store API: labeled examples, consensus, retraining |
| **expert-cli** | Python CLI calling orchestrator REST API |

Shared library crates: `expert-types`, `expert-redis`, `expert-vectors`, `expert-config`.

See [docs/architecture.md](docs/architecture.md) for full data-flow diagrams and design invariants.

## Prerequisites

- **Rust** — version pinned in [`rust-toolchain.toml`](rust-toolchain.toml) (currently 1.94.1)
- **Python >= 3.11** — for the CLI
- **Docker & Docker Compose** — for infrastructure containers

## Quick Start

```bash
# 1. Clone and set up dev environment
git clone <repo-url> && cd expert
make setup          # installs git pre-commit hook

# 2. Start infrastructure (Redis, PostgreSQL+pgvector, Qdrant, llamacpp)
docker compose up -d redis postgres qdrant llamacpp llamacpp-embeddings

# 3. Build and run services (pick one)
cargo build --workspace                          # build all
cargo run -p orchestrator                        # run a single service
docker compose up -d                             # or run everything via compose

# 4. Install the CLI
pip install -e ./cli
expert status
```

Place GGUF model files in `./models/` — the compose file expects `model.gguf` (chat) and `qwen3-embedding-8b.gguf` (embeddings).

## Development

### Initial setup

```bash
make setup
```

This configures git to use the project's [pre-commit hook](.githooks/pre-commit), which runs `cargo fmt --check` before each commit to catch formatting issues locally.

### Useful commands

```bash
make fmt             # auto-format all code
make check           # fmt check + clippy (mirrors CI)
cargo test --workspace --lib --bins   # unit tests
```

### Toolchain

The project uses Rust edition 2024. The exact compiler version is pinned in [`rust-toolchain.toml`](rust-toolchain.toml) so that local development, CI, and `cargo fmt` output are all identical. If you have `rustup` installed, the correct toolchain is installed automatically on first `cargo` invocation.

### Project layout

```
├── crates/
│   ├── expert-types/       # Shared data types (Event, Goal, ActivityState, etc.)
│   ├── expert-redis/       # Redis client wrappers, stream naming, consumer/producer
│   ├── expert-vectors/     # Embedding math: cosine similarity, EMA, centroid
│   └── expert-config/      # Configuration loading, tracing setup
├── services/
│   ├── stream-ingestion/
│   ├── encoder/
│   ├── orchestrator/
│   ├── ssm-worker/
│   ├── context-builder/
│   ├── llm-gateway/
│   ├── rag-service/
│   └── training-service/
├── cli/                    # Python CLI (typer + httpx + rich)
├── docs/                   # Architecture docs, ADRs
├── docker-compose.yml
├── Dockerfile              # Multi-stage build for all Rust services
└── Makefile
```

## Infrastructure

| Container | Image | Purpose |
|---|---|---|
| Redis | `redis:7` | Event bus (Streams), state store (KV), sequence counters |
| PostgreSQL | `pgvector/pgvector:pg16` | Training store |
| Qdrant | `qdrant/qdrant:latest` | RAG vector store |
| llamacpp | `ghcr.io/ggerganov/llama.cpp:server` | LLM inference |
| llamacpp-embeddings | `ghcr.io/ggerganov/llama.cpp:server` | Embedding model (Qwen3-Embedding-8B, 4096-dim) |

## CI

GitHub Actions runs on every push to `main` and on pull requests:

1. **Lint & Unit Tests** — `cargo fmt --check`, `cargo clippy`, `cargo test`
2. **Integration Tests** — full service tests against live Redis, PostgreSQL, and Qdrant
3. **CLI Smoke Test** — `pip install ./cli && expert --help`

## License

MIT
