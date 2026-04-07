# Expert -- Architecture Overview

Expert is a semantic attention system for live event-driven AI agents. It answers: when should an LLM be invoked on a continuous event stream, what context should it receive, and how does invocation quality improve over time?

The system decomposes into 8 Rust microservices, a Python CLI, 4 shared library crates, and 5 infrastructure containers, connected by asynchronous message-passing via Redis Streams.

## Services

| Service | Role | Hot path? |
|---|---|---|
| [stream-ingestion](../services/stream-ingestion/) | Bidirectional adapter: ingest events from external sources, execute outbound domain actions | Yes (inbound) |
| [encoder](../services/encoder/) | Shared singleton: text to embedding vectors | Yes |
| [orchestrator](../services/orchestrator/) | Control plane: activity lifecycle, goal CRUD, worker assignment, fire queue, backpressure | No |
| [ssm-worker](../services/ssm-worker/) | SSM recurrence, scoring, debounce, adaptive thresholds (pooled, many activities per worker) | Yes |
| [context-builder](../services/context-builder/) | Assemble natural language context packages for LLM invocation | No |
| [llm-gateway](../services/llm-gateway/) | llamacpp integration, tool call loop, feedback routing, alignment guardrails | No |
| [rag-service](../services/rag-service/) | Graph DB API: episodes, patterns, goals, session history, consolidation | No |
| [training-service](../services/training-service/) | Training store API: labeled examples, consensus, SLOW/MEDIUM retraining | No |
| [expert-cli](../cli/) | Python CLI calling orchestrator REST API | N/A |

## Shared Crates

| Crate | Purpose |
|---|---|
| [expert-types](../crates/expert-types/) | Canonical data types shared across services (Event, Goal, ActivityState, ContextPackage, etc.) |
| [expert-redis](../crates/expert-redis/) | Redis client wrappers, stream naming schema, consumer/producer helpers |
| [expert-vectors](../crates/expert-vectors/) | Embedding vector math: cosine similarity, EMA, centroid, Mahalanobis distance |
| [expert-config](../crates/expert-config/) | Configuration loading, logging setup |

## Infrastructure

| Container | Image | Role |
|---|---|---|
| Redis | `redis:7` | Event bus (Streams), activity state store (KV), sequence counters |
| PostgreSQL | `pgvector/pgvector:pg16` | Training store |
| Qdrant | `qdrant/qdrant:latest` | RAG vector store ([ADR](decisions/rag-db-selection.md)) |
| llamacpp | `ghcr.io/ggerganov/llama.cpp:server` | LLM inference |
| llamacpp-embeddings | `ghcr.io/ggerganov/llama.cpp:server` | Qwen3-Embedding-8B (4096-dim) ([ADR](decisions/encoder-selection.md)) |

## Data Flow

### Per-event hot path

```
External source
  -> stream-ingestion (normalize to Event)
  -> events.raw.{stream_id}
  -> encoder (embed)
  -> events.embedded.{stream_id}
  -> ssm-worker (feature computation, SSM recurrence, scoring)
  -> [threshold crossed + debounce confirmed]
  -> signals.fire
  -> orchestrator (fire queue, backpressure)
  -> requests.context
  -> context-builder (XREVRANGE lookback, RAG queries, render prompt)
  -> packages.ready
  -> llm-gateway (llamacpp invocation, tool loop)
  -> [feedback tools] -> labels.write -> training-service
  -> [domain tools]   -> actions.{stream_id} -> stream-ingestion (execute)
  -> [goal updates]   -> requests.goal_update -> orchestrator -> encoder -> ssm-worker
  -> [post-invocation] -> episodes.write -> rag-service
                       -> events.exchange.{activity_id} -> rag-service (history)
```

### Goal update propagation

```
LLM calls update_goal()
  -> llm-gateway publishes to requests.goal_update
  -> orchestrator publishes to requests.encode
  -> encoder produces embedding, publishes to results.encode
  -> orchestrator updates goal registry
  -> orchestrator publishes to goals.write (rag-service persists)
  -> orchestrator sends command to ssm-worker via commands.worker.{id}
  -> ssm-worker hot-updates goal matrix row
```

### Session history summarization (background)

```
llm-gateway publishes raw exchange to events.exchange.{activity_id}
  -> rag-service appends to raw log
  -> [if log exceeds threshold] -> requests.summarize
  -> llm-gateway calls llamacpp for summarization
  -> results.summarize
  -> rag-service stores compressed narrative
```

## Key Architectural Decisions

### Goals are atomic and per-activity
A goal exists only in the context of its owning activity. No shared-goal concurrency. The orchestrator can copy goals across activities as a control-plane operation.

### Activities are sequential
One LLM invocation at a time per activity. The lifecycle FSM enforces this.

### All inter-service communication is async
Every service-to-service interaction goes through Redis Streams. No synchronous HTTP call chains in the data path. The orchestrator REST API is the only synchronous endpoint (for external management).

### Pooled workers with stream affinity
SSM workers are pooled (N workers, many activities each). The orchestrator assigns activities to workers, preferring stream affinity so events are read once and fanned out to all co-located activities.

### Three-timescale adaptation
- FAST (per-event): hidden state accumulation + goal conditioning. No gradient. Free.
- MEDIUM (minutes): MAML-style meta-learning fine-tuning. Few gradient steps on few examples.
- SLOW (days/weeks): offline retraining on accumulated labels. Full weight updates.

### Backpressure by graceful degradation
Missing a fire is recoverable -- the SSM re-fires naturally. Fire queue depth 1 per activity, global cap at 2x LLM slots, stale signals dropped after TTL.

### Domain tools route through stream-ingestion
Bidirectional adapters: ingest events AND execute outbound actions. The LLM's domain tool calls are published to `actions.{stream_id}` and executed by the adapter that owns the connection.

## Design Invariants

These constraints are preserved across all implementation decisions (from spec Section 10):

1. The encoder is domain-agnostic and shared.
2. The SSM is reactive, not initiative.
3. The SSM never calls the LLM.
4. The LLM never sees raw stream data or embedding vectors.
5. Goals are independent of activities (deleting an activity preserves its goals in RAG).
6. All training labels are immutable once committed.
7. Suppress and recall are advisory, not binding.
8. Goal embeddings are versioned.
9. Activity suspension completes in < 100ms.
10. Training store and RAG graph are separate systems.
11. Context is natural language, not JSON.

## Open Decisions

- Debounce policy (start fixed, make configurable)

## Resolved Decisions

- [RAG database: Qdrant](decisions/rag-db-selection.md)
- [SSM model: minimal linear SSM (Option A)](decisions/ssm-architecture.md)
- [Encoder model: Qwen3-Embedding-8B via llamacpp](decisions/encoder-selection.md)
