# expert-redis

Redis client wrappers and stream conventions for the Expert system. Every inter-service message flows through Redis Streams. This crate defines the stream naming schema, consumer group conventions, message serialization format, and provides helper functions for common patterns.

## Stream Naming Schema

All streams follow a dotted namespace convention:

### Event Pipeline
- `events.raw.{stream_id}` -- Raw events from stream-ingestion. Consumed by encoder (consumer group: `encoder`).
- `events.embedded.{stream_id}` -- Embedded events from encoder. Consumed by ssm-workers (consumer group per worker: `worker-{worker_id}`). Range-read by context-builder via XREVRANGE.
- `events.exchange.{activity_id}` -- Raw LLM exchanges post-invocation. Consumed by rag-service (consumer group: `rag`).

### Signals
- `signals.fire` -- Fire signals from ssm-workers. Consumed by orchestrator (consumer group: `orchestrator`).

### Request/Response Pairs
- `requests.context` / -- Context assembly requests. Consumed by context-builder (consumer group: `ctx`).
- `requests.encode` / `results.encode` -- On-demand encoding. Consumed by encoder (consumer group: `encoder-od`) / orchestrator (filtered by request_id).
- `queries.rag` / `results.rag` -- RAG graph queries. Consumed by rag-service (consumer group: `rag`) / context-builder (filtered by request_id).
- `requests.summarize` / `results.summarize` -- Session history summarization. Consumed by llm-gateway (consumer group: `llm-summarize`) / rag-service (filtered by request_id).

### Write Streams
- `packages.ready` -- Assembled context packages. Consumed by llm-gateway (consumer group: `llm`).
- `labels.write` -- Training labels. Consumed by training-service (consumer group: `training`).
- `episodes.write` -- Episode data. Consumed by rag-service (consumer group: `rag-episodes`).
- `goals.write` -- Goal persistence. Consumed by rag-service (consumer group: `rag-goals`).
- `actions.{stream_id}` -- Outbound domain tool calls. Consumed by stream-ingestion (consumer group: `adapter-{stream_id}`).

### Command Channels
- `commands.worker.{worker_id}` -- Per-worker lifecycle commands from orchestrator.

### Notifications
- `checkpoints.available` -- Model checkpoint notifications from training-service. Consumed by orchestrator (consumer group: `orchestrator-ckpt`).

## Retention Policy

All streams use MAXLEN for bounded memory. Defaults:

| Stream pattern | Default MAXLEN | Rationale |
|---|---|---|
| `events.raw.*` | 10,000 | Raw events are transient; encoder consumes quickly |
| `events.embedded.*` | 10,000 | Primary retention point; context-builder reads via XREVRANGE |
| `signals.fire` | 1,000 | Fire signals are consumed immediately by orchestrator |
| `requests.*` / `results.*` | 1,000 | Request/response pairs are short-lived |
| `packages.ready` | 100 | Context packages are large; consumed immediately |
| `labels.write` | 10,000 | Training labels consumed by training-service |
| `events.exchange.*` | 1,000 | Raw exchanges consumed by rag-service |
| `actions.*` | 1,000 | Domain tool calls consumed immediately |
| `commands.worker.*` | 100 | Lifecycle commands consumed immediately |

## State Store Key Patterns

Redis is also used as a key-value store for fast state access:

- `state:{activity_id}` -- Serialized ActivityState blob (JSON). Written by ssm-worker on suspend, read on resume.
- `seq:{stream_id}` -- Monotonic sequence counter per stream (INCR). Used by stream-ingestion.
- `assignment:{activity_id}` -- Worker assignment (worker_id string). Written by orchestrator.
- `worker:{worker_id}:heartbeat` -- Last heartbeat timestamp. Written by ssm-worker, read by orchestrator.
- `fire_queue:{activity_id}` -- Pending fire signal for an activity (JSON). Written/replaced by orchestrator.

## Message Serialization

All messages are serialized as JSON. Each Redis Stream entry has a single field `data` containing the JSON-serialized message body. The message type is determined by which stream it's on (no envelope or type discriminator needed).

```
XADD events.raw.mud-01 MAXLEN ~10000 * data '{"id":"...","stream_id":"mud-01",...}'
```

## Helper API (planned)

```rust
pub struct StreamProducer { /* ... */ }
pub struct StreamConsumer { /* ... */ }

impl StreamProducer {
    pub async fn publish<T: Serialize>(&self, stream: &str, msg: &T) -> Result<String>;
}

impl StreamConsumer {
    pub async fn consume<T: DeserializeOwned>(&mut self) -> Result<(String, T)>;
    pub async fn ack(&self, id: &str) -> Result<()>;
}

pub async fn xrevrange<T: DeserializeOwned>(
    conn: &mut Connection,
    stream: &str,
    end: &str,
    start: &str,
    count: usize,
) -> Result<Vec<(String, T)>>;
```
