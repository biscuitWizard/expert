# stream-ingestion design

This document describes the internal architecture of the **stream-ingestion** service: adapter abstraction, normalization, Redis usage, outbound actions, and runtime lifecycle.

## Adapter trait

Each external source type implements a common Rust trait. The trait covers the full duplex lifecycle: connect, receive inbound raw events, send outbound actions, and disconnect. Planned adapter kinds include telnet, WebSocket, HTTP polling, and file tail; all are future implementations behind this interface.

```rust
use std::fmt::Debug;

/// Opaque inbound payload from the wire or source-specific API.
#[derive(Debug)]
pub struct RawEvent {
    // bytes, text, headers, etc. — shape is adapter-specific
}

/// Domain action to execute against the external system (e.g. tool call).
#[derive(Debug)]
pub struct Action {
    // tool name, serialized arguments, invocation context
}

/// Result of executing an action (success, error, response payload).
#[derive(Debug)]
pub struct ActionResult {
    // status, optional body/metadata for event normalization
}

pub trait Adapter: Send {
    async fn connect(&mut self) -> anyhow::Result<()>;
    async fn recv(&mut self) -> anyhow::Result<RawEvent>;
    async fn send(&mut self, action: Action) -> anyhow::Result<ActionResult>;
    async fn disconnect(&mut self) -> anyhow::Result<()>;
}
```

Native `async fn` in traits is available in recent Rust versions; alternatively, equivalent `impl Future` return types or the `async-trait` crate can be used depending on MSRV policy.

Inbound traffic is pulled via `recv`. Outbound tool calls from the LLM are applied via `send`, which maps to the source’s native protocol (TCP bytes, HTTP request, etc.).

## Event normalization

Each `RawEvent` is converted into the canonical `Event` struct from `expert-types` before publication. Required fields: `id` (unique event identifier), `stream_id`, `sequence` (assigned as described below), `timestamp` (typically Unix milliseconds), `raw` (human-readable serialized text from the adapter), and `metadata` (structured key-value data such as channel, sender, or event type).

The `embedding` field is always left `None` at this stage. The **encoder** service consumes raw events, computes embeddings, and publishes to the embedded-event stream downstream.

## Sequence numbering

Ordering within a single logical stream is guaranteed by a monotonic per-stream sequence counter. The counter is stored in Redis under the key `seq:{stream_id}` and incremented with `INCR` when assigning each new event’s `sequence` field. This yields a total order for events on that stream without requiring cross-service coordination beyond Redis atomicity.

## Redis Stream publishing

Normalized events are appended to `events.raw.{stream_id}` using `XADD`. The stream entry fields carry the serialized canonical event (JSON is the expected encoding for the primary payload field). Retention uses approximate trimming with `MAXLEN` on the order of 10 000 entries so bounded memory use is maintained while preserving recent history for consumers and recovery scenarios.

## Outbound action execution

The service consumes messages from `actions.{stream_id}` using a Redis Stream **consumer group**. Each message encodes a tool invocation from the LLM: tool name, arguments, and invocation context (correlation identifiers, trace metadata, etc.). The active adapter for that `stream_id` executes the action against the external source—for example sending bytes on a telnet session or issuing an HTTP POST.

Responses from the external system are not discarded: they are folded back into the inbound path by normalizing the outcome (or error) into a new `Event` and publishing it to `events.raw.{stream_id}` alongside organically received events. That keeps a single chronological stream of “what happened” for downstream agents.

## Lifecycle

Adapters are started and stopped under **orchestrator** commands (register stream, attach adapter configuration, shutdown). Each adapter instance runs inside its own **tokio** task so I/O and protocol handling do not block other streams.

On transport or protocol failure, the adapter attempts reconnection with **exponential backoff** until the orchestrator stops the stream or configuration changes. State that must survive process restarts remains in Redis (sequences, stream backlog); ephemeral connection state remains in the adapter task only.
