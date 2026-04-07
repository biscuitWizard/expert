# Encoder Service — Design

This document describes the internal design of the encoder: dual inputs, model abstraction, micro-batching, caching, output shapes, and domain boundaries.

## 1. Dual input streams

The encoder consumes from two sources:

- **`events.raw.*`** — High-volume event stream. Consumed with Redis consumer group **`encoder`**.
- **`requests.encode`** — Low-volume on-demand requests (e.g. goal description encoding).

Both paths produce embeddings:

- **Raw events** — After embedding, the service publishes to `events.embedded.{stream_id}`, preserving the logical stream identity of the input.
- **Encode requests** — Results are published to `results.encode` as structured responses keyed by request identifier.

The two streams share the same embedding pipeline and model backend; routing differs only at ingress (which stream to read) and egress (which topic and payload shape to publish).

## 2. Model abstraction

Embedding backends are abstracted behind a single trait so deployment can choose Ollama, ONNX, or an external API without changing stream logic.

```rust
#[async_trait]
pub trait EmbeddingModel: Send + Sync {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    fn dimension(&self) -> usize;
}
```

**Selected implementation: `OllamaEmbedder`**

The encoder uses Ollama running **Qwen3-Embedding-8B** (see [encoder selection ADR](../../docs/decisions/encoder-selection.md)). The trait boundary is preserved for future model swaps.

| Type | Backend | Status |
|------|---------|--------|
| `OllamaEmbedder` | Ollama `/api/embed` endpoint (Qwen3-Embedding-8B) | **Active** |
| `OnnxModel` | Local ONNX Runtime | Reserved for future lightweight models |
| `ApiModel` | External HTTP embedding API | Reserved for external providers |

**Output dimension:** 4096 (Qwen3-Embedding-8B default). Matryoshka truncation to smaller dimensions is supported by the model but requires re-embedding all stored content if changed.

## 3. Micro-batching

When events arrive faster than the model can sustain, the encoder **accumulates** work into batches subject to:

- **Maximum batch size** — `max_batch_size` (default **32**). A batch is flushed when it reaches this size.
- **Flush timeout** — A timer (default **10 ms**) flushes a **partial** batch if the batch is non-empty and no new items arrive before the timeout expires.

This combination is the primary **backpressure absorption** mechanism: bursty ingress is smoothed into model-sized chunks without blocking the event loop indefinitely on tiny batches.

If encode requests share the micro-batcher with raw events, configuration or a dedicated queue should cap latency so on-demand work is not starved by large event batches.

## 4. Embedding cache

An **optional LRU cache** keyed by a **hash of raw input text** can store previously computed embedding vectors.

- **Purpose** — Reduce redundant inference for repeated text (e.g. periodic heartbeats or identical payloads).
- **Capacity** — Configurable; eviction is LRU.
- **Encode requests** — On-demand `requests.encode` traffic **must bypass** the cache (or treat cache as explicitly disabled for that path) so goal and control-path encodings always reflect current policy and are not served stale vectors from an earlier identical string if semantics change outside the encoder.

Exact bypass behavior should be implemented as: no cache lookup and no cache insert for the encode-request path unless a future ADR specifies otherwise.

## 5. Output format

**Events**

The encoder **copies** the input `Event`, sets the `embedding` field to the computed vector, and publishes the **complete** `Event` to `events.embedded.{stream_id}`.

**Encode requests**

The encoder publishes an `EncodeResult` (or equivalent struct) containing at least:

- `request_id` — Correlates with the originating request.
- `embedding` — The resulting vector.

Consumers of `results.encode` use `request_id` for correlation with the caller's pending work.

## 6. Domain agnosticism

The encoder **must not** embed domain-specific logic: no scoring rules, no goal semantics, no SSM state, and no specialization by event type beyond what is required to extract embeddable text and dimensionality.

It **embeds text** only. All domain specialization and goal-conditioned behavior belong in the **SSM** (goal-conditioned scoring and related logic). Violating this boundary would break spec invariant 1 and couple the embedding tier to evolving domain models.
