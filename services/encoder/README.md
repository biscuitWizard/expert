# Encoder Service

## Overview

The encoder is the shared singleton embedding service for the Expert attention fabric. It consumes raw events and produces embedding vectors. It also handles on-demand encoding requests (for example, goal descriptions). The service is domain-agnostic: **spec invariant 1** requires that all domain specialization live in the SSM (state-space model / scoring layer), not in the encoder. The encoder may batch events to improve throughput.

## Responsibility

- Ingest text from raw events and from explicit encode requests.
- Run embedding inference (via a pluggable backend).
- Publish enriched events and encode results on the appropriate streams.
- Optionally cache embeddings by content hash to reduce duplicate work.

## Ownership

| Concern | Owned by encoder |
|--------|-------------------|
| Model loading | Yes |
| Batching strategy | Yes |
| Embedding cache | Yes |

## Message Flow

| Direction | Channel |
|-----------|---------|
| Consumes | `events.raw.*` |
| Consumes | `requests.encode` |
| Publishes | `events.embedded.{stream_id}` |
| Publishes | `results.encode` |

## Dependencies

- **Redis** — stream consumption and publishing.
- **Ollama** — embedding inference via `/api/embed` endpoint, running **Qwen3-Embedding-8B** (4096-dim, Matryoshka-capable). Model configured via `EMBEDDINGS_MODEL` env var. See [encoder selection ADR](../../docs/decisions/encoder-selection.md).

## Operational Model

The encoder is **stateless** with respect to business state: it behaves as a pure function (text in, vector out). Ephemeral state such as batch buffers and an optional LRU cache exists only for performance and does not encode domain knowledge.

For architecture, batching, caching, and backend abstraction, see [DESIGN.md](./DESIGN.md).
