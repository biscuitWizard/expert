# ADR: Encoder Model Selection

## Status

**Decided: Qwen3-Embedding-8B served via dedicated llamacpp instance.**

## Context

The encoder service (spec invariant 1: domain-agnostic, shared singleton) converts raw event text and goal descriptions into embedding vectors. The architecture choice affects:
- **Embedding quality** (how well semantic similarity maps to goal relevance)
- **Inference latency** (encoder is on the hot path: every event passes through it)
- **Operational complexity** (model serving, resource requirements, dependencies)
- **Embedding dimension** (affects SSM input size, Qdrant storage, memory per activity for goal matrices)

## Model: Qwen3-Embedding-8B

- **Dimensions:** 4096 default. Supports Matryoshka Representation Learning (MRL) for configurable output dimensions from 32 to 8192 without retraining.
- **Parameters:** 8B. Requires GPU for practical inference throughput.
- **Quality:** State-of-the-art multilingual embedding model. Strong performance on retrieval benchmarks (MTEB).
- **Serving:** Available in GGUF format for llamacpp. Served via the `/embedding` endpoint.

## Serving Architecture

A **dedicated llamacpp container** runs Qwen3-Embedding-8B, separate from the LLM inference container. This avoids contention between embedding requests (high volume, low latency) and LLM completions (low volume, high latency, large context).

```
encoder service  ->  llamacpp-embeddings (port 8081)  ->  Qwen3-Embedding-8B.gguf
llm-gateway      ->  llamacpp (port 8080)             ->  LLM model.gguf
```

The encoder service implements the `LlamaCppModel` backend, calling the embedding container's `/embedding` endpoint.

## Dimension Strategy

Start with the **default 4096 dimensions**. This provides maximum quality from the model.

If memory or latency constraints emerge at scale (4096-dim vectors in Qdrant, goal matrices in ssm-worker, cosine similarity in feature computation), Matryoshka allows reducing to 1024 or 512 dimensions with graceful quality degradation -- no model retraining required, just truncate the output vectors.

**Important:** Changing the effective dimension invalidates all existing embeddings (goals, episodes, patterns, training data). A dimension change requires re-embedding all stored content. This is operationally expensive but mechanically simple (batch re-encode through the encoder, update Qdrant collections).

## Impact on SSM

With d=4096, the SSM parameter count (from the SSM architecture ADR) becomes:
- A: 4096 * 4096 = 16.7M (state transition -- consider diagonal or low-rank approximation)
- B: 4096 * input_dim
- C: k * 4096
- D: k

The full A matrix at 4096 dimensions may be too large for the linear SSM. Two mitigations:
1. **Diagonal A:** Restrict A to a diagonal matrix (4096 parameters instead of 16.7M). This is standard practice in structured state spaces -- each hidden dimension decays independently.
2. **Reduced SSM hidden dimension:** The SSM hidden state need not match the embedding dimension. Use a learned projection to reduce the 4096-dim input to a smaller hidden dimension (e.g., 256 or 512) before the recurrence.

The recommended approach is option 2: a projection layer `P: [hidden_dim, 4096]` reduces the input, then the SSM operates in the smaller hidden space. This keeps the SSM compact while using full-resolution embeddings for feature computation (cosine similarity against goals is computed in the original 4096 space, not the reduced space).

## Consequences

- A dedicated `llamacpp-embeddings` container is added to docker-compose.
- The encoder service uses the `LlamaCppModel` backend implementation, configured to point at the embeddings container.
- All vector dimensions throughout the system are 4096: Qdrant collections, goal embeddings, event embeddings, training store `goal_embedding` column.
- The SSM uses a learned input projection from 4096 to a smaller hidden dimension (e.g., 256).
- Micro-batching in the encoder service remains important: Qwen3-Embedding-8B benefits from batched inference on GPU.
- The `EmbeddingModel` trait in the encoder DESIGN.md is implemented as `LlamaCppModel`. The trait boundary is preserved for future model swaps.
