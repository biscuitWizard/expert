# RAG Service

API layer over the RAG graph database for the Expert attention fabric. The service exposes CRUD for Goal, Episode, Pattern, Domain, and Activity metadata nodes; semantic retrieval (top-K similar episodes); graph traversal; a background consolidation job that clusters episodes into patterns; session history management (raw exchange log, summarization triggers, compressed narrative storage); and goal versioning.

## Responsibility

- **Graph API**: CRUD for **Goal**, **Episode**, **Pattern**, **Domain**, and **Activity** metadata nodes in the backing graph.
- **Semantic retrieval**: Return top-K episodes similar to query embeddings, with summaries suitable for context assembly.
- **Graph traversal**: Answer structural queries over goals, domains, activities, episodes, and patterns as required by upstream consumers.
- **Consolidation**: Background job that clusters recent episodes semantically, maintains **Pattern** nodes, and links episodes via **INSTANCE_OF** edges.
- **Session history**: Append raw exchanges per activity, trigger summarization when logs exceed thresholds, store compressed narratives, and expose history to the context builder at assembly time.
- **Goal versioning**: Maintain versioned goals and associate training or generation-time data with the goal version active when that data was produced.

## Owns

- Graph schema (node and edge types, constraints, and indexing strategy)
- Vector index definitions and similarity policies used for episodes and patterns
- Consolidation clustering logic and pattern lifecycle
- Goal versioning state machine and historical embedding records
- Session history lifecycle (raw logs, thresholds, narrative storage keys)

## Consumes

| Source | Role |
|--------|------|
| `queries.rag` | Asynchronous RAG queries (semantic search, history, goal fetch, and related request types) from the context builder and other authorized producers |
| `goals.write` | Goal create/update events driving versioned goal state |
| `episodes.write` | Episode ingestion and metadata updates |
| `events.exchange.{activity_id}` | Per-activity raw LLM exchange events for session history |
| `results.summarize` | Compressed narrative output from the summarization pipeline |

## Publishes

| Destination | Role |
|-------------|------|
| `results.rag` | Query results correlated by **request_id** for consumers awaiting RAG responses |
| `requests.summarize` | Summarization jobs when per-activity raw exchange logs exceed configured size thresholds |

## Dependencies

- **Qdrant** — vector store for Episodes, Patterns, and Goals. Graph relationships modeled as payload fields with application-level joins. See [RAG DB selection ADR](../../docs/decisions/rag-db-selection.md). Embedding dimension: 4096 (Qwen3-Embedding-8B).
- **Redis** — stream transport for the async query/response protocol and the listed consume/publish topics.

## Stateful

Yes. All durable graph data, vector indices, session logs, compressed narratives, and goal version history live in the backing database and associated storage.

## Related documentation

See [DESIGN.md](./DESIGN.md) for graph schema, retrieval and versioning rules, session lifecycle, consolidation, and the Redis Streams protocol.
