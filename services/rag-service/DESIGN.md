# RAG Service: Design

This document specifies the RAG graph schema, retrieval and goal-versioning behavior, session history pipeline, consolidation job, and asynchronous query protocol for **rag-service**.

## 1. Graph schema

### 1.1 Node types

| Node | Description |
|------|-------------|
| **Episode** | One LLM invocation: inputs such as context and prompt state, model response, feedback, and outcome metadata as defined by the product contract. **Vector-indexed** for semantic retrieval. |
| **Pattern** | Distilled summary of a recurring class of episodes produced by consolidation. **Vector-indexed** for similarity search and alignment with new episodes. |
| **Domain** | Knowledge and metadata scoped to a stream type or domain slice used for filtering and graph proximity. |
| **Goal** | Goal objects, **versioned** per specification Section 9.1 (embedding and content revisions tracked across versions). |
| **Activity** | Activity metadata (identifiers, labels, pointers); does not hold full runtime activity state, which remains elsewhere in the fabric. |

### 1.2 Edge types

| Edge | Semantics |
|------|-----------|
| `Episode -[INSTANCE_OF]-> Pattern` | Episode classified under a consolidated pattern. |
| `Episode -[FOR_GOAL]-> Goal` | Episode generated or evaluated in the context of a specific **goal version** (see Section 3). |
| `Episode -[IN_DOMAIN]-> Domain` | Episode associated with a domain for filtering and proximity boosts. |
| `Pattern -[IN_DOMAIN]-> Domain` | Pattern scoped to a domain. |
| `Goal -[CHILD_OF]-> Goal` | Hierarchical goal structure (child references parent). |
| `Activity -[HAS_GOAL]-> Goal` | Activity linked to its current or relevant goal (exact resolution uses goal versioning rules). |

### 1.3 Qdrant collection mapping

Each node type maps to a Qdrant collection. Graph edges are modeled as payload fields resolved with filtered queries in application code. See [RAG DB selection ADR](../../docs/decisions/rag-db-selection.md) for the full schema.

| Collection | Vector dimension | Key payload fields |
|---|---|---|
| `episodes` | 4096 | activity_id, goal_id, goal_version, domain, created_at, pattern_id, was_suppressed |
| `patterns` | 4096 | domain, episode_count, created_at, updated_at |
| `goals` | 4096 | activity_id, parent_id, domain, version, active, created_at |

Additional indexes and constraints (uniqueness on goal version keys, activity identifiers, etc.) are defined at implementation time but must preserve the invariants in Sections 2–3.

## 2. Semantic retrieval

**Inputs**: An event embedding and a goal embedding (and optional filters such as domain or activity), plus **K** for result count.

**Process**:

1. **Vector search**: Score candidate **Episode** nodes by distance in embedding space relative to the combined or dual-query policy (implementation may concatenate, weight, or run two-stage retrieval; the contract must document the exact metric and weights).
2. **Graph proximity boost**: Increase effective similarity for episodes that share the same **Goal** (or goal lineage within policy) or **Domain** with the query context, so that structurally related experience ranks higher than isolated vector neighbors.
3. **Output**: Top-K **Episode** nodes, each accompanied by a **summary** field (or equivalent prose block) used by the context builder for **RELEVANT PAST CONTEXT** and related sections.

The service does not expose raw embedding vectors to the LLM prompt path; summaries and approved natural-language fields are the assembly-facing artifacts.

## 3. Goal versioning (specification invariant 8)

Goals are versioned monotonically.

- When a goal’s **embedding** (or other version-triggering attribute per Section 9.1) changes, the service **retains** the previous embedding with its **effective time range** (valid from / valid to or equivalent closed-open interval semantics as defined in the spec).
- A **new** goal version is created with an incremented **version number**. Edges and metadata point to the version record appropriate to each consumer.
- **Training data and generation-time tagging**: Episodes and other artifacts produced while a version was active are tagged with that **goal version identifier** so analytics, retrieval filters, and audit trails remain consistent after subsequent edits.

Readers that need the “current” goal follow the latest version pointer; historical queries resolve the version active at a given timestamp.

## 4. Session history lifecycle

Per **activity_id** (and **session_id** where applicable):

1. **Ingest**: Consume raw exchanges from `events.exchange.{activity_id}`.
2. **Append**: Append each exchange to a **per-activity raw exchange log** persisted in the database (ordering and deduplication follow the event schema).
3. **Threshold check**: After each append (or on a bounded schedule), evaluate whether the log **exceeds** the configured size threshold (byte count, message count, or both—implementation-defined but fixed per deployment).
4. **Summarization request**: If the threshold is exceeded, publish a summarization request to `requests.summarize` with enough correlation data for the summarizer to process the relevant log slice (activity_id, session_id, bounds, request id).
5. **Narrative ingest**: Consume the compressed narrative from `results.summarize` and **store** it keyed by **activity_id** + **session_id** (and any revision token if the summarizer emits one).
6. **Consumers**: The **context builder** reads **whatever history exists** at assembly time: compressed narrative when present, plus recent raw exchanges per policy, without assuming a fully summarized state.

Compaction or truncation of raw logs after successful summarization is an implementation detail but must not violate ordering or idempotency guarantees agreed with the summarization service.

## 5. Consolidation background job

A periodic **consolidation** process runs independently of the online query path:

1. **Scan**: Select **recent episodes** (time window or cursor-based, configurable).
2. **Cluster**: Group episodes that are semantically similar (same metric family as retrieval, with clustering parameters tuned for stability).
3. **Patterns**: For each cluster, **create or update** a **Pattern** node (summary text, embedding, domain links as inferred).
4. **Link**: Add or adjust `Episode -[INSTANCE_OF]-> Pattern` edges so episodes point at the best-matching pattern; merge or split patterns according to policy to avoid unbounded fragmentation.

This distills **operational knowledge** from raw experience so future retrieval can surface pattern-level context, not only isolated episodes.

## 6. Query/response protocol

Interaction is **asynchronous** via **Redis Streams** (or the project-standard stream implementation).

| Direction | Stream | Payload expectations |
|-----------|--------|----------------------|
| Inbound | `queries.rag` | **request_id** (correlation id), **query type** (e.g. `semantic_search`, `get_history`, `get_goal`), and type-specific parameters (embeddings, K, activity_id, session_id, goal id, filters). |
| Outbound | `results.rag` | Same **request_id**, success or error envelope, and type-specific result bodies (episode summaries, history blobs, goal version records). |

**Rules**:

- **rag-service** processes messages from `queries.rag` and publishes **exactly one** logical result per accepted request to `results.rag` with matching **request_id**.
- Clients (for example **context-builder**) must correlate replies by **request_id** and enforce **timeouts** so assembly does not block indefinitely.
- Concurrent requests are multiplexed by **request_id**; duplicate ids are rejected or de-duplicated per deployment policy.

CRUD and write paths on `goals.write`, `episodes.write`, and exchange streams follow their respective schemas; this section governs the **request/response** retrieval API used during context assembly.
