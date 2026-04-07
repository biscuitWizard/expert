# ADR: RAG Graph Database Selection

## Status

**Decided: Qdrant.**

## Context

The rag-service needs a backing database that supports:
- **Vector similarity search** on Episode and Pattern embeddings (top-K retrieval for context assembly)
- **Graph relationships** between nodes (Episode -> Pattern, Episode -> Goal, Goal -> Goal hierarchy, etc.)
- **CRUD operations** on typed nodes with structured metadata
- **Background consolidation** that clusters episodes into patterns

The spec (Section 8.1-8.2) defines a knowledge graph with vector index. The question is which technology best serves both the graph traversal and vector retrieval needs.

## Options Evaluated

### Option A: Qdrant (SELECTED)

- **Vector-native, Rust-native.** Written in Rust, well-maintained Rust client (`qdrant-client`), purpose-built for vector similarity search.
- **Metadata filtering.** Supports rich payload filtering alongside vector search (filter by domain, goal_id, activity_id during retrieval).
- **Collections as node types.** Each node type (Episode, Pattern, Goal, etc.) maps to a Qdrant collection.
- **Graph traversal limitation.** No native graph traversal. Relationships (Episode -> Pattern, Goal -> Goal) modeled as payload fields and resolved with multiple queries. Multi-hop traversal requires application-level joins in rag-service.
- **Maturity.** Production-ready, active development, strong performance benchmarks.

### Option B: Neo4j

- Graph-native with Cypher queries, but JVM-based with significant resource requirements. The Rust client (`neo4rs`) is less mature. Vector index support exists but is secondary to the graph engine.

### Option C: SurrealDB

- Multi-model (documents, graph, vectors) and Rust-native, but younger project with less battle-tested vector search.

## Decision Rationale

The primary access pattern is **vector similarity search** -- context assembly queries dominate the read path. Graph traversal is secondary and limited to simple patterns:
- Goal hierarchy traversal: small trees, at most 3-4 levels deep
- Episode -> Pattern linking: single-hop lookup
- Domain filtering: metadata filter, not graph walk

These patterns are efficiently served by Qdrant's metadata filtering + multiple queries in application code. The Rust-native client eliminates protocol translation overhead. Qdrant's operational footprint is lightweight compared to Neo4j's JVM.

## Qdrant Collection Schema

| Collection | Vector dimension | Indexed payload fields |
|---|---|---|
| `episodes` | 4096 (Qwen3-Embedding-8B) | activity_id, goal_id, goal_version, domain, created_at, pattern_id, was_suppressed |
| `patterns` | 4096 | domain, episode_count, created_at, updated_at |
| `goals` | 4096 | activity_id, parent_id, domain, version, active, created_at |

Graph edges are modeled as payload fields:
- `Episode.pattern_id` -> INSTANCE_OF -> Pattern
- `Episode.goal_id` + `Episode.goal_version` -> FOR_GOAL -> Goal
- `Episode.domain` -> IN_DOMAIN -> Domain
- `Goal.parent_id` -> CHILD_OF -> Goal
- `Goal.activity_id` -> part of Activity's goal tree

Domain nodes are implicit (domain is a string field used for filtering, not a separate collection unless domain-level metadata grows complex enough to warrant one).

## Consequences

- rag-service uses the `qdrant-client` Rust crate for all graph DB operations.
- Graph relationships are payload fields, resolved with filtered queries in application code.
- Multi-hop traversals (rare) are multiple Qdrant queries composed in rag-service.
- The rag-service Redis Streams API surface is unchanged regardless of backing DB -- if migrating to Neo4j later, only rag-service internals change.
- Vector dimension is 4096, matching Qwen3-Embedding-8B default output. If Matryoshka dimensionality reduction is used, collection dimensions must be updated accordingly.
