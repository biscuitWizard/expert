# Training Service: Design

This document specifies the training store schema, label ingestion, consensus scoring, batch and meta-learning APIs, SLOW offline retraining, MEDIUM few-shot provisioning, and label provenance rules for the training service.

## 1. Training store schema (PostgreSQL + pgvector)

The canonical table for labeled examples:

```sql
CREATE TABLE training_examples (
    id              UUID PRIMARY KEY,
    activity_id     TEXT NOT NULL,
    stream_id       TEXT NOT NULL,
    domain          TEXT,
    goal_id         TEXT NOT NULL,
    goal_version    INTEGER NOT NULL,
    goal_embedding  vector,
    event_window    JSONB NOT NULL,
    window_vectors  JSONB NOT NULL,
    label           TEXT NOT NULL CHECK (label IN ('positive', 'negative')),
    label_source    TEXT NOT NULL,
    label_weight    REAL NOT NULL DEFAULT 1.0,
    reason          TEXT NOT NULL,
    created_at      BIGINT NOT NULL,
    used_in_batch   BOOLEAN NOT NULL DEFAULT false,
    confidence      REAL NOT NULL DEFAULT 0.0,
    consensus_count INTEGER NOT NULL DEFAULT 0
);
```

**Indices and usage**

- Composite index on `(domain, label, used_in_batch)` to support efficient training batch queries (balanced positives/negatives, coverage via `used_in_batch`).
- Composite index on `(activity_id, created_at)` for audit and time-ordered inspection.
- Vector index on `goal_embedding` (implementation-specific: e.g. IVFFlat or HNSW per pgvector configuration) for cross-domain transfer and similarity-based retrieval.

`event_window` and `window_vectors` hold the serialized event span and precomputed embeddings associated with that span, as produced upstream; overlap detection for consensus uses these fields per policy (e.g. equality keys, interval overlap, or embedding distance thresholds defined in configuration).

## 2. Label ingestion

1. **Subscribe** to `labels.write` (or equivalent queue/stream contract used by the deployment).
2. **Validate** required fields: activity/stream/goal identifiers, window payloads, label class, source, reason, timestamps, and embedding dimensions where applicable.
3. **Assign** a new `id` (UUID) for each accepted message.
4. **Insert** a row into `training_examples` with `confidence` and `consensus_count` initialized according to Section 3 after any post-insert consensus pass.

**Immutability (spec invariant 6)**  
Committed rows are immutable: no `DELETE` and no in-place edits that would erase history. Corrections are modeled by **supersession**: insert a new row for the same logical window with updated `label_weight` (or conflicting label, subject to human-review policy), preserving the prior row for audit. Consumers that need a single "current" label apply deterministic resolution rules (e.g. highest weight, human over LLM per Section 7).

## 3. Consensus scoring

Per **spec Section 7.3**, multiple independent invocations on similar events should converge. After inserting a new label:

1. **Query** for existing `training_examples` that share the same `goal_id` (and `goal_version` unless policy allows cross-version aggregation) and whose `event_window` overlaps the new row's window, using the same overlap definition as in production (shared configuration with inference path where possible).
2. If labels **agree** on class (`positive` / `negative`) and the number of agreeing rows reaches a configured **consensus threshold**, increment or set `consensus_count` and raise `confidence` according to a monotone function (e.g. step or logistic in count).
3. If labels **disagree**, flag the set for **human review** (e.g. review queue row, status bit, or external ticket integration); do not silently overwrite.

Idempotency: duplicate deliveries of the same logical label must not double-count toward consensus; deduplication keys should be derived from producer identity + window hash + goal id where available.

## 4. Batch read API

**Purpose**: Provide **balanced** training batches for SSM training: **N** positive and **N** negative examples for a given `domain`, optionally filtered by `goal_id` or similarity to a query embedding.

**Behavior**

- Select candidates using `(domain, label, used_in_batch)` and optional goal filters; apply random or stratified sampling within each class to meet **N**.
- Optionally supplement with **cross-domain transfer** candidates: nearest neighbors by `goal_embedding` similarity subject to domain and confidence floors.
- After a batch is handed to a consumer, mark included rows with `used_in_batch = true` to track coverage and avoid over-reuse unless policy allows reuse (configurable).

The API should expose parameters: `domain`, `batch_size` (interpreted as N per class unless documented otherwise), optional `goal_id` or query vector, and optional filters on `confidence` / `label_source`.

## 5. SLOW timescale: offline retraining

Periodic or threshold-triggered **full** retraining (configurable interval, e.g. weekly, or when total new labels since last run exceeds a threshold):

1. **Export** a training batch from the store (same balancing and filtering rules as Section 4, possibly at larger scale).
2. **Run SSM training** either in-process or as a subprocess, with resource limits and logging defined by deployment.
3. **Save** the resulting model checkpoint to **checkpoint storage** owned by this service (path layout and versioning documented for the orchestrator).
4. **Publish** a `CheckpointAvailable` (or equivalent) message to `checkpoints.available` so the **orchestrator** can coordinate deployment to workers.

Failure handling: failed runs do not advance the "last successful checkpoint" pointer; partial artifacts are cleaned up or quarantined per operational policy.

## 6. MEDIUM timescale: meta-learning data

When the **orchestrator** (or another control-plane component) requests few-shot examples for a **new domain**:

1. **Query** `training_examples` for the **N** highest-`confidence` rows matching the target `domain`, and/or nearest neighbors by `goal_embedding` to similar goals from other domains.
2. **Return** a **balanced** set (positives and negatives) suitable for **MAML-style** fine-tuning (few gradient steps): small, high-signal, and class-balanced unless the caller specifies otherwise.

This path is **synchronous**: a request–response API or RPC, not a background job. Latency targets and caching are implementation details but must not block SLOW retraining jobs on the same process if isolation is required (separate pools or read replicas).

## 7. Label provenance

Every row records **`label_source`**: e.g. `llm_suppress`, `llm_recall`, `human`, `synthetic` (exact enum may be extended with migration).

**Rules**

- **Human** labels use **`label_weight` = 1.0** by default and **override** LLM-sourced labels in resolution policies where both exist for the same window (supersession or tie-break).
- **LLM** labels may use weights < 1.0 when the producer attaches uncertainty.
- The store is a **full audit trail**: analytics and training pipelines may **filter** or **reweight** by `label_source` without deleting rows.

Together with Section 2, this preserves traceability from checkpoint back to originating feedback channel.
