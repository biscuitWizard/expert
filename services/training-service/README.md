# Training Service

API layer over the training store for the Expert attention fabric. The service consumes labeled training examples, exposes batch reads for SSM training jobs, manages label provenance and consensus scoring, and enforces label immutability. It orchestrates **SLOW** timescale work (offline retraining, model checkpoints) and supports **MEDIUM** timescale work (few-shot labeled example sets for MAML-style fine-tuning).

## Responsibility

- **Training store API**: Single service boundary for PostgreSQL + pgvector persistence of training examples and related metadata.
- **Label ingestion**: Consumes committed labels from upstream producers and persists them with validation and provenance.
- **Batch reads**: Serves balanced training batches for SSM training and tracks which examples have been used in batches.
- **Consensus and confidence**: Scores agreement across independent labels on overlapping event windows for the same goal; surfaces disagreement for human review.
- **Immutability**: Enforces that committed labels are never deleted (spec invariant 6); supersession is modeled by new rows with higher weight or explicit policy, not in-place mutation.
- **SLOW timescale**: Schedules or triggers full offline retraining, checkpoint emission, and publication of availability events for downstream deployment.
- **MEDIUM timescale**: Answers synchronous queries for high-confidence, balanced few-shot sets suitable for meta-learning fine-tuning.

## Owns

- Training store schema (PostgreSQL + pgvector)
- Label validation rules at ingestion
- Consensus scoring logic and thresholds for confidence and review flags
- SLOW offline retraining pipeline (orchestration of export, training invocation, checkpoint write)
- MEDIUM meta-learning data provisioning (query and balancing for few-shot sets)
- Model checkpoint storage location and naming contract (as exposed to consumers)

## Consumes

| Source | Role |
|--------|------|
| `labels.write` | Inbound labeled training examples and metadata after upstream validation |

## Publishes

| Destination | Role |
|-------------|------|
| `checkpoints.available` | Availability of new model checkpoints after offline retraining completes (e.g. `CheckpointAvailable` for orchestrator-driven rollout) |

## Dependencies

- **PostgreSQL** with **pgvector** extension for relational storage and vector similarity over goal embeddings.

## Stateful

Yes. Authoritative state lives in the backing database (examples, labels, batch usage flags, checkpoint references as applicable).

## API surface

HTTP and/or internal RPC for batch reads, meta-learning queries, and operator-facing inspection are defined in [DESIGN.md](./DESIGN.md).
