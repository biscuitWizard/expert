# expert-ssm

Shared crate for the **Semantic State Machine (SSM)** used in the Expert attention fabric: the same model definitions, feature pipeline, checkpoint format, and adaptive threshold logic must apply in **ssm-worker** (online inference) and **training-service** (offline and few-shot training).

## Purpose

- **SSM model** — `LinearSsm` and related recurrence (shared weights and state layout).
- **Feature computation** — Input features (relevance, drift, surprise, timing, etc.) aligned with the worker hot path.
- **Checkpoint serialization** — Stable on-disk / stream format for publishing and reloading checkpoints.
- **Adaptive threshold logic** — Threshold state and updates consistent between inference and training.

## Public modules

| Module | Role |
|--------|------|
| `ssm` | Core SSM recurrence and model types |
| `features` | Feature vector computation and state |
| `checkpoint` | Checkpoint read/write and versioning hooks |
| `threshold` | Per-goal threshold parameters and update rules |

## Consumers

- **[ssm-worker](../../services/ssm-worker/)** — loads checkpoints, runs inference, applies `threshold_update` and `checkpoint_reload` commands.
- **[training-service](../../services/training-service/)** — trains against the same structures and emits checkpoints for deployment.

See [DESIGN.md](../../services/ssm-worker/DESIGN.md) in ssm-worker for the full hot-path contract; training details live under [training-service](../../services/training-service/).
