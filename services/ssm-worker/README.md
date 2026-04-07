# ssm-worker

Service overview for the **Semantic State Machine (SSM) worker** in the Expert attention fabric: the computational hot path for live, event-driven AI agents.

## Responsibility

The worker pool forms the execution core. Each of `N` worker instances hosts many **activities** concurrently. Per embedded event, each affected activity runs:

- SSM recurrence (shared hidden state, per-goal scores)
- Input feature computation (relevance, drift, surprise, timing)
- Scoring and threshold comparison
- Activity **lifecycle finite-state machine** (FSM)
- **Pending-fire debounce** before emitting invocation signals
- **Adaptive thresholds** driven by suppress/recall statistics and optional LLM hints

The worker implements **FAST** timescale adaptation: hidden-state accumulation and goal conditioning on every event, without gradient steps (no training in the hot path).

## Scaling model

**Pooled.** Workers register with the orchestrator and advertise capacity. The orchestrator assigns activities to workers and issues lifecycle commands. **Stream affinity** is the primary placement heuristic: co-locating activities on the same physical stream reduces subscription fan-out and keeps the inner loop cache-friendly.

## Ownership

Per activity, the worker owns:

- SSM model weights
- Hidden state `H`
- Per-goal threshold state `theta`
- Lifecycle FSM state
- Feature computation state (baselines, EMAs, centroids, covariance estimates for surprise)
- Baseline statistics used for normalization and surprise

## I/O

| Direction | Channel |
|-----------|---------|
| **Consumes** | `events.embedded.{stream_id}`, `commands.worker.{worker_id}` |
| **Publishes** | `signals.fire` |

## Dependencies

- **[expert-ssm](../../crates/expert-ssm/)** — shared crate for SSM recurrence (`ssm.rs`), input features (`features.rs`), checkpoint serialization, and threshold logic (moved out of the service binary for reuse with training-service).
- **Redis** — embedded event streams and durable activity state
- **Orchestrator** — registration, capacity, assignments, suspend/resume, goal updates

Per-worker **commands** on `commands.worker.{worker_id}` include lifecycle as today plus **`checkpoint_reload`** (load a published checkpoint into the worker’s model state) and **`threshold_update`** (apply new threshold parameters from training or orchestration).

## Statefulness

**Yes.** SSM hidden states and hot-path statistics live in **process memory** for assigned activities. Durable snapshots are written to Redis on suspend or for operational recovery, per the design contract.

## Performance

The inner loop must sustain **sub-millisecond** processing **per event per activity** under target load, so that stream fan-out across activities remains bounded by hardware, not accidental algorithmic cost.

## Further reading

See [DESIGN.md](./DESIGN.md) for registration, event fan-out, feature vector definition, recurrence, lifecycle, debounce, adaptive thresholds, serialization, multi-timescale adaptation, and goal-matrix hot updates.
