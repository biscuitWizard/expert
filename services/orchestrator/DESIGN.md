# Orchestrator: Design

This document describes the orchestrator control plane: management API, activity lifecycle, worker assignment, fire queue behavior, goal propagation, tool sets, and worker health handling.

## 1. Management REST API

The management API is the primary entry point for human operators and **expert-cli**. It mutates and inspects orchestrator-owned state (activities, goals, lifecycle) and exposes coarse system health.

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/activities` | Create a new activity. Request body includes `stream_id`, goal tree, `domain`, and tool set configuration. |
| `GET` | `/activities` | List all activities with their current status. |
| `GET` | `/activities/{id}` | Inspect one activity: state machine state, scores, firing history, and related metadata. |
| `POST` | `/activities/{id}/suspend` | Suspend an activity: stop scheduling further work according to policy and persist state as required. |
| `POST` | `/activities/{id}/resume` | Resume a previously suspended activity. |
| `DELETE` | `/activities/{id}` | Terminate an activity: release worker assignment, drain or drop pending work per policy, and remove or archive registry entries as defined by configuration. |
| `PUT` | `/activities/{id}/goals` | Replace or patch the goal tree for that activity (exact semantics follow API versioning; orchestrator validates atomic goal constraints). |
| `GET` | `/health` | System health: worker count, queue depth, LLM slots (or equivalent concurrency budget surfaced by the stack). |

All endpoints assume appropriate authentication and authorization at the edge (not defined here).

## 2. Activity lifecycle FSM

Each activity follows a sequential execution model: at most one LLM invocation is in flight at a time. The transition from **FIRED** to **REFRACTORY** is a hard gate: no new fire is accepted until refractory completes and the activity returns to a state where firing is allowed (typically **ACTIVE**).

States:

- **UNINITIALIZED**: Activity record exists but is not yet ready to receive signals or schedule work.
- **COLD_START**: Initial setup (worker assignment, context priming, etc.) before steady operation.
- **ACTIVE**: Ready to accept fire signals subject to queue and backpressure rules.
- **PENDING_FIRE**: A fire has been admitted and is waiting for an available LLM slot or worker dispatch.
- **FIRED**: An invocation is in progress on the worker / LLM path.
- **REFRACTORY**: Mandatory cooldown after a fire; no re-entry to **PENDING_FIRE** until this phase completes.
- **SUSPENDED**: Operator- or system-initiated pause; no scheduling until resumed.
- **TERMINATED**: Terminal state; no further transitions except explicit administrative reset if ever supported.

Typical happy-path loop (simplified): **UNINITIALIZED** to **COLD_START** to **ACTIVE** to **PENDING_FIRE** to **FIRED** to **REFRACTORY** and back to **ACTIVE**. **SUSPENDED** and **TERMINATED** can be entered from multiple states according to explicit transition rules in the implementation.

## 3. Worker pool assignment

Workers register at startup and advertise capacity (concurrent activities, streams, or other advertised limits as defined by the worker contract).

When an activity is spawned, the orchestrator selects a worker using a deterministic policy:

1. **Stream affinity**: Prefer a worker already subscribed to or associated with the activity's `stream_id`, to preserve locality and reduce subscription churn.
2. **Load**: Among affinity candidates, prefer the least-loaded worker by orchestrator-tracked load (e.g. assigned activity count or weighted score).
3. **Capacity**: If no worker has headroom, spawn fails or is retried per policy (HTTP error or asynchronous retry, depending on API design).

The assignment table maps `activity_id` to `worker_id` and is stored in Redis. Commands for that activity are published to `commands.worker.{worker_id}` so only the assigned worker acts on them.

## 4. Fire queue and backpressure

- **Per-activity queue**: Bounded with depth 1: at most one pending fire per activity; a newer fire replaces an older pending entry when the activity is already queued (coalescing / last-write-wins at the fire boundary).
- **Global queue**: Bounded to twice the configured **Ollama** concurrency slot count (or the system's equivalent LLM concurrency limit). When full, admission policy applies (reject or drop according to configuration).
- **Stale signals**: Fire signals older than a configurable TTL (default 30 seconds) are dropped on admission to avoid executing on obsolete events.
- **Ordering**: FIFO among admitted signals, with optional prioritization by goal hierarchy level (higher-priority goals may be ordered ahead when multiple activities compete for global slots).

Together, these rules implement backpressure: workers and LLM throughput are protected while preserving responsiveness for the most recent intent per activity.

## 5. Goal update propagation

When the LLM path (via **llm-gateway**) publishes a goal update that the orchestrator must reconcile:

1. The orchestrator publishes an encoding request to `requests.encode` with the payload required to normalize or embed the update.
2. On `results.encode`, the orchestrator updates the in-registry goal tree for the activity using the encoded result and validation rules.
3. It publishes to `goals.write` so **rag-service** can persist the authoritative goal state for retrieval and audit.
4. It sends a goal-matrix update command to the **ssm-worker** hosting that activity so in-worker state matches the registry.

Ordering and idempotency are enforced at the orchestrator and channel boundaries so duplicate or out-of-order messages do not corrupt the goal tree.

## 6. Tool set management

Each activity carries a configured tool set:

- **Feedback tools** (always included): `suppress`, `recall`, `update_goal`, `set_threshold_hint`, `add_goal`. These provide the minimum control surface for attention and goal manipulation.
- **Domain tools**: Telnet commands, HTTP/API calls, or other environment-specific tools, fixed at spawn time (or updated only through explicit API if supported).

The full tool set is passed to **llm-gateway** on each invocation request via **AssembleRequest** (or equivalent), so the model and gateway always see a consistent, activity-scoped capability list.

## 7. Worker health monitoring

Workers emit heartbeats over Redis pub/sub (or the project's equivalent heartbeat channel). The orchestrator tracks last-seen times and advertised capacity per worker.

If a worker misses heartbeats beyond a configurable threshold:

- Its assigned activities are **suspended** (or moved to a safe state per policy).
- Activity state needed for reassignment is serialized to Redis.
- Activities are **reassigned** to healthy workers using the same placement policy (affinity, load, capacity), with commands routed via the new `commands.worker.{worker_id}` topic.

This limits blast radius when a worker crashes or partitions from the control plane while keeping activity state recoverable from persisted registry data.
