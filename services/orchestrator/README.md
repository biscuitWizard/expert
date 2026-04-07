# Orchestrator

Control plane service for the Expert attention fabric. The orchestrator governs how live, event-driven AI activities are created, scheduled, and torn down, and how goals, workers, and execution pressure interact across the system.

## Responsibility

- **Activity lifecycle**: Spawn, suspend, resume, and terminate activities. Activities are the unit of attention; each has its own execution context and state machine.
- **Goal tree CRUD**: Goals are atomic and scoped per activity. The orchestrator maintains goal trees in its registry and coordinates updates with encoding, persistence, and workers.
- **Worker pool assignment**: Owns the assignment table mapping activities to workers and applies placement policy (affinity, load, capacity).
- **Fire queue and backpressure**: Owns the fire queue and enforces backpressure so the system does not overload LLM or worker capacity.
- **Tool sets**: Assigns and tracks the tool set available to each activity (domain tools plus mandatory feedback tools).
- **Worker health**: Monitors workers via heartbeats and reacts when workers become unhealthy.
- **Goal update propagation**: Coordinates the pipeline when goals change (encode, persist, notify hosting worker).
- **Meta-learning (MEDIUM timescale)**: Triggers MEDIUM-timescale meta-learning in response to system-level signals and activity evolution.

## Owns

- Activity registry
- Per-activity goal trees
- Per-activity tool set definitions
- Worker pool assignment table (`activity_id` to `worker_id`)
- Fire queue and backpressure policy
- System configuration relevant to scheduling and activity policy

## Consumes

| Source | Role |
|--------|------|
| `signals.fire` | Fire signals that drive when an activity should be considered for LLM invocation |
| `requests.goal_update` | Goal updates originating from the LLM path (e.g. via llm-gateway) |
| `results.encode` | Encoded representations used to reconcile registry state after goal changes |
| Worker heartbeats | Liveness and capacity signals from ssm-workers |
| Management API | Human and CLI-driven control (create, suspend, inspect, etc.) |

## Publishes

| Destination | Role |
|-------------|------|
| `commands.worker.{worker_id}` | Commands to the assigned worker for a given activity |
| `requests.context` | Requests for context assembly relevant to invocation |
| `requests.encode` | Requests to encode goal updates before persistence and fan-out |
| `goals.write` | Writes for rag-service to persist goal state |

## Dependencies

- **Redis**: Durable state for the activity registry, assignment table, and related coordination; pub/sub for heartbeats and channels where applicable.
- **rag-service**: Goal persistence aligned with retrieval and long-lived goal storage.

## Stateful

Yes. The activity registry and worker assignment table are authoritative orchestrator state and are persisted in Redis so placement and lifecycle survive process restarts (subject to deployment configuration).

## API surface

HTTP REST management API for operators and **expert-cli**. See [DESIGN.md](./DESIGN.md) for endpoint-level behavior and system interactions.
