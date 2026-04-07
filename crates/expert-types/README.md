# expert-types

Canonical data types shared across all Expert services. This crate defines the contracts between services -- every struct that crosses a service boundary via Redis Streams is defined here.

## Modules

- **`event`** -- `Event`, the canonical post-adapter event struct. Every external event is normalized to this format by stream-ingestion before entering the pipeline.
- **`goal`** -- `Goal`, `GoalAggregation`, the per-activity goal types. Goals are atomic and per-activity (not shared between activities).
- **`activity`** -- `ActivityState`, `ActivityLifecycle`, `PendingFireState`, the full serializable state of an SSM activity instance.
- **`context`** -- `ContextPackage`, `Exchange`, the server-side context assembly types and the rendered output format.
- **`training`** -- `TrainingExample`, `LabelSource`, the training store record types.
- **`signals`** -- `FireSignal`, `AssembleRequest`, `ToolDefinition`, `GoalUpdateRequest`, and other inter-service message types.

## Design Principles

- All types derive `Serialize` and `Deserialize` for Redis Stream transport.
- Embedding vectors are `Vec<f32>` (4096-dim, Qwen3-Embedding-8B) for serialization. Performance-critical paths in ssm-worker and expert-vectors convert to `ndarray` views at the boundary.
- IDs are `String` (UUID v4 formatted). The `uuid` crate is used for generation; string representation is the wire format.
- Timestamps are `u64` Unix milliseconds throughout.
- No service-specific logic lives here -- only data definitions and trivial constructors.
