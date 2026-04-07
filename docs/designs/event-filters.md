# Design: Event Filters

## Overview

Activities declare filters over event metadata fields to restrict which events on a shared stream are delivered to their SSM instance. This provides sensory isolation: an activity monitoring Discord DMs from Alice does not process messages from Bob's channel, even though both flow through the same `events.embedded.discord` stream.

Event filters are the primary mechanism for preventing activities from receiving irrelevant events. They complement but do not replace the response coordination mechanisms described in [ADR: Response Coordination](../decisions/response-coordination.md), which handles the case where multiple activities with overlapping filters legitimately fire on the same event.

## Motivation

Streams are keyed by external source (`events.raw.{stream_id}`), not by activity. A single Discord adapter produces one stream containing messages from all channels, DMs, and threads. Without filtering, every activity on that stream receives every message and must rely entirely on SSM scoring to ignore irrelevant events. This wastes compute (feature computation and SSM updates on events the activity will never care about) and creates the preconditions for duplicate responses.

Adapters already populate `Event.metadata` with structured fields (channel ID, sender, event type, thread ID, etc.). Filters leverage this existing metadata to gate event delivery at the SSM worker's fan-out point, before any feature computation or scoring occurs.

## Filter Model

### Predicate structure

A filter is a tree of predicates combined with boolean operators. The leaf nodes are field-level comparisons against `Event.metadata` values.

```rust
/// A single predicate comparing a metadata field to a value.
pub struct FieldPredicate {
    /// Dot-delimited path into Event.metadata (e.g. "channel_id", "sender.name").
    pub field: String,
    /// Comparison operator.
    pub op: FilterOp,
    /// Value to compare against. Type must be compatible with the operator.
    pub value: serde_json::Value,
}

pub enum FilterOp {
    /// Field equals value (string, number, or bool).
    Eq,
    /// Field does not equal value.
    Ne,
    /// Field value is in a set of values (value must be a JSON array).
    In,
    /// Field value is not in a set of values.
    NotIn,
    /// Field exists in metadata (value is ignored).
    Exists,
    /// Field does not exist in metadata (value is ignored).
    NotExists,
    /// String field contains substring (value must be a string).
    Contains,
    /// String field matches a regex pattern (value must be a string).
    Matches,
}
```

### Composition

Predicates compose into a tree using `And` and `Or` nodes:

```rust
pub enum EventFilter {
    /// All children must match.
    And(Vec<EventFilter>),
    /// At least one child must match.
    Or(Vec<EventFilter>),
    /// Negate a child filter.
    Not(Box<EventFilter>),
    /// Leaf predicate on a metadata field.
    Field(FieldPredicate),
    /// Match all events (no filtering). Default when no filter is specified.
    All,
}
```

An activity with no filter (or `EventFilter::All`) receives all events on its stream, preserving backward compatibility.

### JSON representation

Filters are serialized as JSON for the REST API and Redis commands. The JSON schema uses a tagged union:

```json
{
  "and": [
    { "field": "channel_id", "op": "eq", "value": "dm-alice-123" },
    { "field": "event_type", "op": "ne", "value": "typing_indicator" }
  ]
}
```

```json
{
  "or": [
    { "field": "channel_id", "op": "eq", "value": "dm-alice-123" },
    { "field": "channel_id", "op": "eq", "value": "dm-alice-456" }
  ]
}
```

```json
{ "field": "sender.name", "op": "ne", "value": "self" }
```

Omitting the filter field entirely or passing `null` is equivalent to `EventFilter::All`.

### Examples

**Discord: isolate a DM conversation**
```json
{
  "and": [
    { "field": "channel_id", "op": "eq", "value": "dm-alice-123" },
    { "field": "author_id", "op": "ne", "value": "bot-self-id" }
  ]
}
```

**MUD telnet: only combat-related lines**
```json
{
  "or": [
    { "field": "event_type", "op": "eq", "value": "combat" },
    { "field": "raw_prefix", "op": "contains", "value": "attacks" }
  ]
}
```

**Market data: specific ticker**
```json
{ "field": "symbol", "op": "in", "value": ["AAPL", "MSFT", "GOOG"] }
```

## Enforcement Point

Filtering is enforced in the **SSM worker** at the event fan-out step. This is the narrowest chokepoint where events are matched to activities, and filtering here prevents wasted feature computation, SSM updates, and downstream fire signals.

Current fan-out code (`ssm-worker/src/main.rs`, `process_event`):

```rust
let activity_ids: Vec<String> = ws
    .activities
    .iter()
    .filter(|(_, a)| a.stream_id == event.stream_id)
    .map(|(id, _)| id.clone())
    .collect();
```

With event filters, the fan-out adds a predicate check:

```rust
let activity_ids: Vec<String> = ws
    .activities
    .iter()
    .filter(|(_, a)| {
        a.stream_id == event.stream_id && a.event_filter.matches(&event.metadata)
    })
    .map(|(id, _)| id.clone())
    .collect();
```

The `matches` method on `EventFilter` evaluates the predicate tree against the event's metadata map. It is a pure function with no I/O, allocation-light (operates on references into the existing `HashMap<String, serde_json::Value>`), and terminates in time proportional to the predicate tree depth times the number of metadata fields accessed.

### Why not filter earlier?

**At ingestion:** The adapter does not know which activities exist or what their filters are. Filtering at ingestion would require the adapter to maintain a synchronized copy of all activity filters, coupling stream-ingestion to the orchestrator's registry.

**At the encoder:** The encoder processes events before they reach activities. It has no activity context. Filtering here would prevent embedding computation for events no activity cares about, but would require the encoder to also maintain filter state, and would break if a new activity is created after the event was already dropped.

**At the orchestrator:** The orchestrator does not sit on the hot event path. Events flow directly from encoder to SSM worker via Redis Streams. Adding the orchestrator as an intermediary would add latency and a point of failure to the hot path.

The SSM worker is the correct enforcement point: it already owns the activity-to-stream mapping, processes events in the hot loop, and is the last point before compute-intensive feature computation begins.

## Required Changes

### 1. `expert-types`: new `EventFilter` type

Add `event_filter.rs` to the `expert-types` crate with the types described above (`EventFilter`, `FieldPredicate`, `FilterOp`) and a `matches(&self, metadata: &HashMap<String, serde_json::Value>) -> bool` implementation.

The `matches` method handles:
- Dot-delimited field paths: `"sender.name"` traverses nested JSON objects in the metadata map.
- Type coercion: numeric comparisons work across JSON number types; string comparisons are case-sensitive by default.
- Missing fields: a predicate on a non-existent field returns `false` (except `NotExists`, which returns `true`).
- `Matches` operator: compiled regex is cached per `FieldPredicate` instance (lazy initialization) to avoid recompilation on every event.

### 2. `expert-types`: update `ActivityState`

Add an `event_filter: EventFilter` field to `ActivityState`. Defaults to `EventFilter::All` for backward compatibility. Serialized to Redis alongside the rest of the state.

### 3. Orchestrator: API and registry changes

**`POST /activities`** gains an optional `event_filter` field in the request body. The orchestrator validates the filter (well-formed JSON, known operators, non-empty field names) and stores it in the `ManagedActivity` record.

**`PUT /activities/{id}/filter`** (new endpoint) allows updating the filter on a running activity. The orchestrator:
1. Updates the registry.
2. Sends a `filter_update` command to the assigned SSM worker via `commands.worker.{worker_id}`.

**`GET /activities/{id}`** includes the filter in the response.

The `assign` command sent to SSM workers when creating an activity includes the `event_filter` alongside `activity_id`, `stream_id`, and `goals`.

### 4. SSM worker: filter storage and evaluation

`ActivityInstance` gains an `event_filter: EventFilter` field, populated from the `assign` command. The `process_event` fan-out loop (described above) checks the filter before delivering events to each activity.

A new `filter_update` command type in `handle_command` updates the filter on a live activity without requiring reassignment.

### 5. CLI: filter support

`expert-cli create-activity` gains an optional `--filter` argument accepting a JSON string or file path. The filter is passed to the orchestrator in the activity creation request.

`expert-cli status` displays the active filter for each activity in the status output.

## Validation Rules

The orchestrator validates filters on submission:

- `field` must be a non-empty string.
- `op` must be a recognized `FilterOp` variant.
- `Eq`, `Ne`, `Contains`: `value` must be a string, number, or bool.
- `In`, `NotIn`: `value` must be a JSON array.
- `Exists`, `NotExists`: `value` is ignored (may be omitted or null).
- `Matches`: `value` must be a string that compiles as a valid regex.
- `And`, `Or`: must contain at least one child.
- `Not`: must contain exactly one child.
- Maximum nesting depth: 8 levels (prevents pathological filter trees).
- Maximum total predicates: 64 (prevents expensive evaluation on every event).

## Performance Considerations

Filter evaluation runs on every event for every activity on that stream, in the SSM worker's hot loop. The evaluation must be fast:

- **Field lookup**: `HashMap::get` on the metadata map is O(1). Dot-path traversal adds a nested lookup per path segment, bounded by path depth.
- **String comparison**: standard equality or `contains` on typically short metadata strings.
- **Regex**: compiled once, cached on the `FieldPredicate`. Execution time depends on pattern complexity; the validation step should reject pathological patterns (catastrophic backtracking). Consider using `regex::Regex` with a size/complexity limit or a timeout.
- **Short-circuit evaluation**: `And` returns `false` on first failing child. `Or` returns `true` on first passing child.

For the expected case (1-5 predicates per filter, simple equality checks on short strings), evaluation cost is negligible compared to the feature computation and SSM update that follow.

## Interaction with Response Coordination

Event filters reduce but do not eliminate the need for response coordination:

- **Disjoint filters**: Activities with non-overlapping filters (e.g., different `channel_id` values) will never fire on the same event. No coordination needed.
- **Overlapping filters**: Activities with partially overlapping filters (e.g., one filters by channel, the other filters by event type) may both receive and fire on the same event. Coordination is still needed.
- **Identical filters / no filters**: Multiple activities on the same stream with the same (or no) filter receive identical event sequences. Full coordination is required.

Event filters should be the **first line of defense** -- use them to isolate activities wherever the event metadata supports it. Response coordination (per the companion ADR) handles the residual overlap.
