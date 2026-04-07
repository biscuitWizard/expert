# ADR: Response Coordination Across Shared Streams

## Status

**Proposed.** Problem documented; solution not yet selected.

## Context

Multiple activities can share a `stream_id`. The architecture explicitly supports this: the SSM worker fans out each event to all activities assigned to the same stream, and the orchestrator prefers stream-affinity worker placement to enable this efficiently.

The problem is that independent activities on the same stream can independently fire, independently invoke the LLM, and independently publish domain actions back to the shared `actions.{stream_id}` stream. Nothing in the current architecture prevents duplicate or conflicting responses to the same event.

### Concrete scenario

A Discord adapter ingests messages from a server into `events.raw.discord-server-1`. Two activities exist:

- **Activity A** watches for combat-related events (goal: "alert on threats")
- **Activity B** watches for quest progression (goal: "alert on quest updates")

A message arrives: *"The goblin attacks you! Your quest log updates: kill 3/5 goblins."*

Both activities score this event highly against their respective goals. Both enter `PENDING_FIRE`. Both pass debounce. Both publish independent `FireSignal` messages to `signals.fire`. The orchestrator processes both (the per-activity lifecycle check passes for each). Both trigger context assembly, LLM invocation, and domain tool calls. The user sees two separate bot replies to the same message.

### Code path trace

The gap is visible across four services:

**1. SSM worker fan-out** (`ssm-worker/src/main.rs`): Every embedded event is delivered to every activity on the matching stream. Each activity scores independently.

**2. Independent fire signals** (`ssm-worker/src/main.rs`): After scoring, each activity that crosses its threshold produces its own `FireSignal` published to `signals.fire`. There is no cross-activity deduplication.

**3. Orchestrator fire handling** (`orchestrator/src/workers.rs`): `handle_fire_signal` checks only the firing activity's lifecycle state (`Active` or `ColdStart`). It has no concept of "another activity on the same stream is already in flight." The per-activity depth-1 queue and global capacity cap are throughput limiters, not coordination mechanisms.

**4. Domain action dispatch** (`llm-gateway/src/tools.rs`): Domain tools publish to `actions.{stream_id}` -- a stream keyed by source, not by activity. If two activities both call domain tools, both actions are published to the same stream and both are executed by the adapter.

### Relationship to event filters

A companion design ([event-filters](../designs/event-filters.md)) addresses the separate problem of *sensory isolation* -- ensuring activities only receive events matching a declared predicate on event metadata. Event filters reduce the overlap surface (Activity A only sees DMs from Alice, Activity B only sees DMs from Bob), but they do not eliminate it. Activities with overlapping or identical filters -- or activities that intentionally share the full event stream -- still face the duplicate-response problem.

## Options

### Option A: Per-stream fire lock at orchestrator

When the orchestrator accepts a fire signal for an activity on stream S, it acquires a short-lived lock (Redis key or in-memory mutex) keyed by `stream_id`. Subsequent fire signals for other activities on the same stream are **held** until the lock releases (after the first activity's invocation completes or times out). The held signals are then re-evaluated: if the triggering event has already been responded to, they are discarded; otherwise the next one proceeds.

**Pros:**
- Centralizes coordination in the orchestrator, which already owns the fire queue and lifecycle FSM.
- No changes needed in SSM workers, context-builder, or llm-gateway.
- Straightforward to implement: a `HashMap<String, StreamLock>` in the orchestrator's registry, checked in `handle_fire_signal`.

**Cons:**
- Serializes all activity firings on a shared stream. If Activity A fires on an unrelated event, Activity B is blocked even though there is no actual conflict.
- Lock duration is tied to LLM invocation latency (seconds to tens of seconds), creating a significant window where legitimate fires are delayed.
- Does not distinguish between "same trigger event" and "different trigger events that happen to be on the same stream." Two activities firing on genuinely different events should not block each other.
- Requires timeout and deadlock handling for the lock.

### Option B: Activity groups with fire arbitration

Activities on the same stream are organized into **groups** (explicit or implicit by shared `stream_id`). When multiple activities in a group have pending fire signals within a configurable window, the orchestrator runs an **arbitration** step: it selects the best candidate (highest score, most specific goal, or operator-defined priority) and dispatches only that one. Losing candidates either have their fire dropped or deferred to a subsequent window.

**Pros:**
- Allows multiple activities to coexist on a stream without serialization -- only concurrent fires on overlapping events are arbitrated.
- The orchestrator can use rich signals for arbitration (score magnitude, goal specificity, activity priority, recent firing history).
- Natural extension of the existing fire queue logic.
- Operator-configurable: priority rules, arbitration window, and tie-breaking policy.

**Cons:**
- Requires defining "concurrent" -- a time window or event-sequence window within which fires are considered competing. Too short and fires slip through; too long and legitimate sequential fires are falsely grouped.
- Adds complexity to the orchestrator: group tracking, arbitration logic, deferred-fire lifecycle.
- Losing activities still updated their hidden state as if they were going to fire. Dropping their fire is a form of information loss (the SSM entered refractory without the LLM getting to provide feedback). May require a new lifecycle state or a "silent refractory" that differs from post-fire refractory.
- Groups must be maintained as activities are created and destroyed.

### Option C: Action deduplication at stream-ingestion

The stream-ingestion adapter tracks recently executed actions (by trigger event sequence, tool name, or a content hash). Before executing an outbound action, it checks whether an equivalent action was already executed within a deduplication window. Duplicates are dropped or logged.

**Pros:**
- Does not require any changes to the SSM worker, orchestrator, or llm-gateway. Fully localized to stream-ingestion.
- Catches duplicates regardless of their source -- even if the orchestrator dispatches both fires, only one action reaches the external system.
- Simple to reason about: a dedup cache with a TTL.

**Cons:**
- Reactive, not preventive. Both LLM invocations still happen, consuming LLM slots and producing episodes, labels, and session history entries. The wasted work is significant.
- Determining "equivalent action" is non-trivial. Two LLM responses to the same event will likely produce different text. Exact dedup catches only identical payloads; fuzzy dedup (semantic similarity on action content) adds latency and complexity to the hot path.
- Does not help with non-action side effects: both invocations still write training labels, episodes, and session history. The RAG graph and training store accumulate duplicates.
- The dedup window is a magic number. Too short and racing actions slip through; too long and legitimate repeated actions are suppressed.

### Option D: Shared refractory per stream

After any activity fires on a stream, all other activities on the same stream enter a **shared refractory** period. During shared refractory, other activities continue updating hidden state but cannot enter `PENDING_FIRE`. This is analogous to the existing per-activity refractory but applied at stream scope.

**Pros:**
- Simple model: one fire per stream per refractory window.
- Directly prevents duplicate responses at the SSM level, before fire signals are even produced.
- Can be implemented in the SSM worker with a per-stream timestamp, requiring no orchestrator changes.

**Cons:**
- Blunt instrument. If Activity A fires on event 100 and enters refractory, Activity B cannot fire until the shared refractory expires -- even if event 105 is highly relevant to B and completely irrelevant to A.
- Refractory duration is currently a per-activity config. A shared refractory needs its own duration, which may conflict with per-activity needs.
- Does not scale to streams with many activities and diverse goals. The more activities share a stream, the longer the effective blackout window.
- Cross-activity coordination in the SSM worker adds shared mutable state between activity instances that are currently independent.

### Option E: LLM-level awareness (context enrichment)

The context package is enriched with information about other activities on the same stream: their goals, whether they are currently in-flight, and recent actions they have taken. The LLM is instructed (via system prompt) to check whether another activity has already handled the triggering event and to avoid redundant responses.

**Pros:**
- No architectural changes to the pipeline. Pure data enrichment in the context-builder.
- The LLM can make nuanced decisions: "Activity A already responded about combat, but I should add quest context" vs. "Activity A already said everything relevant."
- Graceful degradation: if the LLM ignores the coordination context, the system still works (duplicates happen, but nothing breaks).

**Cons:**
- Relies on the LLM to consistently self-coordinate. LLMs are unreliable at following meta-instructions, especially under prompt pressure. This is the same alignment risk as trusting the LLM to label training data accurately.
- Adds context window consumption. Every activity's goals, state, and recent actions must be serialized into the prompt. With many activities this is expensive.
- Race condition: if two LLMs are invoked near-simultaneously, neither sees the other's in-flight response. The "other activity actions" context is stale by definition.
- Does not prevent the wasted LLM invocation itself -- both calls happen, and the losing one must still produce a coherent response (even if that response is "nothing to add").

## Recommendation (preliminary)

Options are not mutually exclusive. A layered approach may be appropriate:

1. **Option B (arbitration)** as the primary mechanism -- it operates at the right level (orchestrator fire queue), has access to the richest signals, and can be tuned per deployment.
2. **Option C (action dedup)** as a safety net -- cheap to add in stream-ingestion, catches anything that slips through arbitration.
3. **Option E (LLM awareness)** as a soft enhancement -- useful even without the coordination problem, since activities benefit from knowing what their siblings are doing.

Options A and D are simpler but too coarse for streams with diverse activities.

## Open Questions

- Should arbitration be per-trigger-event (compare fires on the same event sequence number) or per-time-window (compare fires within N ms of each other)?
- When a fire is dropped by arbitration, should the losing activity enter refractory (preventing near-future re-fire) or return to `ACTIVE` (allowing it to fire on the next event)?
- Should activity priority be explicit (operator-assigned rank) or implicit (derived from goal hierarchy level, score magnitude, or recent firing success rate)?
- How does this interact with the Initiative module (spec Section 5.4)? Initiative-driven fires bypass the SSM score mechanism but still produce actions on the shared stream.

## Consequences (deferred)

To be documented once a solution is selected.
