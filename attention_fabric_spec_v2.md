# Attention Fabric & Orchestrator — Technical Specification v2

**Project:** Semantic Attention System for Live Event-Driven AI Agents  
**Purpose:** Agent alignment document — canonical description of intended architecture.  
**Changes from v1:** Goal matrix + hierarchy, Activity terminology, pending-fire debounce, time handling, context format, storage split, LLM history in context.

---

## 1. Problem Statement

Standard LLM-based agents operate in a request/response loop. This breaks down in live environments producing continuous, unbounded event streams (game telnet, market ticks, log streams, sensor feeds). This system answers:

- When should an LLM be invoked? (Not every event warrants reasoning.)
- What context should it receive? (A collated, goal-relevant package — not the raw stream.)
- How does invocation quality improve over time without human retraining?
- How do multiple concurrent activities share learned knowledge?

---

## 2. Terminology

| Term | Definition |
|---|---|
| **Activity** | The operational runtime unit. One SSM instance + one LLM session handle + a goal tree + a stream subscription + lifecycle state. The SSM+LLM "pair" from v1 is now called an activity. |
| **Goal** | A semantic objective that an activity is oriented toward. Goals are first-class objects with hierarchy. They exist independently of activities and can be referenced by multiple activities. |
| **Goal matrix** | The leaf layer of the goal tree, as a matrix `G ∈ ℝ^(k×d)`. What the SSM scores against at runtime. |
| **SSM** | Semantic State Machine — the lightweight always-on sequence model inside an activity that watches the stream and produces a score vector. |
| **Encoder** | Shared model that converts raw events to embedding vectors. Singleton, not per-activity. |
| **RAG graph** | Shared knowledge graph with vector index. Stores episodes, patterns, domain knowledge. Used for context assembly and cross-activity transfer. |
| **Training store** | Separate database for SSM training labels. Purpose-built for ML training data access patterns. Not the RAG graph. |
| **Orchestrator** | Top-level manager that spawns, allocates, monitors, and suspends activities. |
| **Salience score vector** | `S ∈ ℝ^k` — one scalar per goal, output of the SSM at each timestep. |
| **Threshold vector** | `θ ∈ ℝ^k` — per-goal firing thresholds, adaptive. |
| **Pending-fire** | An activity state where one or more goal scores crossed threshold but are in debounce wait before firing. |

---

## 3. System Overview

```
┌──────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR                         │
│   spawns · allocates · monitors · goal tree management       │
└──────────────────────────────┬───────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
     ┌──────────┐        ┌──────────┐        ┌──────────┐
     │ACTIVITY 1│        │ACTIVITY 2│        │ACTIVITY N│
     │SSM + LLM │        │SSM + LLM │        │SSM + LLM │
     │goal tree │        │goal tree │        │goal tree │
     └────┬─────┘        └────┬─────┘        └────┬─────┘
          │                   │                   │
     ┌────▼─────┐        ┌────▼─────┐        ┌────▼─────┐
     │Stream 1  │        │Stream 2  │        │Stream N  │
     └──────────┘        └──────────┘        └──────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              │
           ┌──────────────────┴─────────────────┐
           │                                    │
    ┌──────▼──────┐                   ┌─────────▼────────┐
    │  RAG GRAPH  │                   │  TRAINING STORE  │
    │episodes     │                   │labeled examples  │
    │patterns     │                   │event windows     │
    │domain KG    │                   │feedback labels   │
    └─────────────┘                   └──────────────────┘
```

**Per-activity data flow:**

```
Event arrives
  → Encoder produces eₜ
  → SSM input features computed (cosine, EMA, drift, surprise, Δt)
  → SSM state update: hₜ = f(hₜ₋₁, input, G)
  → Score vector: S = σ(C·hₜ + D)  ∈ ℝ^k
  → Per-goal threshold check: Sᵢ ≥ θᵢ for any i?
      → NO:  accumulate, continue
      → YES: enter PENDING_FIRE state
              → wait debounce_ms
              → re-evaluate score
                  → still ≥ θᵢ? → FIRE
                      → assemble context package (natural language)
                      → invoke LLM session
                      → LLM acts, optionally calls feedback tools
                      → labels written to training store
                      → enter REFRACTORY
                  → dropped below θᵢ? → DISCARD (return to ACTIVE)
```

---

## 4. Goals and Goal Hierarchy

### 4.1 Goals as First-Class Objects

Goals are not just embeddings attached to activities. They are independent objects stored in the RAG graph, versioned, referenceable by multiple activities, and composable into trees.

```typescript
interface Goal {
    id: string
    name: string                      // human-readable label
    description: string               // natural language
    embedding: Float32Array           // encoded from description
    parent_id: string | null          // null = root goal
    children: string[]                // child goal IDs
    aggregation: "max" | "weighted_sum" | "all"  // how child scores combine
    weights?: number[]                // used if aggregation = weighted_sum
    domain: string | null             // optional domain affiliation
    created_at: number
    version: number
}
```

### 4.2 Goal Hierarchy (DAG)

Goals form a directed acyclic graph. An example:

```
"Help me play this MUD"  [root]
    ├── "Alert on threats"  [aggregation: max]
    │     ├── "Enemy initiates combat"
    │     └── "Health drops below 30%"
    └── "Alert on progression"  [aggregation: any]
          ├── "Quest objective updates"
          └── "Level up event"
```

Leaf goals are the scored goals — they each have an embedding and appear as rows in the goal matrix `G`. Internal (parent) goals are not directly scored; their score is derived by aggregating their children's scores according to the aggregation rule.

The full score for a non-leaf goal is computed bottom-up after the SSM produces leaf scores:

```
score("Alert on threats") = max(
    score("Enemy initiates combat"),
    score("Health drops below 30%")
)
score("Help me play this MUD") = weighted_sum(
    score("Alert on threats"),
    score("Alert on progression")
)
```

The orchestrator can fire on any goal at any level of the hierarchy, each with its own threshold. A high-level goal firing triggers a broader LLM context; a leaf goal firing triggers a more focused one.

### 4.3 Goal Matrix

The goal matrix `G ∈ ℝ^(k×d)` is the runtime representation passed to the SSM: a matrix where each row is the embedding of one leaf goal. `k` = number of active leaf goals, `d` = embedding dimension.

The SSM's per-goal scoring:

```
Hₜ = A·Hₜ₋₁ + B·[eₜ_features]      # hidden state update (shared across goals)
sᵢ_t = σ(Cᵢ·hₜ + gᵢ·hₜ + Dᵢ)      # per-goal score using goal row gᵢ from G
S_t = [s₁_t, s₂_t, ..., sₖ_t]      # score vector
```

The hidden state `h` is shared — the SSM maintains one view of the stream. Goal-specific scoring heads read from this shared state, each biased by its goal embedding. This is efficient: one state update per event, k cheap score computations.

### 4.4 How Goals Differ from Activities

| Concept | What it is | Persists across sessions? | Has lifecycle? |
|---|---|---|---|
| **Goal** | Semantic objective (a vector + hierarchy) | Yes — stored in RAG graph | No |
| **Activity** | Runtime instance watching for goals | No — ephemeral, but state is serializable | Yes |

A goal can be assigned to many activities. An activity always has exactly one goal tree (which may have many leaf goals = many rows in G). Deleting an activity does not delete its goals. Goals accumulate training history; activities accumulate hidden state.

---

## 5. The SSM

### 5.1 Input Feature Vector

Per event, before the SSM recurrence, a feature vector is assembled:

```
cos_relevance_i  = cos(eₜ, gᵢ)                          # per-goal relevance [k values]
ema_relevance    = α·mean(cos_relevance) + (1-α)·EMAₜ₋₁  # smoothed average relevance
centroid_t       = mean(eₜ₋ₙ ... eₜ)                     # rolling window centroid
drift_magnitude  = ‖centroid_t - centroid_t₋ₙ‖            # how much stream is moving
drift_direction  = cos(centroid_t - centroid_t₋ₙ, ḡ - centroid_t) # toward goal centroid?
surprise         = mahal_dist(eₜ, μ_stream, Σ_stream)     # anomaly vs stream baseline
delta_t          = (timestamp_t - timestamp_t₋₁) / norm   # time since last event (normalized)
silence_duration = time since last event exceeding cos threshold  # inactivity signal
```

These scalars are concatenated with `eₜ` as the full SSM input:

```
input_t = [eₜ ‖ cos_relevance_1..k ‖ ema_relevance ‖ drift_magnitude ‖
           drift_direction ‖ surprise ‖ delta_t ‖ silence_duration]
```

### 5.2 Time Handling in the SSM

Time enters the SSM as **input features, not as a clock tick**. The SSM is event-driven — it updates on event arrival, not on a timer. Time is represented by:

- `delta_t` — normalized time between consecutive events. The SSM learns that unusual gaps or unusually rapid events are themselves meaningful.
- `silence_duration` — time since the last event that crossed a per-goal relevance threshold. Allows the SSM to detect "relevant events have stopped arriving," which can itself be salient.

**What the SSM does not do:** fire on a timer in the absence of events. That is the **Initiative domain** — a separate, peer concern described in Section 5.4.

### 5.3 Activity Lifecycle States

```
UNINITIALIZED
    → spawn(goal_tree, stream_id) → COLD_START

COLD_START
    → warm-up window (N events, no firing, SSM building baseline statistics)
    → ACTIVE

ACTIVE
    → Sᵢ ≥ θᵢ for any i → PENDING_FIRE

PENDING_FIRE
    → wait debounce_ms
    → re-evaluate:
        Sᵢ still ≥ θᵢ → FIRED
        Sᵢ dropped below → ACTIVE  (signal was transient)

FIRED
    → LLM session invoked
    → LLM completes response
    → REFRACTORY

REFRACTORY
    → Score accumulation continues (hidden state still updates)
    → No new PENDING_FIRE entries for refractory_ms
    → ACTIVE

SUSPENDED
    → State fully serialized to store
    → Can resume: ACTIVE
    → Can discard: TERMINATED

TERMINATED
    → State discarded. Goal tree preserved in RAG graph.
```

### 5.4 Initiative Domain (Out of Scope for SSM)

The SSM is a **reactive** system — it responds to events. The following are **initiative** behaviors and belong to a separate, peer module:

- "Fire after N minutes of stream inactivity" (idle timeout)
- "Fire on a schedule regardless of stream content" (polling)  
- "Fire at end of each trading day to summarize" (temporal agenda)
- "Check if I should fire proactively given current world state" (proactive agent behavior)

The initiative module, when it decides to invoke an LLM, does so through the same context assembly pipeline and LLM invocation path as the SSM — it shares the infrastructure but bypasses the score-threshold mechanism. This is a separate design problem and should be specified independently. Do not bake initiative logic into the SSM equation.

### 5.5 Pending-Fire Debounce

```typescript
interface PendingFireState {
    triggered_goals: number[]    // indices into goal matrix that crossed threshold
    scores_at_trigger: number[]  // scores at moment of threshold crossing
    triggered_at: number         // unix ms
    debounce_ms: number          // configurable per activity, default 500ms
    recheck_at: number           // triggered_at + debounce_ms
}
```

During `PENDING_FIRE`, the SSM continues updating hidden state. At `recheck_at`, scores are re-evaluated. If any triggered goal is still above threshold, the highest-scoring triggered goal is selected as the primary firing goal and the context package is assembled for that goal's level in the hierarchy.

If multiple goals fire simultaneously, the highest-level ancestor goal that covers all triggered leaves is used to frame the invocation — this produces a richer, more contextual LLM prompt than firing separately for each leaf.

### 5.6 Adaptive Threshold

Each goal has its own adaptive threshold `θᵢ`. Thresholds evolve based on:

```python
# After each invocation cycle completes:
recent_suppress_rate = suppresses / (suppresses + non_suppresses)  # rolling window
recent_recall_rate   = recalls / total_invocations                 # rolling window

if recent_suppress_rate > MAX_SUPPRESS_RATE:   # threshold too low
    θᵢ *= RAISE_FACTOR     # e.g. 1.05
elif recent_recall_rate > MAX_RECALL_RATE:     # threshold too high  
    θᵢ *= LOWER_FACTOR     # e.g. 0.97

# LLM hint also feeds in (see Section 7.3 tools):
θᵢ += hint_delta * HINT_WEIGHT

θᵢ = clip(θᵢ, θ_min, θ_max)
```

---

## 6. Context Package — Format

### 6.1 Format Decision: Structured Natural Language, Not JSON

The context package is rendered as a **structured natural language prompt**, not a JSON object. Reasons:

- JSON structural tokens consume context window budget wastefully
- JSON syntax primes the model toward JSON-like outputs
- Natural language sections are more robust to partial content (truncation doesn't break a bracket)
- The model is trained on natural language; it reasons better over prose than over key-value noise

Embedding vectors, IDs, and scores are used **server-side only** for retrieval and assembly. They never appear in the text passed to the LLM. What the LLM receives is the *results* of retrieval, rendered as prose.

### 6.2 Context Package Template

```
=== ACTIVITY CONTEXT ===
Activity: {activity_name}
Stream: {stream_description}

--- CURRENT GOAL ---
{goal_tree_natural_language}
[If hierarchy: parent goal followed by the specific sub-goal that fired]

--- WHAT TRIGGERED THIS ---
The following event crossed the attention threshold (confidence: {score:.0%}):

{trigger_event_natural_language}

--- RECENT STREAM (last {N} events, most recent last) ---
{recent_events_as_prose_or_list}
[These events did NOT trigger attention — include them so you can use the
recall() tool if you believe attention should have fired earlier.]

--- RELEVANT PAST CONTEXT ---
{top_k_retrieved_episodes_as_prose_summaries}
[Retrieved from memory based on relevance to current event and goal.]

--- SESSION HISTORY ---
{compressed_llm_history}
[Summary of prior LLM interactions in this activity session, including
what actions were taken and their outcomes.]

--- PREVIOUS CHAT ENTRIES ---
{n_most_recent_raw_exchanges}
[Verbatim last M exchanges for immediate continuity.]

=== END CONTEXT ===
```

### 6.3 Context Assembly Priority (Truncation Order)

If the assembled context exceeds the LLM context limit, truncate in this order (last to first):

1. Previous chat entries (reduce M)
2. Relevant past context (reduce k)
3. Recent stream events (reduce N)
4. Session history (compress further)
5. Goal description (never truncate — always present in full)
6. Trigger event (never truncate — always present in full)

### 6.4 Session History Construction

The session history is a **rolling compressed narrative** maintained across invocations:

- After each invocation, the LLM's response and any tool calls are appended to a raw exchange log
- When the raw log exceeds a size threshold, it is summarized by a lightweight LLM call into a compressed narrative
- The compressed narrative is stored in the RAG graph, keyed by activity ID and session ID
- Both the compressed narrative and the last M raw exchanges are included in the context package

This gives the LLM two levels of history: far history (compressed, lossy but long-range) and near history (verbatim, lossless but recent).

---

## 7. LLM Session and Feedback Tools

### 7.1 LLM Session Properties

The LLM session is **stateless in weights** between invocations. Continuity is entirely provided by the context package. Each invocation is a fresh LLM call with a constructed context.

The LLM has access to:
- Domain-specific tools (game commands, trading actions, log queries, etc.)
- Feedback tools (below) — present in every activity, regardless of domain

### 7.2 Feedback Tools

```typescript
suppress(reason: string): void
```
Marks the current invocation as a false positive. Writes a negative training label to the training store. Increments suppression rate monitor. Should be called when the LLM determines the triggering event was not worth attention given the current goal.

```typescript
recall(
    event_indices: number[],  // indices into recent_events in the context package
    reason: string
): void
```
Marks prior events as a missed trigger (false negative). Writes a positive training label to the training store. Should be called when the LLM notices that attention should have fired on events that did not cross the threshold. The LLM should use the "recent stream" section of its context to identify these.

```typescript
update_goal(
    description: string,      // natural language description of new/updated goal
    target_goal_id: string,   // which goal in the tree to update (or null = root)
    blend_factor: number      // 0.0 = hard replace, 1.0 = no change
): void
```
Pushes a goal update to the SSM. Hard replacement resets EMA and drift signals for that goal; soft blend preserves continuity.

```typescript
set_threshold_hint(
    goal_id: string,
    direction: "raise" | "lower",
    magnitude: "slight" | "moderate" | "strong"
): void
```
Advises the adaptive threshold mechanism. Does not directly set `θ` — feeds into the adaptive update as a weighted signal with `HINT_WEIGHT` (lower weight than observed firing rate signals).

```typescript
add_goal(
    description: string,
    parent_goal_id: string | null,
    aggregation?: "max" | "weighted_sum" | "all"
): string  // returns new goal ID
```
Adds a new goal to the activity's goal tree at runtime. Causes the goal matrix to be updated (new row added). Warm-up period applies to new goal before it can fire.

### 7.3 Alignment Risk and Mitigations

The LLM writing its own trainer's labels is an alignment risk — a lazy or misaligned LLM could game the system by suppressing everything to avoid invocation.

Mitigations:

| Risk | Mitigation |
|---|---|
| LLM suppresses excessively | Suppression rate guard: quarantine labels above MAX_SUPPRESS_RATE |
| LLM recalls excessively | Recall rate guard: flag as systematic SSM blind spot, report to orchestrator |
| LLM bias in label content | Label consensus scoring: multiple independent invocations on similar events must agree |
| LLM optimizes for wrong objective | Human labels always override LLM labels with weight = 1.0 |
| Catastrophic mislabeling | Max label write rate per invocation: cannot write > N labels in one session |
| Training data drift | Training store maintains label provenance; can audit and revert by source |

---

## 8. Storage Architecture

### 8.1 Two Stores, Not One

The training data store and the RAG graph are **separate systems** with different requirements. They must not be conflated.

| Property | RAG Graph | Training Store |
|---|---|---|
| Primary operation | Semantic retrieval, graph traversal | Append-write, batch-read |
| Query type | "Find K most similar episodes to this event+goal" | "Give me N labeled examples for this domain, balanced positive/negative" |
| Data structure | Knowledge graph + vector index | Labeled event windows with metadata |
| Schema flexibility | High — new node/edge types added as system learns | Low — fixed schema for ML pipeline |
| Suitable technology | Neo4j, Weaviate (graph+vector), Qdrant with metadata | PostgreSQL + pgvector, or a purpose-built ML data store |
| Retention policy | Long-term, episodes summarized into patterns over time | Rolling window + archival; old low-confidence labels can be pruned |
| Write pattern | Per-invocation (episode nodes), periodic (pattern consolidation) | Per-feedback-tool-call |
| Read pattern | Real-time (context assembly) | Batch (training jobs) |

### 8.2 RAG Graph Schema

**Nodes:**
- `Episode` — one LLM invocation: context, response, feedback, outcome
- `Pattern` — distilled summary of recurring episode types (created by consolidation job)
- `Domain` — knowledge about a stream type, linked to its episodes and patterns
- `Goal` — goal objects (see Section 4.1 schema)
- `Activity` — activity metadata (not full state — state is in the state store)

**Edges:**
- `Episode -[INSTANCE_OF]→ Pattern`
- `Episode -[FOR_GOAL]→ Goal`
- `Episode -[IN_DOMAIN]→ Domain`
- `Pattern -[IN_DOMAIN]→ Domain`
- `Goal -[CHILD_OF]→ Goal`
- `Activity -[HAS_GOAL]→ Goal`

**Vector index:** Episode and Pattern nodes are vector-indexed on their embedding for semantic retrieval.

### 8.3 Training Store Schema

```sql
training_examples (
    id              UUID PRIMARY KEY,
    activity_id     TEXT,
    stream_id       TEXT,
    domain          TEXT,
    goal_id         TEXT,
    goal_embedding  VECTOR(d),
    event_window    JSONB,           -- serialized event sequence
    window_vectors  VECTOR[],        -- event embeddings for the window
    label           TEXT,            -- 'positive' | 'negative'
    label_source    TEXT,            -- 'llm_suppress' | 'llm_recall' | 'human' | 'synthetic'
    label_weight    FLOAT,
    reason          TEXT,
    created_at      BIGINT,          -- unix ms
    used_in_batch   BOOLEAN,
    confidence      FLOAT,
    consensus_count INTEGER
)
```

Indices: `(domain, label, used_in_batch)` for training batch queries; `(activity_id, created_at)` for per-activity audit; vector index on `goal_embedding` for cross-domain transfer queries.

### 8.4 Activity State Store

A third store for SSM state serialization — lightweight, fast read/write, keyed by activity ID:

```typescript
// Serialized activity state (see full schema in Section 9.2)
// Suitable technology: Redis (fast resume), or a simple key-value store
// Key: activity_id
// Value: serialized ActivityState blob
```

---

## 9. Data Schemas

### 9.1 Goal (stored in RAG graph)
```typescript
interface Goal {
    id: string
    name: string
    description: string
    embedding: Float32Array
    parent_id: string | null
    children: string[]
    aggregation: "max" | "weighted_sum" | "all"
    weights?: number[]
    domain: string | null
    created_at: number
    version: number
    active: boolean
}
```

### 9.2 ActivityState (serialized, state store)
```typescript
interface ActivityState {
    activity_id: string
    stream_id: string
    domain: string
    goal_tree_root_id: string
    goal_matrix: Float32Array          // flattened G ∈ ℝ^(k×d)
    goal_indices: string[]             // goal IDs for each row of G
    h: Float32Array                    // SSM hidden state
    theta: Float32Array                // per-goal adaptive thresholds [k]
    ema: number                        // scalar EMA accumulator
    centroid: Float32Array             // rolling event centroid
    cov_matrix: Float32Array           // flattened covariance for Mahalanobis
    stream_mean: Float32Array          // running stream mean
    firing_history: number[][]         // [goal_index, timestamp] pairs, recent N
    suppress_count: number             // rolling window
    recall_count: number               // rolling window
    invocation_count: number
    event_count: number
    pending_fire: PendingFireState | null
    refractory_until: number           // unix ms, 0 if not refractory
    lifecycle_state: ActivityLifecycle
    created_at: number
    last_active: number
    session_history_id: string         // pointer to compressed history in RAG graph
}
```

### 9.3 Event (canonical, post-adapter)
```typescript
interface Event {
    id: string
    stream_id: string
    sequence: number                   // monotonic per-stream
    timestamp: number                  // unix ms
    raw: string                        // serialized text (adapter output)
    embedding?: Float32Array           // populated by encoder
    metadata: Record<string, unknown>  // domain-specific
}
```

### 9.4 ContextPackage (internal assembly object, rendered before LLM call)
```typescript
interface ContextPackage {
    // Assembly inputs (server-side only, not passed to LLM as-is)
    activity_id: string
    firing_goals: Goal[]               // goals that crossed threshold
    trigger_event: Event
    trigger_scores: number[]
    recent_events: Event[]             // last N uninvoked
    retrieved_episodes: Episode[]      // from RAG graph
    compressed_history: string         // from RAG graph
    recent_raw_exchanges: Exchange[]   // last M verbatim exchanges
    // Rendered output (what LLM receives)
    rendered_prompt: string            // natural language, assembled per Section 6.2
}
```

---

## 10. Design Invariants

Constraints that must be preserved across all implementation decisions:

1. **The encoder is domain-agnostic and shared.** All domain specialization lives in the SSM, not the encoder. The encoder is a singleton service.

2. **The SSM is reactive, not initiative.** It responds to events. Time-based proactive firing is a separate initiative module that shares the invocation infrastructure but bypasses the SSM score mechanism.

3. **The SSM never calls the LLM.** The SSM produces a score vector and signals PENDING_FIRE. Invocation logic lives in the orchestrator. This keeps the SSM lightweight and testable in isolation.

4. **The LLM never sees raw stream data or embedding vectors.** It receives the rendered natural language context package only. Raw streams may be large, noisy, and structurally confusing to the LLM.

5. **Goals are independent of activities.** Deleting or suspending an activity does not delete its goals. Goals accumulate history and can be reused across activities and sessions.

6. **All training labels are immutable once committed.** Labels can be superseded (new label for same window with higher weight) but not deleted. This preserves audit trail.

7. **Suppress and recall are advisory, not binding.** LLM feedback influences training data but does not directly modify SSM weights or thresholds in real time. Adaptation is asynchronous to maintain stability.

8. **Goal embeddings are versioned.** When a goal changes, the old embedding is stored with its effective time range. Training data is tagged with the goal version active at time of generation.

9. **Activity suspension must complete in < 100ms.** Serializing the hidden state vector is the only large artifact. Suspension is non-blocking.

10. **Training store and RAG graph are separate systems.** They have different schemas, access patterns, and suitable technologies. Merging them is a false optimization that would compromise both.

11. **Context is natural language, not JSON.** The rendered prompt passed to the LLM contains no raw JSON structures, embedding vectors, or internal IDs. These live server-side only.

---

## 11. Open Questions

Decisions required before implementation:

- **SSM base architecture:** Mamba, Mamba-2, or hybrid SSM/attention? Tradeoff: pure SSMs are fastest at inference; hybrids may score more accurately on semantically complex streams.
- **Encoder selection:** Distil-BERT, purpose-trained small encoder, or hosted embedding API? Tradeoff: API = no maintenance cost, but per-event latency and cost. Streaming events may require batching.
- **RAG graph technology:** Neo4j (graph-native, rich traversal) vs. Weaviate/Qdrant (vector-native with metadata graph). Decision depends on how heavily graph traversal (episode → pattern → domain) is used vs. pure vector retrieval.
- **Training store technology:** PostgreSQL + pgvector vs. a dedicated ML data platform. Depends on expected label volume and training job complexity.
- **Debounce duration policy:** Fixed ms, adaptive based on stream cadence, or event-count-based? Fixed is simplest; event-count-based is rate-agnostic.
- **Multi-activity goal sharing:** Can two activities share a goal tree? If yes, updates and feedback from one activity affect the other. Requires a shared-goal locking or versioning policy.
- **Session history compression model:** Dedicated small LLM for summarization, or use the activity's LLM with a summarization prompt? Latency and cost tradeoff.
- **Multi-tenancy isolation:** Do activities from different users share the RAG graph and training store? Affects both privacy guarantees and the quality of cross-user transfer learning.
- **Initiative module specification:** Defined as out-of-scope here but required before the system is complete. Needs its own spec document.

---

## 12. Glossary

| Term | Definition |
|---|---|
| Activity | Runtime unit: SSM instance + LLM session handle + goal tree + stream subscription + lifecycle |
| Goal | Semantic objective, first-class object, hierarchical, independent of activities |
| Goal matrix | `G ∈ ℝ^(k×d)` — leaf goals as rows, the SSM scores against this at runtime |
| Goal tree | DAG of goals with aggregation rules; root = top-level objective |
| SSM | Semantic State Machine — lightweight always-on sequence model inside an activity |
| Encoder | Shared singleton model converting raw events to embedding vectors |
| Salience score vector | `S ∈ ℝ^k` — one scalar per leaf goal, SSM output at each timestep |
| Threshold vector | `θ ∈ ℝ^k` — per-goal adaptive firing thresholds |
| Pending-fire | Activity state: one or more goal scores crossed threshold, in debounce wait |
| Refractory | Activity state: post-fire cooldown, no new firing, hidden state still updates |
| RAG graph | Shared knowledge graph + vector index: episodes, patterns, domain knowledge, goals |
| Training store | Separate database for SSM labeled training examples |
| Activity state store | Key-value store for serialized SSM hidden state (fast suspend/resume) |
| Context package | Assembled inputs for LLM invocation, rendered as structured natural language |
| Session history | Rolling compressed narrative of prior LLM exchanges in an activity session |
| Suppress | LLM feedback tool: current invocation was a false positive |
| Recall | LLM feedback tool: prior events in lookback window should have triggered firing |
| Δt | Time delta between consecutive events, encoded as SSM input feature |
| Initiative | Separate module for time-based proactive firing; not part of the SSM |
| Orchestrator | Top-level manager: spawns, allocates, monitors, suspends activities |
| Domain | Category of stream type (mud, market, logs) used to organize specialization in RAG graph |
