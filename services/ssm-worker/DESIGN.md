# ssm-worker — Design

Detailed design for worker registration, the per-event inner loop, SSM mathematics, lifecycle, debounce, adaptive thresholds, persistence, adaptation timescales, and goal-matrix updates.

---

## 1. Worker registration and assignment

On startup, each worker instance registers with the **orchestrator**, advertising its identity and **capacity** (how many activities it can host concurrently).

The worker subscribes to a dedicated command channel:

```
commands.worker.{worker_id}
```

The orchestrator sends **assignment**, **suspend**, **resume**, and related control messages on this channel. For every command, the worker **acknowledges** receipt (and outcome) on the agreed reply path so the orchestrator can drive deterministic lifecycle management and retries.

Stream placement follows the pooled model: the orchestrator chooses a worker (stream affinity first), then the worker binds stream subscriptions and activity state accordingly.

---

## 2. Event loop and stream fan-out

For each **distinct** `stream_id` on which the worker has at least one assigned activity, the worker maintains a subscription to:

```
events.embedded.{stream_id}
```

When an embedded event arrives for `stream_id`, the worker iterates **all activities** currently assigned to that stream and, for each activity, executes the full per-event pipeline (features → SSM → scoring → FSM → debounce → optional publish). This iteration is the **inner loop** and dominates steady-state CPU.

Activities on other streams do not run for that event; fan-out is **per stream**, not global.

---

## 3. SSM input feature vector

For each event at time `t` and each activity, build a fixed-layout feature block from the current embedding `e_t`, goal matrix rows `g_i`, history, and timestamps. Below, `k` is the number of leaf goals; `alpha` is an EMA coefficient; norms and means are as defined by implementation constants.

**Per-goal cosine relevance (k values):**

```
cos_relevance_i = cos(e_t, g_i)    for i = 1..k
```

**EMA of mean cosine relevance:**

```
ema_relevance_t = alpha * mean(cos_relevance_1..k) + (1 - alpha) * ema_relevance_{t-1}
```

**Rolling centroid and drift:**

```
centroid_t = rolling_mean(recent embeddings)

drift_magnitude_t = || centroid_t - centroid_{t-n} ||

drift_direction_t = cos( centroid_t - centroid_{t-n}, goal_centroid - centroid_t )
```

where `goal_centroid` is an aggregate of goal directions (e.g., mean of `g_i`) and `n` is a configured lag in events or time.

**Surprise (Mahalanobis distance in embedding space):**

```
surprise_t = mahalanobis( e_t; mu_stream, Sigma_stream )
```

using stream-local mean `mu_stream` and covariance `Sigma_stream` (or diagonal/spherical approximations if used).

**Inter-arrival normalization:**

```
delta_t = ( timestamp_t - timestamp_{t-1} ) / norm
```

with `norm` a fixed time scale (e.g., seconds or stream-specific statistic).

**Silence per goal:**

```
silence_duration_i = time_since_last_event_where( cos(e, g_i) >= relevance_threshold_i )
```

The **full SSM input** is the concatenation of engineered scalars above with the raw embedding `e_t` (and any fixed positional encodings the model expects), in a stable column order shared by training and inference.

---

## 4. SSM recurrence

One **shared** hidden state per activity; one transition per event; `k` lightweight score heads. Let `input_features` denote the full vector from Section 3 (including the `e_t` block and engineered scalars).

**Hidden update:**

```
H_t = A * H_{t-1} + B * input_features
```

`A` and `B` are activity-local or shared per deployment policy.

**Per-goal scores** — goal row `g_i` from `G` couples to `H_t` (implementation fixes whether `g_i * H_t` is Hadamard, inner product, or another bilinear map consistent with training):

```
s_i_t = sigmoid( C_i * H_t + g_i * H_t + D_i )
```

**Score vector:**

```
S_t = [ s_1_t, s_2_t, ..., s_k_t ]
```

Complexity: **one** state update, **`k`** small score computations per event. The hidden state is **shared** across goals; only the score heads differ.

---

## 5. Lifecycle FSM

The activity lifecycle is a strict finite-state machine. Valid states:

| State | Role |
|-------|------|
| `UNINITIALIZED` | Constructed, not yet bound to stream/goals |
| `COLD_START` | Warm-up window: first `N` events; statistics and baselines update; **no firing** |
| `ACTIVE` | Normal scoring; eligible for `PENDING_FIRE` |
| `PENDING_FIRE` | Threshold crossed; debounce window open |
| `FIRED` | Fire signal committed; handoff to orchestrator/invocation path |
| `REFRACTORY` | Post-invocation cooldown; scores may still update; no new pending-fire |
| `SUSPENDED` | Serialized to Redis; no stream processing |
| `TERMINATED` | Terminal; resources released |

Transitions follow orchestrator commands (e.g., suspend) and internal guards (debounce completion, refractory expiry).

---

## 6. Pending-fire debounce

When **any** `s_i_t >= theta_i` while in `ACTIVE`, transition to `PENDING_FIRE`. Record:

- set of triggered goal indices
- scores at trigger time

Start a timer for `debounce_ms` (default **500 ms**). When the timer fires:

- If all previously triggered goals still satisfy `s_i >= theta_i` (per policy: may require **all** or **any** — default **any** crossing remains salient): emit a **`FireSignal`** on `signals.fire` and enter the post-fire path (`FIRED` / `REFRACTORY` per product rules).
- If scores have fallen below threshold (transient spike): return to **`ACTIVE`** without publishing.

No second `PENDING_FIRE` is started while one is pending for the same activity unless the design explicitly allows restacking (default: **single** pending window).

---

## 7. Adaptive thresholds

Each goal `i` maintains a scalar threshold `theta_i`. After each **invocation cycle** (or on a bounded schedule tied to feedback ingestion), update using **suppress** and **recall** rates computed from operator or LLM feedback:

- If **suppress rate** `> MAX_SUPPRESS_RATE`:  
  ```
  theta_i *= RAISE_FACTOR    # e.g. 1.05
  ```
- If **recall rate** `> MAX_RECALL_RATE`:  
  ```
  theta_i *= LOWER_FACTOR    # e.g. 0.97
  ```

**LLM threshold hints** (when provided) apply a **weighted delta**:

```
theta_i += w * hint_delta_i
```

Finally:

```
theta_i = clip( theta_i, theta_min, theta_max )
```

Adaptive updates are **asynchronous** to the sub-ms inner loop; they run on feedback boundaries, not per event.

---

## 8. State serialization

For **suspend** and operational migration, the worker serializes **full `ActivityState`** to Redis:

```
SET state:{activity_id} -> serialized_blob
```

The blob includes: hidden state, thresholds, FSM state, debounce timers metadata, feature statistics, goal-matrix snapshot handle, and any worker-local counters required for resume. Target: complete persistence in **< 100 ms**.

On **resume**, deserialize and restore the lifecycle to the saved phase (e.g., `SUSPENDED` → `ACTIVE` under orchestrator command), re-subscribe to `events.embedded.{stream_id}`, and continue processing without resetting learned short-term statistics unless explicitly commanded.

---

## 9. Three-timescale adaptation

| Timescale | Mechanism | Effect |
|-----------|-----------|--------|
| **FAST** | Hidden state `H_t` accumulation + goal conditioning via `g_i` in score heads | Every event; no gradient; no checkpoint I/O |
| **MEDIUM** | Load new weight checkpoint after **meta-learning** fine-tuning | Reload matrices for affected activities; reinitialize those activities to **`COLD_START`** |
| **SLOW** | Load new weight checkpoint after **offline retraining** | Same as medium: atomic swap of weights + **`COLD_START`** for affected activities |

FAST is continuous in memory. MEDIUM and SLOW are explicit **checkpoint** events coordinated with the orchestrator so stream assignments and Redis snapshots stay consistent.

---

## 10. Goal matrix hot update

The goal matrix `G` has rows `g_i`. When the orchestrator sends a **goal update** command for goal index `i`:

1. **In-place row update:** write the new embedding into row `i` of `G` without tearing down the activity.
2. **Blend policy:** if `blend_factor = 0.0`, **reset** goal-specific accumulators: **EMA** and **drift** terms tied to that goal (and any goal-local silence clocks as defined).
3. If `blend_factor > 0`, blend old and new embeddings:

```
g_i_new := (1 - blend_factor) * g_i_old + blend_factor * g_i_incoming
```

(optionally renormalize). Surrounding rows and shared `H_t` continue unless cold-start is requested.

Acknowledge the command after `G` and dependent statistics are consistent for the next event.

---

## Summary

The worker is **stateful**, **stream-partitioned**, and **assignment-driven**: Redis carries durability and streams; the orchestrator carries policy; the inner loop stays **O(k) per event** with strict latency targets. This document is the behavioral contract for implementations of `ssm-worker` in the Expert attention fabric.
