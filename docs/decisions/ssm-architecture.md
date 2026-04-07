# ADR: SSM Model Architecture

## Status

**Decided: Option A (minimal linear SSM).** Upgrade path to selective scan (Mamba) preserved via trait boundary, triggered by empirical evidence of scoring plateau with sufficient training data.

## Context

The SSM (Semantic State Machine) is the core computational model in the ssm-worker. It maintains a hidden state that updates on every event, producing per-goal salience scores. The architecture choice affects:
- **Inference speed** (must sustain < 1ms per event per activity)
- **Scoring quality** (accurately detecting salient events given goal conditioning)
- **Trainability** (must support MAML-style meta-learning for MEDIUM timescale and full offline retraining for SLOW timescale)
- **State size** (hidden state must serialize in < 100ms for suspend/resume)

The spec (Section 5) defines the recurrence as:
```
H_t = A * H_{t-1} + B * input_features    (shared hidden state update)
s_i_t = sigmoid(C_i * h_t + g_i * h_t + D_i)  (per-goal score)
```

This is a linear recurrence with goal-conditioned scoring heads.

## Options Evaluated

### Option A: Minimal hand-rolled linear SSM (SELECTED)

A direct implementation of the spec's equations. The state transition is a linear recurrence: `h_t = A * h_{t-1} + B * x_t` where A and B are learned matrices. Scoring heads are per-goal linear projections with sigmoid activation.

- **Pros:** Simplest to implement. Fastest inference (single matrix multiply per event). Easiest to debug and test. Smallest state. Trivial serialization. Converges on very few training labels.
- **Cons:** Limited expressiveness. May struggle with complex temporal patterns that require selective memory (knowing what to remember and what to forget).
- **Meta-learning:** MAML works straightforwardly. Linear SSMs have well-behaved gradients and smooth loss landscapes, which is ideal for few-shot adaptation.

### Option B: Mamba (Structured State Space with Selective Mechanisms)

Input-dependent (selective) state transitions: A and B are functions of the input, allowing the model to decide what information to retain at each step.

- ~4x more parameters than Option A.
- Requires hundreds to thousands of labeled examples to learn meaningful selectivity.
- No mature Rust crate exists. Requires implementing the selective scan in Rust or binding to C/CUDA.

### Option C: Mamba-2 / SSD (State Space Duality)

Reformulates selective state space with attention duality for GPU tensor core utilization.

- At single-step inference (our use case: one event at a time), Mamba-2 reduces to the same recurrence as Mamba-1. The SSD parallel form only benefits training throughput over full sequences, not per-event inference.
- Requires Rust-to-CUDA integration (candle/burn/cudarc), creating cross-system boundaries that are expensive to debug and fragile in CI.
- GPU memory management, driver dependencies, and Python-to-Rust weight format parity introduce entire bug classes absent from CPU-only code.

### Option D: Hybrid SSM + Attention

SSM for state transition plus attention over a window of recent states.

- Breaks O(1) per-event property. Largest state to serialize. Most parameters.

## Decision: Option A

### Why the feature engineering makes complex models unnecessary

The SSM input vector (spec Section 5.1) is heavily engineered:

- **Per-goal cosine similarity** -- direct semantic relevance signal
- **EMA relevance** -- temporal smoothing of relevance
- **Drift magnitude and direction** -- stream is changing, and toward/away from goals
- **Mahalanobis surprise** -- anomaly detection against stream baseline
- **Delta_t** -- temporal spacing signal
- **Silence duration** -- relevant events have stopped arriving

These features already capture semantic relevance, temporal dynamics, anomaly detection, and stream characterization. The SSM's job on top of this is to learn a **temporal weighting** -- "this combination of features over the last N events means fire." A linear recurrence can learn that weighting. Mamba's selective scan lets the model learn what to remember and what to forget at each step, but the feature engineering has already pre-solved that problem: relevance is explicitly computed, not latent.

### Why the self-training bootstrap favors simplicity

The system starts with **zero training labels**. Labels arrive through the feedback loop: SSM fires -> LLM evaluates -> LLM calls suppress()/recall() -> label written to training store. This means:

- Labels trickle in at the rate of firing events -- one per invocation cycle
- A linear SSM with ~(d^2 + k*d) parameters converges on 50-100 labels. Useful scoring within the first hours of operation.
- A selective scan with ~4x more parameters needs hundreds to thousands of labels to learn meaningful selectivity. Until then, the selective gates are either random noise or collapsed to constant values, in which case it **is** a linear SSM but worse-initialized.
- MAML few-shot adaptation (MEDIUM timescale) works best when the model is small and the loss landscape around the initialization is smooth. More parameters means harder MAML convergence and worse few-shot adaptation -- the opposite of what we need.

### Why integration complexity is a disproportionate risk

Option A (linear SSM on CPU) crosses **zero system boundaries**: pure Rust, pure `ndarray`, pure math. Debuggable by printing `h_t` and inspecting values. No external runtime dependencies. Trivially testable in CI.

Options B/C introduce system boundary crossings per integration point:
- Rust <-> CUDA (driver versions, device synchronization, memory lifecycle)
- Python training <-> Rust inference (weight format, tensor layout, numerical precision)
- GPU memory <-> CPU memory (state serialization on suspend/resume)
- Batched activity inference (one corrupted state in a batch can poison others)

Each boundary is a class of bug that is hard to reproduce, hard to test in CI, and disproportionately expensive to resolve through iterative debugging.

### Performance headroom

| Architecture | Estimated per-update (CPU, d=256, k=8, embed=4096) | Headroom at 10k updates/sec |
|---|---|---|
| Linear SSM + projection | ~10-20 us (projection dominates) | ~5-10x margin |
| Mamba + projection | ~30-60 us | ~1.5-3x margin |
| Mamba-2 + projection | ~30-60 us (CPU, same as Mamba) | ~1.5-3x margin |

The linear SSM has the most headroom for scaling to more activities per worker.

## Implementation Specification

### SsmCore trait

The ssm-worker isolates the SSM behind a trait so the recurrence implementation can be swapped without changing the event loop, lifecycle management, serialization, or feature computation:

```rust
pub trait SsmCore: Send + Sync {
    /// Process one event: update hidden state, return per-goal scores.
    fn update(&mut self, input: &[f32], goal_matrix: &GoalMatrix) -> Vec<f32>;

    /// Export hidden state for serialization (suspend/resume).
    fn state(&self) -> &[f32];

    /// Restore hidden state from serialized form.
    fn load_state(&mut self, state: &[f32]);

    /// Reset to initial state (for COLD_START after weight reload).
    fn reset(&mut self);

    /// Load new model weights from a checkpoint.
    fn load_weights(&mut self, checkpoint: &SsmCheckpoint) -> Result<()>;
}
```

### LinearSsm implementation

The initial and currently selected implementation:

```rust
pub struct LinearSsm {
    // State transition matrices
    a: Array2<f32>,     // [hidden_dim, hidden_dim] -- state transition
    b: Array2<f32>,     // [hidden_dim, input_dim]  -- input projection

    // Per-goal scoring head parameters
    c: Array2<f32>,     // [k, hidden_dim] -- goal-independent score projection
    d: Array1<f32>,     // [k] -- bias per goal

    // Hidden state
    h: Array1<f32>,     // [hidden_dim] -- current hidden state
}
```

Per-event update:
```
h_t = A * h_{t-1} + B * input_features
s_i = sigmoid(C_i . h_t + g_i . h_t + D_i)   for each goal i
```

The embedding dimension is 4096 (Qwen3-Embedding-8B). A learned projection `P: [hidden_dim, 4096]` reduces the input to the SSM's hidden dimension. Cosine similarity and scalar features are computed in the original 4096 space, then concatenated with the projected embedding to form the SSM input vector.

Parameter count for hidden_dim=256, k=8, input_dim=256+8+5=269 (projected embedding + k cosine + 5 scalars):
- P: 256 * 4096 = 1,048,576 (input projection from embedding space)
- A: 256 * 256 = 65,536
- B: 256 * 269 = 68,864
- C: 8 * 256 = 2,048
- D: 8
- **Total: ~1,185,032 parameters** (~1.13M, dominated by the projection layer)

The projection layer P is large relative to the SSM core but is a single matrix multiply applied once per event before the recurrence. It does not affect the hidden state size or serialization cost.

State size: 256 floats = 1 KB. Serialization is trivially under 100ms.

### Weight initialization

With no training data available at first deployment:
- **A** initialized as a scaled identity matrix (stable recurrence, slow decay of state)
- **B** initialized with small random values (Xavier/Glorot)
- **C** initialized with small random values
- **D** initialized to zero (no bias)

This produces an SSM that passes information through (stable hidden state) but scores near 0.5 for all goals. Initial firing is driven primarily by the cosine similarity features and the initial threshold settings. As labels accumulate, the SSM learns to weight the features meaningfully.

### Training integration

- **SLOW timescale:** training-service exports labeled batches. Training is a standard supervised loop: forward pass over event windows, compute loss (binary cross-entropy on scores vs labels), backprop, update weights. Implemented in Python with PyTorch for the training loop. Checkpoint exported as raw float arrays (safetensors or numpy). Rust loads via the `SsmCheckpoint` format.
- **MEDIUM timescale:** MAML few-shot. Same Python training code with a MAML outer loop. The small parameter count makes MAML convergence fast and reliable.
- **FAST timescale:** Pure hidden state accumulation in Rust. No training code involved.

## Upgrade Trigger

Revisit this decision when **all** of the following are true:
1. The training store contains > 5,000 labeled examples across the target domain
2. The linear SSM's scoring quality has measurably plateaued (suppress rate and recall rate are stable but suboptimal)
3. Feature engineering improvements have been explored and exhausted
4. The full pipeline (all 8 services) is stable and well-tested

At that point, implement Mamba (Option B) behind the same `SsmCore` trait. The upgrade is localized to one struct and its trait implementation. Everything else -- event loop, fan-out, lifecycle, serialization, adaptive thresholds, debounce -- is unchanged.

## Consequences

- Initial implementation uses `ndarray` for matrix operations with a linear SSM. Pure CPU, zero external runtime dependencies.
- The `SsmCore` trait boundary preserves the upgrade path. Swapping to Mamba later is a localized change in the ssm-worker crate.
- First model weights are initialized for stability (scaled identity A), not trained. Early scoring quality depends on feature engineering and initial thresholds.
- The self-training feedback loop begins producing useful labels immediately. The linear SSM converges to useful scoring within ~50-100 labels.
- MAML meta-learning is validated first with the linear SSM, where it works best (few parameters, smooth loss landscape).
- No GPU dependencies in the ssm-worker. GPU resources are reserved for llamacpp.
