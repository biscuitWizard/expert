# ADR: Training Approach — Shared SSM Crate and Gradient Computation

## Status

**Decided.**

## Context

**training-service** must train the **same** SSM as **ssm-worker** serves in production. Duplicating model code across services risks subtle drift (wrong dimensions, mismatched feature order, incompatible checkpoints). Implementing **BPTT** (backpropagation through time) by hand for the full recurrence is easy to get wrong and expensive to validate.

## Decision

1. **Extract [expert-ssm](../../crates/expert-ssm/)** as the single shared crate for SSM types, feature computation, checkpoint serialization, and threshold logic. Both ssm-worker and training-service depend on it; checkpoints and command semantics stay aligned by construction.

2. **Initial training loop** uses **output-layer gradient updates** (matrices **C** and **D** — readout / observation mapping) rather than full manual BPTT through the recurrence. This keeps gradients tractable and reduces the risk of incorrect through-time derivatives.

3. **Future option:** adopt **candle** (or similar) **autodiff** for **full BPTT** through the SSM when we need end-to-end recurrence training; the shared crate remains the single source of truth for the forward graph.

## Consequences

- **Simpler, more correct** gradient story for the first shipping loop; fewer moving parts than hand-rolled BPTT.
- **One checkpoint format** and one feature definition path — fewer deployment mismatches.
- **Straightforward extension** later: swap or augment the trainer with autodiff without forking model code.

## Related

- [SSM model architecture (minimal linear SSM)](ssm-architecture.md)
