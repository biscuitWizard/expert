use serde::{Deserialize, Serialize};

use crate::event_filter::EventFilter;

/// Activity lifecycle states. The FSM enforces sequential invocation:
/// an activity processes one LLM invocation at a time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ActivityLifecycle {
    Uninitialized,
    ColdStart,
    Active,
    PendingFire,
    Fired,
    Refractory,
    Suspended,
    Terminated,
}

/// Debounce state captured when one or more goal scores cross threshold.
/// The SSM continues updating hidden state during the debounce window.
/// At `recheck_at`, scores are re-evaluated to confirm the signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingFireState {
    /// Indices into the goal matrix that crossed threshold.
    pub triggered_goals: Vec<usize>,
    /// Scores at the moment of threshold crossing.
    pub scores_at_trigger: Vec<f32>,
    /// Unix ms when threshold was first crossed.
    pub triggered_at: u64,
    /// Configurable per activity, default 500ms.
    pub debounce_ms: u64,
    /// `triggered_at + debounce_ms`.
    pub recheck_at: u64,
}

/// Full serializable state of an SSM activity instance.
/// Serialized to Redis for suspend/resume. Must serialize in < 100ms
/// (spec invariant 9).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityState {
    pub activity_id: String,
    pub stream_id: String,
    pub domain: String,
    pub goal_tree_root_id: String,
    /// Flattened goal matrix G ∈ ℝ^(k×d), row-major.
    pub goal_matrix: Vec<f32>,
    /// Goal IDs corresponding to each row of G.
    pub goal_indices: Vec<String>,
    /// SSM hidden state vector.
    pub h: Vec<f32>,
    /// Per-goal adaptive thresholds θ ∈ ℝ^k.
    pub theta: Vec<f32>,
    /// Scalar EMA accumulator for smoothed average relevance.
    pub ema: f32,
    /// Rolling event centroid.
    pub centroid: Vec<f32>,
    /// Flattened covariance matrix for Mahalanobis distance.
    pub cov_matrix: Vec<f32>,
    /// Running stream mean for surprise computation.
    pub stream_mean: Vec<f32>,
    /// Recent firing history: (goal_index, timestamp) pairs.
    pub firing_history: Vec<(usize, u64)>,
    /// Rolling window suppress count.
    pub suppress_count: u32,
    /// Rolling window recall count.
    pub recall_count: u32,
    pub invocation_count: u64,
    pub event_count: u64,
    pub pending_fire: Option<PendingFireState>,
    /// Unix ms, 0 if not in refractory.
    pub refractory_until: u64,
    pub lifecycle_state: ActivityLifecycle,
    /// Unix ms.
    pub created_at: u64,
    /// Unix ms of last event processed.
    pub last_active: u64,
    /// Pointer to compressed session history in RAG graph.
    pub session_history_id: Option<String>,
    /// Event filter restricting which events on the stream are delivered
    /// to this activity. Defaults to `All` (no filtering).
    #[serde(default)]
    pub event_filter: EventFilter,
}
