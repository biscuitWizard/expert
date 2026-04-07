use serde::{Deserialize, Serialize};

/// Fire signal published by ssm-worker when debounce confirms a threshold
/// crossing. Consumed by the orchestrator to enter the fire queue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FireSignal {
    pub activity_id: String,
    pub stream_id: String,
    /// Goal IDs that crossed threshold.
    pub firing_goal_ids: Vec<String>,
    /// Scores at the time of confirmed firing.
    pub scores: Vec<f32>,
    /// Redis Stream sequence ID of the trigger event.
    pub trigger_event_seq: String,
    /// Redis Stream sequence ID of the last event that caused a previous fire
    /// (or stream start). Used by context-builder for XREVRANGE lookback.
    pub last_fired_seq: Option<String>,
    /// Unix ms.
    pub timestamp: u64,
}

/// Request from orchestrator to context-builder to assemble a context package.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssembleRequest {
    pub activity_id: String,
    pub stream_id: String,
    pub fire_signal: FireSignal,
    /// The activity's goal tree (all goals, not just firing ones).
    pub goal_tree: Vec<crate::goal::Goal>,
    /// Tool definitions for this activity's LLM session.
    pub tool_definitions: Vec<ToolDefinition>,
}

/// A tool the LLM can call. Feedback tools are always present;
/// domain-specific tools are configured per activity by the orchestrator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    /// JSON Schema for the tool's parameters.
    pub parameters_schema: serde_json::Value,
    /// Whether this tool targets an external system (routed to stream-ingestion)
    /// or is handled internally (feedback tools, goal updates).
    pub is_domain_tool: bool,
}

/// Goal update request from llm-gateway to orchestrator, triggered when
/// the LLM calls update_goal() or add_goal().
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalUpdateRequest {
    pub activity_id: String,
    /// Which goal to update. `None` means add a new root-level goal.
    pub target_goal_id: Option<String>,
    pub description: String,
    /// 0.0 = hard replace (resets EMA/drift), 1.0 = no change.
    pub blend_factor: f32,
    /// For add_goal: optional parent and aggregation.
    pub parent_goal_id: Option<String>,
    pub aggregation: Option<crate::goal::GoalAggregation>,
}

/// Request to encode a text string (e.g. goal description) into an embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodeRequest {
    pub request_id: String,
    pub text: String,
}

/// Result of an encoding request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodeResult {
    pub request_id: String,
    pub embedding: Vec<f32>,
}

/// Threshold hint from the LLM, routed through llm-gateway to orchestrator
/// and then to the ssm-worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdHint {
    pub activity_id: String,
    pub goal_id: String,
    pub direction: ThresholdDirection,
    pub magnitude: ThresholdMagnitude,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ThresholdDirection {
    Raise,
    Lower,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ThresholdMagnitude {
    Slight,
    Moderate,
    Strong,
}

/// Filter update request from llm-gateway to orchestrator, triggered when
/// the LLM calls update_event_filter().
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterUpdateRequest {
    pub activity_id: String,
    pub event_filter: crate::event_filter::EventFilter,
}

/// Session history summarization request from rag-service to llm-gateway.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummarizeRequest {
    pub activity_id: String,
    pub session_id: String,
    /// The raw exchange text to be summarized.
    pub raw_text: String,
}

/// Result of a summarization request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummarizeResult {
    pub activity_id: String,
    pub session_id: String,
    pub compressed_narrative: String,
}

/// Notification from training-service that a new model checkpoint is available.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointAvailable {
    pub checkpoint_id: String,
    pub domain: Option<String>,
    /// Filesystem path or object store key for the checkpoint.
    pub path: String,
    pub created_at: u64,
    /// "slow" or "medium" -- which training timescale produced this checkpoint.
    pub timescale: Option<String>,
}
