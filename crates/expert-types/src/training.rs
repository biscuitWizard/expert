use serde::{Deserialize, Serialize};

/// Source of a training label.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LabelSource {
    /// LLM called suppress() -- negative label.
    LlmSuppress,
    /// LLM called recall() -- positive label.
    LlmRecall,
    /// Human-provided label (always weight 1.0, overrides LLM labels).
    Human,
    /// Synthetically generated (e.g. augmented from existing labels).
    Synthetic,
}

/// Label polarity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Label {
    Positive,
    Negative,
}

/// A labeled training example for the SSM. Written to the training store
/// by the feedback pipeline. Immutable once committed (spec invariant 6);
/// can be superseded by a new label with higher weight but never deleted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub id: String,
    pub activity_id: String,
    pub stream_id: String,
    pub domain: Option<String>,
    pub goal_id: String,
    pub goal_version: u32,
    pub goal_embedding: Vec<f32>,
    /// Serialized event window (the sequence of events around the trigger).
    pub event_window: Vec<crate::event::Event>,
    /// Event embeddings for the window.
    pub window_vectors: Vec<Vec<f32>>,
    pub label: Label,
    pub label_source: LabelSource,
    pub label_weight: f32,
    pub reason: String,
    /// Unix ms.
    pub created_at: u64,
    pub used_in_batch: bool,
    pub confidence: f32,
    pub consensus_count: u32,
}

/// Request a balanced batch of training examples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingBatchRequest {
    pub request_id: String,
    pub domain: Option<String>,
    pub goal_id: Option<String>,
    pub batch_size: usize,
    pub min_confidence: f32,
}

/// A balanced batch of training examples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingBatch {
    pub request_id: String,
    pub examples: Vec<TrainingExample>,
    pub positive_count: usize,
    pub negative_count: usize,
}
