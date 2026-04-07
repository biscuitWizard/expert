use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Canonical event format. Every external event is normalized to this
/// struct by stream-ingestion before entering the pipeline.
///
/// The `embedding` field is `None` when first published to `events.raw.*`
/// and populated by the encoder service before publishing to `events.embedded.*`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: String,
    pub stream_id: String,
    /// Monotonically increasing per-stream, assigned by stream-ingestion.
    pub sequence: u64,
    /// Unix milliseconds.
    pub timestamp: u64,
    /// Serialized text output from the adapter (the human-readable event content).
    pub raw: String,
    /// Populated by the encoder service. `None` in raw events.
    pub embedding: Option<Vec<f32>>,
    /// Domain-specific key-value metadata. Adapters may attach structured
    /// fields here (e.g. sender, channel, event_type).
    pub metadata: HashMap<String, serde_json::Value>,
}
