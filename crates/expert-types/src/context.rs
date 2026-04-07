use serde::{Deserialize, Serialize};

use crate::event::Event;
use crate::goal::Goal;

/// A single LLM exchange: the prompt sent and the response received,
/// including any tool calls made.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Exchange {
    /// Unix ms when the invocation started.
    pub timestamp: u64,
    pub rendered_prompt: String,
    pub response: String,
    pub tool_calls: Vec<ToolCall>,
}

/// A tool call made by the LLM during an invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub tool_name: String,
    pub arguments: serde_json::Value,
    pub result: Option<serde_json::Value>,
}

/// An episode node as stored in the RAG graph. Represents one complete
/// LLM invocation cycle: trigger, context, response, feedback, outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub id: String,
    pub activity_id: String,
    pub goal_id: String,
    pub domain: Option<String>,
    /// Embedding for semantic retrieval (derived from trigger event + goal).
    pub embedding: Vec<f32>,
    pub trigger_event_id: String,
    pub trigger_scores: Vec<f32>,
    pub rendered_prompt: String,
    pub response: String,
    pub tool_calls: Vec<ToolCall>,
    /// Whether the LLM called suppress() on this invocation.
    pub was_suppressed: bool,
    /// Event indices the LLM flagged via recall().
    pub recalled_event_indices: Vec<usize>,
    /// Unix ms.
    pub created_at: u64,
}

/// A pattern node in the RAG graph. Distilled summary of recurring
/// episode types, created by the consolidation background job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    pub id: String,
    pub domain: Option<String>,
    pub embedding: Vec<f32>,
    pub summary: String,
    pub episode_count: u64,
    pub created_at: u64,
    pub updated_at: u64,
}

/// Wrapper carrying an exchange with its activity ID, for the centralized
/// `exchanges.all` stream where multiple activities share a single stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityExchange {
    pub activity_id: String,
    pub exchange: Exchange,
}

/// Server-side context assembly object. The `rendered_prompt` is what
/// the LLM receives; all other fields are retained server-side for
/// tool call resolution (e.g. recall() needs access to event embeddings).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPackage {
    pub activity_id: String,
    pub firing_goals: Vec<Goal>,
    pub trigger_event: Event,
    pub trigger_scores: Vec<f32>,
    /// Last N uninvoked events with embeddings -- retained server-side
    /// so recall(event_indices) can dereference to actual event data.
    pub recent_events: Vec<Event>,
    pub retrieved_episodes: Vec<Episode>,
    pub compressed_history: Option<String>,
    pub recent_raw_exchanges: Vec<Exchange>,
    /// The natural language prompt assembled per spec Section 6.2.
    pub rendered_prompt: String,
    /// Per-activity tool set (feedback tools + domain-specific tools).
    pub tool_definitions: Vec<crate::signals::ToolDefinition>,
}
