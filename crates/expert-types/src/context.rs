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
    /// Whether this episode was triggered by operator force-fire (high-confidence training data).
    #[serde(default)]
    pub operator_forced: bool,
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

/// A node in the self-knowledge graph stored in Qdrant. Represents one piece
/// of Zero's understanding of itself -- core identity, preferences, capabilities,
/// or reflections. Retrieved via semantic search against the current context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfKnowledgeNode {
    pub id: String,
    /// One of: "core_identity", "preference", "capability", "reflection".
    pub category: String,
    pub content: String,
    #[serde(default)]
    pub embedding: Vec<f32>,
    pub created_at: u64,
    pub updated_at: u64,
}

/// A single turn in a per-channel conversation buffer, stored in Redis.
/// Lightweight struct for short-term conversational memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    /// "user" or "assistant"
    pub role: String,
    pub content: String,
    pub timestamp: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_knowledge_node_serde_roundtrip() {
        let node = SelfKnowledgeNode {
            id: "sk-1".to_string(),
            category: "core_identity".to_string(),
            content: "I am Zero, an autonomous expert system.".to_string(),
            embedding: vec![0.1, 0.2, 0.3],
            created_at: 1000,
            updated_at: 2000,
        };
        let json = serde_json::to_string(&node).unwrap();
        let back: SelfKnowledgeNode = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "sk-1");
        assert_eq!(back.category, "core_identity");
        assert_eq!(back.content, "I am Zero, an autonomous expert system.");
        assert_eq!(back.embedding, vec![0.1, 0.2, 0.3]);
        assert_eq!(back.created_at, 1000);
        assert_eq!(back.updated_at, 2000);
    }

    #[test]
    fn self_knowledge_node_embedding_defaults_empty() {
        let json = r#"{
            "id": "sk-2",
            "category": "preference",
            "content": "I prefer concise answers.",
            "created_at": 1000,
            "updated_at": 1000
        }"#;
        let node: SelfKnowledgeNode = serde_json::from_str(json).unwrap();
        assert!(node.embedding.is_empty());
        assert_eq!(node.category, "preference");
    }

    #[test]
    fn self_knowledge_node_all_categories() {
        for category in &["core_identity", "preference", "capability", "reflection"] {
            let node = SelfKnowledgeNode {
                id: format!("sk-{category}"),
                category: category.to_string(),
                content: format!("Test {category}"),
                embedding: Vec::new(),
                created_at: 0,
                updated_at: 0,
            };
            let json = serde_json::to_string(&node).unwrap();
            let back: SelfKnowledgeNode = serde_json::from_str(&json).unwrap();
            assert_eq!(back.category, *category);
        }
    }
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
    /// Whether this invocation was triggered by an operator force-fire.
    #[serde(default)]
    pub operator_forced: bool,
}
