use expert_types::context::Episode;
use expert_types::event::Event;
use expert_types::goal::{Goal, GoalAggregation};
use expert_types::signals::{AssembleRequest, FireSignal, ToolDefinition};
use expert_types::training::{Label, LabelSource, TrainingExample};
use std::collections::HashMap;

pub async fn redis_conn() -> redis::aio::MultiplexedConnection {
    let url = std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
    expert_redis::connect(&url)
        .await
        .expect("failed to connect to Redis")
}

pub async fn flush_redis(conn: &mut redis::aio::MultiplexedConnection) {
    let _: () = redis::cmd("FLUSHDB")
        .query_async(conn)
        .await
        .expect("FLUSHDB failed");
}

pub fn unique_stream(prefix: &str) -> String {
    format!("{prefix}.{}", uuid::Uuid::new_v4())
}

pub fn fake_event(stream_id: &str) -> Event {
    Event {
        id: uuid::Uuid::new_v4().to_string(),
        stream_id: stream_id.to_string(),
        sequence: 1,
        timestamp: 1000,
        raw: "test event".to_string(),
        embedding: Some(vec![0.1; 8]),
        metadata: HashMap::new(),
    }
}

pub fn fake_goal(name: &str, dim: usize) -> Goal {
    Goal {
        id: uuid::Uuid::new_v4().to_string(),
        name: name.to_string(),
        description: format!("Watch for {name}"),
        embedding: vec![0.1; dim],
        parent_id: None,
        children: Vec::new(),
        aggregation: GoalAggregation::Max,
        weights: None,
        domain: Some("test".to_string()),
        created_at: 1000,
        version: 1,
        active: true,
    }
}

pub fn fake_fire_signal(activity_id: &str, stream_id: &str) -> FireSignal {
    FireSignal {
        activity_id: activity_id.to_string(),
        stream_id: stream_id.to_string(),
        firing_goal_ids: vec!["goal-1".to_string()],
        scores: vec![0.85],
        trigger_event_seq: "1000-0".to_string(),
        last_fired_seq: None,
        timestamp: 2000,
        operator_forced: false,
    }
}

pub fn fake_assemble_request(activity_id: &str, stream_id: &str) -> AssembleRequest {
    let goal = fake_goal("test-goal", 8);
    AssembleRequest {
        activity_id: activity_id.to_string(),
        stream_id: stream_id.to_string(),
        fire_signal: fake_fire_signal(activity_id, stream_id),
        goal_tree: vec![goal],
        tool_definitions: vec![ToolDefinition {
            name: "suppress".to_string(),
            description: "suppress tool".to_string(),
            parameters_schema: serde_json::json!({"type": "object"}),
            is_domain_tool: false,
        }],
        bot_identity: None,
    }
}

pub fn fake_episode(dim: usize) -> Episode {
    Episode {
        id: uuid::Uuid::new_v4().to_string(),
        activity_id: "act-test".to_string(),
        goal_id: "goal-test".to_string(),
        domain: Some("test".to_string()),
        embedding: vec![0.5; dim],
        trigger_event_id: "evt-1".to_string(),
        trigger_scores: vec![0.9],
        rendered_prompt: "test prompt".to_string(),
        response: "test response".to_string(),
        tool_calls: Vec::new(),
        was_suppressed: false,
        recalled_event_indices: Vec::new(),
        created_at: 1000,
        operator_forced: false,
    }
}

pub fn fake_training_example() -> TrainingExample {
    TrainingExample {
        id: uuid::Uuid::new_v4().to_string(),
        activity_id: "act-test".to_string(),
        stream_id: "stream-test".to_string(),
        domain: Some("test".to_string()),
        goal_id: "goal-test".to_string(),
        goal_version: 1,
        goal_embedding: vec![0.1; 8],
        event_window: vec![fake_event("stream-test")],
        window_vectors: vec![vec![0.1; 8]],
        label: Label::Positive,
        label_source: LabelSource::LlmRecall,
        label_weight: 0.8,
        reason: "test reason".to_string(),
        created_at: 1000,
        used_in_batch: false,
        confidence: 0.9,
        consensus_count: 1,
    }
}
