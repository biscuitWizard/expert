use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;
use tracing::{info, warn};

use expert_redis::StreamProducer;
use expert_redis::names;
use expert_types::context::ContextPackage;
use expert_types::signals::{GoalUpdateRequest, ToolDefinition};
use expert_types::training::{Label, LabelSource, TrainingExample};

pub struct ToolRouter<'a> {
    package: &'a ContextPackage,
    max_labels: usize,
    label_count: usize,
    suppressed: bool,
    recalled_indices: Vec<usize>,
}

impl<'a> ToolRouter<'a> {
    pub fn new(package: &'a ContextPackage, max_labels: usize) -> Self {
        Self {
            package,
            max_labels,
            label_count: 0,
            suppressed: false,
            recalled_indices: Vec::new(),
        }
    }

    pub fn was_suppressed(&self) -> bool {
        self.suppressed
    }

    pub fn recalled_indices(&self) -> &[usize] {
        &self.recalled_indices
    }

    pub async fn execute(
        &mut self,
        tool_name: &str,
        arguments: &Value,
        producer: &mut StreamProducer,
    ) -> Value {
        match tool_name {
            "suppress" => self.handle_suppress(arguments, producer).await,
            "recall" => self.handle_recall(arguments, producer).await,
            "update_goal" => self.handle_update_goal(arguments, producer).await,
            "add_goal" => self.handle_add_goal(arguments, producer).await,
            "set_threshold_hint" => self.handle_threshold_hint(arguments, producer).await,
            _ => {
                // Domain tool -- log but don't execute for MVP
                info!(tool = tool_name, "domain tool call (not executing in MVP)");
                serde_json::json!({"status": "acknowledged", "note": "domain tool execution not available in MVP"})
            }
        }
    }

    async fn handle_suppress(&mut self, arguments: &Value, producer: &mut StreamProducer) -> Value {
        if self.label_count >= self.max_labels {
            return serde_json::json!({"error": "label limit reached for this invocation"});
        }

        let reason = arguments
            .get("reason")
            .and_then(|r| r.as_str())
            .unwrap_or("no reason given")
            .to_string();

        self.suppressed = true;

        let now = now_ms();
        let example = TrainingExample {
            id: uuid::Uuid::new_v4().to_string(),
            activity_id: self.package.activity_id.clone(),
            stream_id: self.package.trigger_event.stream_id.clone(),
            domain: self
                .package
                .firing_goals
                .first()
                .and_then(|g| g.domain.clone()),
            goal_id: self
                .package
                .firing_goals
                .first()
                .map(|g| g.id.clone())
                .unwrap_or_default(),
            goal_version: self
                .package
                .firing_goals
                .first()
                .map(|g| g.version)
                .unwrap_or(1),
            goal_embedding: self
                .package
                .firing_goals
                .first()
                .map(|g| g.embedding.clone())
                .unwrap_or_default(),
            event_window: self.package.recent_events.clone(),
            window_vectors: self
                .package
                .recent_events
                .iter()
                .filter_map(|e| e.embedding.clone())
                .collect(),
            label: Label::Negative,
            label_source: LabelSource::LlmSuppress,
            label_weight: 0.8,
            reason,
            created_at: now,
            used_in_batch: false,
            confidence: 0.8,
            consensus_count: 1,
        };

        if let Err(e) = producer.publish(names::LABELS_WRITE, &example).await {
            warn!(error = %e, "failed to publish suppress label");
        }

        self.label_count += 1;
        info!(activity_id = %self.package.activity_id, "suppress() recorded");
        serde_json::json!({"status": "recorded", "label": "negative"})
    }

    async fn handle_recall(&mut self, arguments: &Value, producer: &mut StreamProducer) -> Value {
        if self.label_count >= self.max_labels {
            return serde_json::json!({"error": "label limit reached for this invocation"});
        }

        let indices: Vec<usize> = arguments
            .get("event_indices")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default();

        let reason = arguments
            .get("reason")
            .and_then(|r| r.as_str())
            .unwrap_or("no reason given")
            .to_string();

        self.recalled_indices.extend(&indices);

        // Build event window from recalled indices
        let event_window: Vec<_> = indices
            .iter()
            .filter_map(|&i| self.package.recent_events.get(i).cloned())
            .collect();

        let window_vectors: Vec<_> = event_window
            .iter()
            .filter_map(|e| e.embedding.clone())
            .collect();

        let now = now_ms();
        let example = TrainingExample {
            id: uuid::Uuid::new_v4().to_string(),
            activity_id: self.package.activity_id.clone(),
            stream_id: self.package.trigger_event.stream_id.clone(),
            domain: self
                .package
                .firing_goals
                .first()
                .and_then(|g| g.domain.clone()),
            goal_id: self
                .package
                .firing_goals
                .first()
                .map(|g| g.id.clone())
                .unwrap_or_default(),
            goal_version: self
                .package
                .firing_goals
                .first()
                .map(|g| g.version)
                .unwrap_or(1),
            goal_embedding: self
                .package
                .firing_goals
                .first()
                .map(|g| g.embedding.clone())
                .unwrap_or_default(),
            event_window,
            window_vectors,
            label: Label::Positive,
            label_source: LabelSource::LlmRecall,
            label_weight: 0.8,
            reason,
            created_at: now,
            used_in_batch: false,
            confidence: 0.8,
            consensus_count: 1,
        };

        if let Err(e) = producer.publish(names::LABELS_WRITE, &example).await {
            warn!(error = %e, "failed to publish recall label");
        }

        self.label_count += 1;
        info!(activity_id = %self.package.activity_id, indices = ?indices, "recall() recorded");
        serde_json::json!({"status": "recorded", "label": "positive", "events_recalled": indices.len()})
    }

    async fn handle_update_goal(
        &mut self,
        arguments: &Value,
        producer: &mut StreamProducer,
    ) -> Value {
        let goal_id = arguments
            .get("goal_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let description = arguments
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let blend_factor = arguments
            .get("blend_factor")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5) as f32;

        let req = GoalUpdateRequest {
            activity_id: self.package.activity_id.clone(),
            target_goal_id: Some(goal_id.to_string()),
            description: description.to_string(),
            blend_factor,
            parent_goal_id: None,
            aggregation: None,
        };

        if let Err(e) = producer.publish(names::REQUESTS_GOAL_UPDATE, &req).await {
            warn!(error = %e, "failed to publish goal update");
            return serde_json::json!({"error": "failed to submit goal update"});
        }

        info!(goal_id, "update_goal() submitted");
        serde_json::json!({"status": "submitted", "goal_id": goal_id})
    }

    async fn handle_add_goal(&mut self, arguments: &Value, producer: &mut StreamProducer) -> Value {
        let description = arguments
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let name = arguments
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or(description);

        let req = GoalUpdateRequest {
            activity_id: self.package.activity_id.clone(),
            target_goal_id: None,
            description: description.to_string(),
            blend_factor: 0.0,
            parent_goal_id: None,
            aggregation: None,
        };

        if let Err(e) = producer.publish(names::REQUESTS_GOAL_UPDATE, &req).await {
            warn!(error = %e, "failed to publish add goal");
            return serde_json::json!({"error": "failed to submit new goal"});
        }

        info!(name, "add_goal() submitted");
        serde_json::json!({"status": "submitted", "name": name})
    }

    async fn handle_threshold_hint(
        &mut self,
        arguments: &Value,
        producer: &mut StreamProducer,
    ) -> Value {
        // Route as a goal update request with the hint embedded
        let goal_id = arguments
            .get("goal_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let direction = arguments
            .get("direction")
            .and_then(|v| v.as_str())
            .unwrap_or("raise");
        let magnitude = arguments
            .get("magnitude")
            .and_then(|v| v.as_str())
            .unwrap_or("slight");

        let hint = expert_types::signals::ThresholdHint {
            activity_id: self.package.activity_id.clone(),
            goal_id: goal_id.to_string(),
            direction: match direction {
                "lower" => expert_types::signals::ThresholdDirection::Lower,
                _ => expert_types::signals::ThresholdDirection::Raise,
            },
            magnitude: match magnitude {
                "moderate" => expert_types::signals::ThresholdMagnitude::Moderate,
                "strong" => expert_types::signals::ThresholdMagnitude::Strong,
                _ => expert_types::signals::ThresholdMagnitude::Slight,
            },
        };

        // Publish hint as part of goal update stream (orchestrator processes it)
        if let Err(e) = producer.publish(names::REQUESTS_GOAL_UPDATE, &hint).await {
            warn!(error = %e, "failed to publish threshold hint");
            return serde_json::json!({"error": "failed to submit threshold hint"});
        }

        info!(
            goal_id,
            direction, magnitude, "set_threshold_hint() submitted"
        );
        serde_json::json!({"status": "submitted"})
    }
}

pub fn build_tools_json(tool_defs: &[ToolDefinition]) -> Vec<Value> {
    tool_defs
        .iter()
        .map(|td| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": td.name,
                    "description": td.description,
                    "parameters": td.parameters_schema,
                }
            })
        })
        .collect()
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tool(name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            description: format!("{name} tool"),
            parameters_schema: serde_json::json!({"type": "object"}),
            is_domain_tool: false,
        }
    }

    #[test]
    fn test_build_tools_json_shape() {
        let defs = vec![make_tool("suppress"), make_tool("recall")];
        let json = build_tools_json(&defs);
        assert_eq!(json.len(), 2);
        for tool in &json {
            assert_eq!(tool["type"], "function");
            assert!(tool["function"]["name"].is_string());
            assert!(tool["function"]["description"].is_string());
            assert!(tool["function"]["parameters"].is_object());
        }
    }

    #[test]
    fn test_build_tools_json_empty() {
        let json = build_tools_json(&[]);
        assert!(json.is_empty());
    }

    #[test]
    fn test_build_tools_json_preserves_names() {
        let defs = vec![
            make_tool("suppress"),
            make_tool("recall"),
            make_tool("update_goal"),
            make_tool("add_goal"),
            make_tool("set_threshold_hint"),
        ];
        let json = build_tools_json(&defs);
        let names: Vec<&str> = json
            .iter()
            .map(|t| t["function"]["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            vec![
                "suppress",
                "recall",
                "update_goal",
                "add_goal",
                "set_threshold_hint"
            ]
        );
    }
}
