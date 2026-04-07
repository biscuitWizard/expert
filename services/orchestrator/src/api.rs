use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tracing::{error, info};

use expert_redis::names;
use expert_types::event_filter::EventFilter;
use expert_types::goal::{Goal, GoalAggregation};
use expert_types::signals::{EncodeRequest, EncodeResult, ToolDefinition};

use crate::AppState;

const MVP_WORKER_ID: &str = "worker-1";

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/activities", post(create_activity))
        .route("/activities", get(list_activities))
        .route("/activities/{id}", get(get_activity))
        .route("/activities/{id}", delete(delete_activity))
        .route("/activities/{id}/filter", axum::routing::put(update_filter))
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
}

#[derive(Deserialize)]
struct CreateActivityRequest {
    stream_id: String,
    domain: Option<String>,
    goals: Vec<GoalInput>,
    #[serde(default)]
    tool_definitions: Vec<ToolDefinition>,
    #[serde(default)]
    event_filter: EventFilter,
}

#[derive(Deserialize)]
struct GoalInput {
    name: String,
    description: String,
}

#[derive(Serialize)]
struct ActivityResponse {
    activity_id: String,
    stream_id: String,
    domain: String,
    lifecycle_state: String,
    goal_count: usize,
    event_count: u64,
    invocation_count: u64,
    created_at: u64,
}

async fn create_activity(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateActivityRequest>,
) -> impl IntoResponse {
    let activity_id = uuid::Uuid::new_v4().to_string();
    let domain = req.domain.unwrap_or_else(|| "default".to_string());

    // Validate event filter
    let filter_errors = req.event_filter.validate();
    if !filter_errors.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "invalid event_filter", "details": filter_errors})),
        )
            .into_response();
    }

    info!(activity_id, stream_id = %req.stream_id, goals = req.goals.len(), "creating activity");

    // Encode all goal descriptions
    let mut goals = Vec::new();
    for goal_input in &req.goals {
        let request_id = uuid::Uuid::new_v4().to_string();
        let encode_req = EncodeRequest {
            request_id: request_id.clone(),
            text: goal_input.description.clone(),
        };

        // Publish encode request
        {
            let mut producer = state.producer.write().await;
            if let Err(e) = producer.publish(names::REQUESTS_ENCODE, &encode_req).await {
                error!(error = %e, "failed to publish encode request");
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": "encoding failed"})),
                )
                    .into_response();
            }
        }

        // Poll for result (simplified: poll state store with timeout)
        let embedding = match poll_encode_result(&state, &request_id).await {
            Some(emb) => emb,
            None => {
                error!("timeout waiting for goal encoding");
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": "encoding timeout"})),
                )
                    .into_response();
            }
        };

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        goals.push(Goal {
            id: uuid::Uuid::new_v4().to_string(),
            name: goal_input.name.clone(),
            description: goal_input.description.clone(),
            embedding,
            parent_id: None,
            children: Vec::new(),
            aggregation: GoalAggregation::Max,
            weights: None,
            domain: Some(domain.clone()),
            created_at: now,
            version: 1,
            active: true,
        });
    }

    // Build feedback tool definitions (always present)
    let mut all_tools = feedback_tool_definitions();
    all_tools.extend(req.tool_definitions);

    // Register activity
    let response = {
        let mut registry = state.registry.write().await;
        let managed = registry.create_activity(
            activity_id.clone(),
            req.stream_id.clone(),
            domain,
            goals.clone(),
            all_tools,
            req.event_filter.clone(),
            MVP_WORKER_ID.to_string(),
        );

        ActivityResponse {
            activity_id: managed.state.activity_id.clone(),
            stream_id: managed.state.stream_id.clone(),
            domain: managed.state.domain.clone(),
            lifecycle_state: format!("{:?}", managed.state.lifecycle_state),
            goal_count: managed.goals.len(),
            event_count: 0,
            invocation_count: 0,
            created_at: managed.state.created_at,
        }
    };

    // Persist activity state to Redis
    {
        let registry = state.registry.read().await;
        if let Some(managed) = registry.get(&activity_id) {
            let mut store = state.state_store.write().await;
            let _ = store
                .set_json(&names::state_key(&activity_id), &managed.state)
                .await;
            let _ = store
                .set_str(&names::assignment_key(&activity_id), MVP_WORKER_ID)
                .await;
        }
    }

    // Send assignment command to worker
    {
        let cmd = serde_json::json!({
            "type": "assign",
            "activity_id": activity_id,
            "stream_id": req.stream_id,
            "goals": goals,
            "event_filter": req.event_filter,
        });
        let mut producer = state.producer.write().await;
        let _ = producer
            .publish(&names::commands_worker(MVP_WORKER_ID), &cmd)
            .await;
    }

    // Persist goals to RAG
    for goal in &goals {
        let mut producer = state.producer.write().await;
        let _ = producer.publish(names::GOALS_WRITE, goal).await;
    }

    info!(activity_id = %response.activity_id, "activity created");
    state
        .event_log
        .push(
            "activity_created",
            format!(
                "Activity {} on stream {}",
                &response.activity_id[..8],
                response.stream_id
            ),
            Some(serde_json::json!({
                "activity_id": response.activity_id,
                "stream_id": response.stream_id,
                "domain": response.domain,
                "goal_count": response.goal_count,
            })),
        )
        .await;
    (
        StatusCode::CREATED,
        Json(serde_json::to_value(response).unwrap()),
    )
        .into_response()
}

async fn list_activities(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let registry = state.registry.read().await;
    let activities: Vec<ActivityResponse> = registry
        .list()
        .iter()
        .map(|m| ActivityResponse {
            activity_id: m.state.activity_id.clone(),
            stream_id: m.state.stream_id.clone(),
            domain: m.state.domain.clone(),
            lifecycle_state: format!("{:?}", m.state.lifecycle_state),
            goal_count: m.goals.len(),
            event_count: m.state.event_count,
            invocation_count: m.state.invocation_count,
            created_at: m.state.created_at,
        })
        .collect();
    Json(activities)
}

async fn get_activity(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let registry = state.registry.read().await;
    match registry.get(&id) {
        Some(m) => {
            let resp = serde_json::json!({
                "activity_id": m.state.activity_id,
                "stream_id": m.state.stream_id,
                "domain": m.state.domain,
                "lifecycle_state": format!("{:?}", m.state.lifecycle_state),
                "goals": m.goals.iter().map(|g| serde_json::json!({
                    "id": g.id,
                    "name": g.name,
                    "description": g.description,
                    "version": g.version,
                })).collect::<Vec<_>>(),
                "event_filter": m.event_filter,
                "event_count": m.state.event_count,
                "invocation_count": m.state.invocation_count,
                "suppress_count": m.state.suppress_count,
                "recall_count": m.state.recall_count,
                "theta": m.state.theta,
                "created_at": m.state.created_at,
                "last_active": m.state.last_active,
            });
            (StatusCode::OK, Json(resp)).into_response()
        }
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn delete_activity(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let mut registry = state.registry.write().await;
    match registry.remove(&id) {
        Some(_) => {
            // Clean up Redis state
            let mut store = state.state_store.write().await;
            let _ = store.del(&names::state_key(&id)).await;
            let _ = store.del(&names::assignment_key(&id)).await;
            let _ = store.del(&names::fire_queue_key(&id)).await;
            info!(activity_id = %id, "activity deleted");
            state
                .event_log
                .push(
                    "activity_deleted",
                    format!("Activity {} deleted", &id[..id.len().min(8)]),
                    Some(serde_json::json!({"activity_id": id})),
                )
                .await;
            StatusCode::NO_CONTENT
        }
        None => StatusCode::NOT_FOUND,
    }
}

async fn update_filter(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(new_filter): Json<EventFilter>,
) -> impl IntoResponse {
    let filter_errors = new_filter.validate();
    if !filter_errors.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "invalid event_filter", "details": filter_errors})),
        )
            .into_response();
    }

    let mut registry = state.registry.write().await;
    let activity = match registry.activities.get_mut(&id) {
        Some(a) => a,
        None => return StatusCode::NOT_FOUND.into_response(),
    };

    activity.event_filter = new_filter.clone();
    activity.state.event_filter = new_filter.clone();
    let worker_id = activity.worker_id.clone();
    drop(registry);

    // Notify SSM worker
    let cmd = serde_json::json!({
        "type": "filter_update",
        "activity_id": id,
        "event_filter": new_filter,
    });
    let mut producer = state.producer.write().await;
    let _ = producer
        .publish(&names::commands_worker(&worker_id), &cmd)
        .await;

    info!(activity_id = %id, "event filter updated");
    (
        StatusCode::OK,
        Json(serde_json::json!({"status": "updated", "event_filter": new_filter})),
    )
        .into_response()
}

/// Poll results.encode stream for a specific request_id.
/// Simplified for MVP: reads from the stream with a timeout loop.
async fn poll_encode_result(state: &AppState, request_id: &str) -> Option<Vec<f32>> {
    let conn = {
        let store = state.state_store.read().await;
        // Get a connection from the store's internal clone
        drop(store);
        expert_redis::connect(&state.config.redis_url).await.ok()?
    };

    let mut consumer = expert_redis::StreamConsumer::new(
        conn,
        names::RESULTS_ENCODE.to_string(),
        format!("orch-encode-{request_id}"),
        "orch-0".to_string(),
        500,
    )
    .await
    .ok()?;

    for _ in 0..20 {
        // 20 * 500ms = 10s max
        match consumer.consume::<EncodeResult>().await {
            Ok(Some((id, result))) => {
                let _ = consumer.ack(&id).await;
                if result.request_id == request_id {
                    return Some(result.embedding);
                }
            }
            Ok(None) => {}
            Err(_) => {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
        }
    }
    None
}

fn feedback_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "suppress".to_string(),
            description: "Signal that this invocation was unnecessary (false positive). Provide a reason.".to_string(),
            parameters_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Why this firing was a false positive"}
                },
                "required": ["reason"]
            }),
            is_domain_tool: false,
        },
        ToolDefinition {
            name: "recall".to_string(),
            description: "Signal that earlier events in the recent stream should have triggered firing. Provide event indices and a reason.".to_string(),
            parameters_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "event_indices": {"type": "array", "items": {"type": "integer"}, "description": "Indices into the recent stream events that should have triggered"},
                    "reason": {"type": "string", "description": "Why these events were relevant"}
                },
                "required": ["event_indices", "reason"]
            }),
            is_domain_tool: false,
        },
        ToolDefinition {
            name: "update_goal".to_string(),
            description: "Update an existing goal's description and embedding.".to_string(),
            parameters_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "goal_id": {"type": "string"},
                    "description": {"type": "string", "description": "New goal description"},
                    "blend_factor": {"type": "number", "description": "0.0 = hard replace, 1.0 = no change"}
                },
                "required": ["goal_id", "description", "blend_factor"]
            }),
            is_domain_tool: false,
        },
        ToolDefinition {
            name: "add_goal".to_string(),
            description: "Add a new goal to the activity.".to_string(),
            parameters_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "Goal description"},
                    "name": {"type": "string", "description": "Short goal name"}
                },
                "required": ["description", "name"]
            }),
            is_domain_tool: false,
        },
        ToolDefinition {
            name: "set_threshold_hint".to_string(),
            description: "Suggest adjusting the firing threshold for a goal.".to_string(),
            parameters_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "goal_id": {"type": "string"},
                    "direction": {"type": "string", "enum": ["raise", "lower"]},
                    "magnitude": {"type": "string", "enum": ["slight", "moderate", "strong"]}
                },
                "required": ["goal_id", "direction", "magnitude"]
            }),
            is_domain_tool: false,
        },
        ToolDefinition {
            name: "update_event_filter".to_string(),
            description: "Update the event filter for this activity to control which events from the stream are delivered. Use this to ignore irrelevant channels, senders, or event types. Pass null or omit to receive all events.".to_string(),
            parameters_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "filter": {
                        "description": "Event filter object. Use {\"field\": \"...\", \"op\": \"eq\"|\"ne\"|\"in\"|\"not_in\"|\"contains\"|\"matches\"|\"exists\"|\"not_exists\", \"value\": ...} for field predicates. Combine with {\"and\": [...]} or {\"or\": [...]}. Pass null to clear the filter and receive all events.",
                        "type": ["object", "null"]
                    }
                },
                "required": ["filter"]
            }),
            is_domain_tool: false,
        },
    ]
}
