use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use tracing::{error, info, warn};

use expert_redis::StreamConsumer;
use expert_redis::names;
use expert_types::activity::ActivityLifecycle;
use expert_types::signals::{
    AssembleRequest, EncodeRequest, EncodeResult, FireSignal, GoalUpdateRequest,
};

use crate::AppState;
use crate::registry::PendingFire;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// Consume fire signals from ssm-workers, manage fire queue, dispatch context assembly.
pub async fn spawn_fire_consumer(state: Arc<AppState>) {
    let conn = match expert_redis::connect(&state.config.redis_url).await {
        Ok(c) => c,
        Err(e) => {
            error!(error = %e, "failed to connect for fire consumer");
            return;
        }
    };

    let fire_queue_ttl = state.config.fire_queue_ttl_ms;

    tokio::spawn(async move {
        let mut consumer = match StreamConsumer::new(
            conn,
            names::SIGNALS_FIRE.to_string(),
            "orchestrator".to_string(),
            "orch-fire-0".to_string(),
            500,
        )
        .await
        {
            Ok(c) => c,
            Err(e) => {
                error!(error = %e, "failed to create fire consumer");
                return;
            }
        };

        loop {
            match consumer.consume::<FireSignal>().await {
                Ok(Some((id, signal))) => {
                    let _ = consumer.ack(&id).await;
                    info!(
                        activity_id = %signal.activity_id,
                        goals = ?signal.firing_goal_ids,
                        "received fire signal"
                    );
                    handle_fire_signal(&state, signal, fire_queue_ttl).await;
                }
                Ok(None) => {
                    // Check for stale fires on timeout
                    drain_stale_fires(&state, fire_queue_ttl).await;
                }
                Err(e) => {
                    warn!(error = %e, "fire consumer error");
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                }
            }
        }
    });
}

async fn handle_fire_signal(state: &AppState, signal: FireSignal, _ttl_ms: u64) {
    let now = now_ms();
    let mut registry = state.registry.write().await;

    let activity = match registry.activities.get_mut(&signal.activity_id) {
        Some(a) => a,
        None => {
            warn!(activity_id = %signal.activity_id, "fire signal for unknown activity");
            return;
        }
    };

    // Only accept fires when activity is in ACTIVE or COLD_START state
    match activity.state.lifecycle_state {
        ActivityLifecycle::Active | ActivityLifecycle::ColdStart => {}
        other => {
            info!(activity_id = %signal.activity_id, state = ?other, "ignoring fire signal in non-active state");
            return;
        }
    }

    // Depth-1 queue: replace any existing pending fire
    activity.pending_fire = Some(PendingFire {
        signal: signal.clone(),
        received_at: now,
    });

    // Immediately dispatch (depth-1 means we dispatch right away)
    activity.state.lifecycle_state = ActivityLifecycle::Fired;
    activity.state.invocation_count += 1;

    let assemble_req = AssembleRequest {
        activity_id: signal.activity_id.clone(),
        stream_id: signal.stream_id.clone(),
        fire_signal: signal,
        goal_tree: activity.goals.clone(),
        tool_definitions: activity.tool_definitions.clone(),
    };

    drop(registry);

    let mut producer = state.producer.write().await;
    if let Err(e) = producer
        .publish(names::REQUESTS_CONTEXT, &assemble_req)
        .await
    {
        error!(error = %e, "failed to publish assemble request");
    } else {
        info!(
            activity_id = %assemble_req.activity_id,
            "dispatched context assembly request"
        );
    }
}

async fn drain_stale_fires(state: &AppState, ttl_ms: u64) {
    let now = now_ms();
    let mut registry = state.registry.write().await;

    for (_, activity) in registry.activities.iter_mut() {
        if let Some(ref pf) = activity.pending_fire {
            if now - pf.received_at > ttl_ms {
                info!(
                    activity_id = %activity.state.activity_id,
                    "dropping stale fire signal"
                );
                activity.pending_fire = None;
                if activity.state.lifecycle_state == ActivityLifecycle::Fired {
                    activity.state.lifecycle_state = ActivityLifecycle::Active;
                }
            }
        }
    }
}

/// Consume goal update requests from llm-gateway, coordinate encoding + propagation.
pub async fn spawn_goal_update_consumer(state: Arc<AppState>) {
    let conn = match expert_redis::connect(&state.config.redis_url).await {
        Ok(c) => c,
        Err(e) => {
            error!(error = %e, "failed to connect for goal update consumer");
            return;
        }
    };

    tokio::spawn(async move {
        let mut consumer = match StreamConsumer::new(
            conn,
            names::REQUESTS_GOAL_UPDATE.to_string(),
            "orchestrator-goals".to_string(),
            "orch-goal-0".to_string(),
            500,
        )
        .await
        {
            Ok(c) => c,
            Err(e) => {
                error!(error = %e, "failed to create goal update consumer");
                return;
            }
        };

        loop {
            match consumer.consume::<GoalUpdateRequest>().await {
                Ok(Some((id, req))) => {
                    let _ = consumer.ack(&id).await;
                    info!(
                        activity_id = %req.activity_id,
                        target = ?req.target_goal_id,
                        "received goal update request"
                    );
                    handle_goal_update(&state, req).await;
                }
                Ok(None) => {}
                Err(e) => {
                    warn!(error = %e, "goal update consumer error");
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                }
            }
        }
    });
}

async fn handle_goal_update(state: &AppState, req: GoalUpdateRequest) {
    // Encode the new goal description
    let request_id = uuid::Uuid::new_v4().to_string();
    let encode_req = EncodeRequest {
        request_id: request_id.clone(),
        text: req.description.clone(),
    };

    {
        let mut producer = state.producer.write().await;
        if let Err(e) = producer.publish(names::REQUESTS_ENCODE, &encode_req).await {
            error!(error = %e, "failed to publish encode request for goal update");
            return;
        }
    }

    // Poll for encoding result
    let embedding = match poll_encode_result_bg(state, &request_id).await {
        Some(emb) => emb,
        None => {
            error!("timeout waiting for goal encoding in update");
            return;
        }
    };

    let mut registry = state.registry.write().await;
    let activity = match registry.activities.get_mut(&req.activity_id) {
        Some(a) => a,
        None => {
            warn!(activity_id = %req.activity_id, "goal update for unknown activity");
            return;
        }
    };

    if let Some(target_id) = &req.target_goal_id {
        // Update existing goal
        if let Some(goal) = activity.goals.iter_mut().find(|g| &g.id == target_id) {
            goal.description = req.description;
            goal.embedding = embedding.clone();
            goal.version += 1;
        }
    } else {
        // Add new goal
        let now = now_ms();
        let new_goal = expert_types::goal::Goal {
            id: uuid::Uuid::new_v4().to_string(),
            name: req.description.clone(),
            description: req.description,
            embedding: embedding.clone(),
            parent_id: req.parent_goal_id,
            children: Vec::new(),
            aggregation: req
                .aggregation
                .unwrap_or(expert_types::goal::GoalAggregation::Max),
            weights: None,
            domain: Some(activity.state.domain.clone()),
            created_at: now,
            version: 1,
            active: true,
        };
        activity.goals.push(new_goal);
    }

    // Rebuild goal matrix
    let goal_matrix: Vec<f32> = activity
        .goals
        .iter()
        .flat_map(|g| g.embedding.clone())
        .collect();
    let goal_indices: Vec<String> = activity.goals.iter().map(|g| g.id.clone()).collect();
    activity.state.goal_matrix = goal_matrix;
    activity.state.goal_indices = goal_indices;
    activity.state.theta.resize(activity.goals.len(), 0.5);

    // Send goal matrix update to worker
    let worker_id = activity.worker_id.clone();
    let goals_snapshot = activity.goals.clone();
    drop(registry);

    let mut producer = state.producer.write().await;

    // Notify worker
    let cmd = serde_json::json!({
        "type": "goal_update",
        "activity_id": req.activity_id,
        "goals": goals_snapshot,
    });
    let _ = producer
        .publish(&names::commands_worker(&worker_id), &cmd)
        .await;

    // Persist to RAG
    for goal in &goals_snapshot {
        let _ = producer.publish(names::GOALS_WRITE, goal).await;
    }

    info!(activity_id = %req.activity_id, "goal update propagated");
}

async fn poll_encode_result_bg(state: &AppState, request_id: &str) -> Option<Vec<f32>> {
    let conn = expert_redis::connect(&state.config.redis_url).await.ok()?;

    let mut consumer = StreamConsumer::new(
        conn,
        names::RESULTS_ENCODE.to_string(),
        format!("orch-encode-bg-{request_id}"),
        "orch-bg-0".to_string(),
        500,
    )
    .await
    .ok()?;

    for _ in 0..20 {
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
