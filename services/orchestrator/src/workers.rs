use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tracing::{error, info, warn};

use expert_redis::StreamConsumer;
use expert_redis::names;
use expert_types::activity::ActivityLifecycle;
use expert_types::service_log::{LogLevel, ServiceLogEntry};
use expert_types::signals::{
    AssembleRequest, CheckpointAvailable, EncodeRequest, EncodeResult, FilterUpdateRequest,
    FireSignal, GoalUpdateRequest, InvocationComplete,
};

use crate::AppState;
use crate::ModelWarmupStatus;
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
    state
        .event_log
        .push(
            "fire_received",
            format!(
                "Fire signal for {} (goals: {})",
                &signal.activity_id[..signal.activity_id.len().min(8)],
                signal.firing_goal_ids.len()
            ),
            Some(serde_json::json!({
                "activity_id": signal.activity_id,
                "firing_goal_ids": signal.firing_goal_ids,
                "scores": signal.scores,
            })),
        )
        .await;

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
        bot_identity: activity.bot_identity.clone(),
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
        state
            .event_log
            .push(
                "context_dispatched",
                format!("Context assembly for {}", &assemble_req.activity_id[..8]),
                Some(serde_json::json!({
                    "activity_id": assemble_req.activity_id,
                    "stream_id": assemble_req.stream_id,
                })),
            )
            .await;
    }
}

async fn drain_stale_fires(state: &AppState, ttl_ms: u64) {
    let now = now_ms();
    let mut registry = state.registry.write().await;

    for (_, activity) in registry.activities.iter_mut() {
        if let Some(ref pf) = activity.pending_fire
            && now - pf.received_at > ttl_ms
        {
            info!(
                activity_id = %activity.state.activity_id,
                "dropping stale fire signal"
            );
            state
                .event_log
                .push(
                    "stale_fire_dropped",
                    format!(
                        "Stale fire dropped for {}",
                        &activity.state.activity_id[..8]
                    ),
                    Some(serde_json::json!({"activity_id": activity.state.activity_id})),
                )
                .await;
            activity.pending_fire = None;
            if activity.state.lifecycle_state == ActivityLifecycle::Fired {
                activity.state.lifecycle_state = ActivityLifecycle::Active;
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

    let is_update = req.target_goal_id.is_some();
    let description_for_log = req.description.clone();

    if let Some(target_id) = &req.target_goal_id {
        if let Some(goal) = activity.goals.iter_mut().find(|g| &g.id == target_id) {
            goal.description = req.description;
            goal.embedding = embedding.clone();
            goal.version += 1;
        }
    } else {
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

    let goal_matrix: Vec<f32> = activity
        .goals
        .iter()
        .flat_map(|g| g.embedding.clone())
        .collect();
    let goal_indices: Vec<String> = activity.goals.iter().map(|g| g.id.clone()).collect();
    activity.state.goal_matrix = goal_matrix;
    activity.state.goal_indices = goal_indices;
    activity.state.theta.resize(activity.goals.len(), 0.5);

    let worker_id = activity.worker_id.clone();
    let goals_snapshot = activity.goals.clone();
    drop(registry);

    let mut producer = state.producer.write().await;

    let cmd = serde_json::json!({
        "type": "goal_update",
        "activity_id": req.activity_id,
        "goals": goals_snapshot,
    });
    let _ = producer
        .publish(&names::commands_worker(&worker_id), &cmd)
        .await;

    for goal in &goals_snapshot {
        let _ = producer.publish(names::GOALS_WRITE, goal).await;
    }

    let kind = if is_update {
        "goal_updated"
    } else {
        "goal_added"
    };
    state
        .event_log
        .push(
            kind,
            format!(
                "{} on activity {}",
                if is_update {
                    "Goal updated"
                } else {
                    "Goal added"
                },
                &req.activity_id[..8]
            ),
            Some(serde_json::json!({
                "activity_id": req.activity_id,
                "target_goal_id": req.target_goal_id,
                "description": description_for_log,
            })),
        )
        .await;
    info!(activity_id = %req.activity_id, "goal update propagated");
}

/// Consume checkpoint notifications and dispatch reload commands to affected workers.
pub async fn spawn_checkpoint_consumer(state: Arc<AppState>) {
    let conn = match expert_redis::connect(&state.config.redis_url).await {
        Ok(c) => c,
        Err(e) => {
            error!(error = %e, "failed to connect for checkpoint consumer");
            return;
        }
    };

    tokio::spawn(async move {
        let mut consumer = match StreamConsumer::new(
            conn,
            names::CHECKPOINTS_AVAILABLE.to_string(),
            "orchestrator-ckpt".to_string(),
            "orch-ckpt-0".to_string(),
            500,
        )
        .await
        {
            Ok(c) => c,
            Err(e) => {
                error!(error = %e, "failed to create checkpoint consumer");
                return;
            }
        };

        loop {
            match consumer.consume::<CheckpointAvailable>().await {
                Ok(Some((id, notification))) => {
                    let _ = consumer.ack(&id).await;
                    info!(
                        checkpoint_id = %notification.checkpoint_id,
                        domain = ?notification.domain,
                        timescale = ?notification.timescale,
                        "received checkpoint notification"
                    );
                    handle_checkpoint(&state, notification).await;
                }
                Ok(None) => {}
                Err(e) => {
                    warn!(error = %e, "checkpoint consumer error");
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                }
            }
        }
    });
}

async fn handle_checkpoint(state: &AppState, notification: CheckpointAvailable) {
    let registry = state.registry.read().await;

    // Group affected activities by worker_id
    let mut worker_activities: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();

    for (activity_id, entry) in &registry.activities {
        let domain_match = match (&notification.domain, &entry.state.domain) {
            (Some(d), domain) => d == domain,
            (None, _) => true,
        };

        if domain_match {
            worker_activities
                .entry(entry.worker_id.clone())
                .or_default()
                .push(activity_id.clone());
        }
    }

    drop(registry);

    let mut producer = state.producer.write().await;
    for (worker_id, activity_ids) in &worker_activities {
        let cmd = serde_json::json!({
            "type": "checkpoint_reload",
            "checkpoint_id": notification.checkpoint_id,
            "path": notification.path,
            "activity_ids": activity_ids,
        });

        if let Err(e) = producer
            .publish(&names::commands_worker(worker_id), &cmd)
            .await
        {
            error!(error = %e, worker_id, "failed to dispatch checkpoint reload");
        } else {
            info!(
                worker_id,
                activities = activity_ids.len(),
                "dispatched checkpoint reload"
            );
            state
                .event_log
                .push(
                    "checkpoint_reload",
                    format!(
                        "Checkpoint reload to {} ({} activities)",
                        worker_id,
                        activity_ids.len()
                    ),
                    Some(serde_json::json!({
                        "checkpoint_id": notification.checkpoint_id,
                        "worker_id": worker_id,
                        "activity_count": activity_ids.len(),
                    })),
                )
                .await;
        }
    }
}

/// Spawn a consumer that dispatches threshold update commands to workers
/// after invocation cycles complete.
pub async fn spawn_threshold_feedback_task(state: Arc<AppState>) {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;

            let registry = state.registry.read().await;
            let mut updates: Vec<(String, String, f32, f32)> = Vec::new();

            for (activity_id, entry) in &registry.activities {
                if entry.state.invocation_count == 0 {
                    continue;
                }

                let suppress_rate =
                    entry.state.suppress_count as f32 / entry.state.invocation_count as f32;
                let recall_rate =
                    entry.state.recall_count as f32 / entry.state.invocation_count as f32;

                updates.push((
                    activity_id.clone(),
                    entry.worker_id.clone(),
                    suppress_rate,
                    recall_rate,
                ));
            }

            drop(registry);

            if updates.is_empty() {
                continue;
            }

            let mut producer = state.producer.write().await;
            for (activity_id, worker_id, suppress_rate, recall_rate) in &updates {
                let cmd = serde_json::json!({
                    "type": "threshold_update",
                    "activity_id": activity_id,
                    "suppress_rate": suppress_rate,
                    "recall_rate": recall_rate,
                });

                let _ = producer
                    .publish(&names::commands_worker(worker_id), &cmd)
                    .await;
            }
        }
    });
}

/// Consume filter update requests from llm-gateway (LLM calling update_event_filter()).
pub async fn spawn_filter_update_consumer(state: Arc<AppState>) {
    let conn = match expert_redis::connect(&state.config.redis_url).await {
        Ok(c) => c,
        Err(e) => {
            error!(error = %e, "failed to connect for filter update consumer");
            return;
        }
    };

    tokio::spawn(async move {
        let mut consumer = match StreamConsumer::new(
            conn,
            names::REQUESTS_FILTER_UPDATE.to_string(),
            "orchestrator-filters".to_string(),
            "orch-filter-0".to_string(),
            500,
        )
        .await
        {
            Ok(c) => c,
            Err(e) => {
                error!(error = %e, "failed to create filter update consumer");
                return;
            }
        };

        loop {
            match consumer.consume::<FilterUpdateRequest>().await {
                Ok(Some((id, req))) => {
                    let _ = consumer.ack(&id).await;
                    info!(
                        activity_id = %req.activity_id,
                        "received filter update request from LLM"
                    );
                    handle_filter_update(&state, req).await;
                }
                Ok(None) => {}
                Err(e) => {
                    warn!(error = %e, "filter update consumer error");
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                }
            }
        }
    });
}

async fn handle_filter_update(state: &AppState, req: FilterUpdateRequest) {
    let errors = req.event_filter.validate();
    if !errors.is_empty() {
        warn!(
            activity_id = %req.activity_id,
            errors = ?errors,
            "rejecting invalid filter update from LLM"
        );
        return;
    }

    let mut registry = state.registry.write().await;
    let activity = match registry.activities.get_mut(&req.activity_id) {
        Some(a) => a,
        None => {
            warn!(activity_id = %req.activity_id, "filter update for unknown activity");
            return;
        }
    };

    activity.event_filter = req.event_filter.clone();
    activity.state.event_filter = req.event_filter.clone();
    let worker_id = activity.worker_id.clone();
    drop(registry);

    let cmd = serde_json::json!({
        "type": "filter_update",
        "activity_id": req.activity_id,
        "event_filter": req.event_filter,
    });
    let mut producer = state.producer.write().await;
    let _ = producer
        .publish(&names::commands_worker(&worker_id), &cmd)
        .await;

    info!(activity_id = %req.activity_id, "filter update from LLM propagated");
}

/// Consume invocation-complete signals from llm-gateway and clear pending fires.
pub async fn spawn_invocation_complete_consumer(state: Arc<AppState>) {
    let conn = match expert_redis::connect(&state.config.redis_url).await {
        Ok(c) => c,
        Err(e) => {
            error!(error = %e, "failed to connect for invocation-complete consumer");
            return;
        }
    };

    tokio::spawn(async move {
        let mut consumer = match StreamConsumer::new(
            conn,
            names::SIGNALS_INVOCATION_COMPLETE.to_string(),
            "orchestrator-invoke".to_string(),
            "orch-invoke-0".to_string(),
            500,
        )
        .await
        {
            Ok(c) => c,
            Err(e) => {
                error!(error = %e, "failed to create invocation-complete consumer");
                return;
            }
        };

        loop {
            match consumer.consume::<InvocationComplete>().await {
                Ok(Some((id, signal))) => {
                    let _ = consumer.ack(&id).await;
                    handle_invocation_complete(&state, signal).await;
                }
                Ok(None) => {}
                Err(e) => {
                    warn!(error = %e, "invocation-complete consumer error");
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                }
            }
        }
    });
}

async fn handle_invocation_complete(state: &AppState, signal: InvocationComplete) {
    let mut registry = state.registry.write().await;

    let activity = match registry.activities.get_mut(&signal.activity_id) {
        Some(a) => a,
        None => {
            warn!(activity_id = %signal.activity_id, "invocation complete for unknown activity");
            return;
        }
    };

    activity.pending_fire = None;
    if activity.state.lifecycle_state == ActivityLifecycle::Fired {
        activity.state.lifecycle_state = ActivityLifecycle::Active;
    }

    drop(registry);

    let mut detail = serde_json::json!({
        "activity_id": signal.activity_id,
        "success": signal.success,
    });
    if let Some(ref et) = signal.event_type {
        detail["event_type"] = serde_json::json!(et);
    }
    if let Some(ref name) = signal.author_name {
        detail["author_name"] = serde_json::json!(name);
    }
    if let Some(ref resp) = signal.response_preview {
        detail["response"] = serde_json::json!(resp);
    }

    if signal.success {
        info!(activity_id = %signal.activity_id, "invocation completed successfully");
        state
            .event_log
            .push(
                "invocation_complete",
                format!("Invocation complete for {}", &signal.activity_id[..8]),
                Some(detail),
            )
            .await;
    } else {
        warn!(activity_id = %signal.activity_id, "invocation failed");
        state
            .event_log
            .push(
                "invocation_failed",
                format!("Invocation failed for {}", &signal.activity_id[..8]),
                Some(detail),
            )
            .await;
    }
}

/// Consume structured log entries from all services and surface them on the panel EventLog.
pub async fn spawn_service_log_consumer(state: Arc<AppState>) {
    let conn = match expert_redis::connect(&state.config.redis_url).await {
        Ok(c) => c,
        Err(e) => {
            error!(error = %e, "failed to connect for service-log consumer");
            return;
        }
    };

    tokio::spawn(async move {
        let mut consumer = match StreamConsumer::new(
            conn,
            names::LOGS_SERVICE.to_string(),
            "orchestrator-logs".to_string(),
            "orch-logs-0".to_string(),
            500,
        )
        .await
        {
            Ok(c) => c,
            Err(e) => {
                error!(error = %e, "failed to create service-log consumer");
                return;
            }
        };

        loop {
            match consumer.consume::<ServiceLogEntry>().await {
                Ok(Some((id, entry))) => {
                    let _ = consumer.ack(&id).await;

                    let kind = match entry.level {
                        LogLevel::Error => format!("svc_error:{}", entry.service),
                        LogLevel::Warn => format!("svc_warn:{}", entry.service),
                        LogLevel::Info => format!("svc_info:{}", entry.service),
                    };

                    state
                        .event_log
                        .push(kind, &entry.message, entry.detail)
                        .await;
                }
                Ok(None) => {}
                Err(e) => {
                    warn!(error = %e, "service-log consumer error");
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    });
}

/// Periodically send warmup/keepalive pings to Ollama for both models
/// when at least one activity is registered.
pub async fn spawn_warmup_task(state: Arc<AppState>) {
    let interval_secs = state.config.warmup_interval_secs;
    let keep_alive = state.config.warmup_keep_alive.clone();
    let ollama_url = state.config.ollama_url.clone();
    let ollama_embeddings_url = state.config.ollama_embeddings_url.clone();
    let llm_model = state.config.llm_model.clone();
    let embeddings_model = state.config.embeddings_model.clone();

    info!(
        interval_secs,
        keep_alive = %keep_alive,
        llm_model = %llm_model,
        embeddings_model = %embeddings_model,
        "starting Ollama warmup task"
    );

    tokio::spawn(async move {
        let client = reqwest::Client::new();
        let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));

        loop {
            interval.tick().await;

            let has_activities = {
                let registry = state.registry.read().await;
                !registry.list().is_empty()
            };

            if !has_activities {
                let mut status = state.warmup_status.write().await;
                status.insert(llm_model.clone(), ModelWarmupStatus::Cold);
                status.insert(embeddings_model.clone(), ModelWarmupStatus::Cold);
                continue;
            }

            // Warm both models concurrently
            let llm_fut = warmup_model(&client, &ollama_url, &llm_model, &keep_alive);
            let emb_fut = warmup_embedding_model(
                &client,
                &ollama_embeddings_url,
                &embeddings_model,
                &keep_alive,
            );

            {
                let mut status = state.warmup_status.write().await;
                if *status.get(&llm_model).unwrap_or(&ModelWarmupStatus::Cold)
                    != ModelWarmupStatus::Warm
                {
                    status.insert(llm_model.clone(), ModelWarmupStatus::Warming);
                }
                if *status
                    .get(&embeddings_model)
                    .unwrap_or(&ModelWarmupStatus::Cold)
                    != ModelWarmupStatus::Warm
                {
                    status.insert(embeddings_model.clone(), ModelWarmupStatus::Warming);
                }
            }

            let (llm_result, emb_result) = tokio::join!(llm_fut, emb_fut);

            let mut status = state.warmup_status.write().await;

            match llm_result {
                Ok(()) => {
                    status.insert(llm_model.clone(), ModelWarmupStatus::Warm);
                }
                Err(e) => {
                    warn!(model = %llm_model, error = %e, "LLM warmup failed");
                    status.insert(llm_model.clone(), ModelWarmupStatus::Error(e.to_string()));
                }
            }

            match emb_result {
                Ok(()) => {
                    status.insert(embeddings_model.clone(), ModelWarmupStatus::Warm);
                }
                Err(e) => {
                    warn!(model = %embeddings_model, error = %e, "embeddings warmup failed");
                    status.insert(
                        embeddings_model.clone(),
                        ModelWarmupStatus::Error(e.to_string()),
                    );
                }
            }
        }
    });
}

async fn warmup_model(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    keep_alive: &str,
) -> Result<(), reqwest::Error> {
    let url = format!("{}/api/generate", base_url.trim_end_matches('/'));
    let body = serde_json::json!({
        "model": model,
        "prompt": "",
        "keep_alive": keep_alive,
    });

    client
        .post(&url)
        .json(&body)
        .timeout(Duration::from_secs(120))
        .send()
        .await?
        .error_for_status()?;

    Ok(())
}

/// Embedding-only models don't support /api/generate; use /api/embed instead.
async fn warmup_embedding_model(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    keep_alive: &str,
) -> Result<(), reqwest::Error> {
    let url = format!("{}/api/embed", base_url.trim_end_matches('/'));
    let body = serde_json::json!({
        "model": model,
        "input": "",
        "keep_alive": keep_alive,
    });

    client
        .post(&url)
        .json(&body)
        .timeout(Duration::from_secs(120))
        .send()
        .await?
        .error_for_status()?;

    Ok(())
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
