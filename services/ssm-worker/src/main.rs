mod activity;
mod features;
mod ssm;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

use expert_config::Config;
use expert_redis::names;
use expert_redis::{StreamConsumer, StreamProducer};
use expert_types::event::Event;
use expert_types::goal::Goal;

use activity::ActivityInstance;

const WORKER_ID: &str = "worker-1";

struct WorkerState {
    activities: HashMap<String, ActivityInstance>,
    producer: StreamProducer,
    config: Config,
}

#[tokio::main]
async fn main() -> Result<()> {
    expert_config::init_tracing();
    let config = Config::from_env();
    info!(worker_id = WORKER_ID, "starting ssm-worker");

    let conn = expert_redis::connect(&config.redis_url).await?;
    let producer = StreamProducer::new(conn.clone(), config.stream_maxlen);

    let state = Arc::new(RwLock::new(WorkerState {
        activities: HashMap::new(),
        producer,
        config: config.clone(),
    }));

    // Spawn command consumer
    {
        let state = state.clone();
        let conn = conn.clone();
        tokio::spawn(async move {
            let mut consumer = match StreamConsumer::new(
                conn,
                names::commands_worker(WORKER_ID),
                format!("worker-{WORKER_ID}"),
                "cmd-0".to_string(),
                500,
            )
            .await
            {
                Ok(c) => c,
                Err(e) => {
                    error!(error = %e, "failed to create command consumer");
                    return;
                }
            };

            loop {
                match consumer.consume::<serde_json::Value>().await {
                    Ok(Some((id, cmd))) => {
                        let _ = consumer.ack(&id).await;
                        handle_command(&state, cmd).await;
                    }
                    Ok(None) => {}
                    Err(e) => {
                        warn!(error = %e, "command consumer error");
                        tokio::time::sleep(Duration::from_secs(1)).await;
                    }
                }
            }
        });
    }

    // Main event loop: poll assigned streams
    // For MVP, we dynamically discover streams from assigned activities
    let conn_for_events = conn.clone();
    let state_for_events = state.clone();

    loop {
        let stream_ids: Vec<String> = {
            let s = state_for_events.read().await;
            s.activities
                .values()
                .map(|a| a.stream_id.clone())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect()
        };

        if stream_ids.is_empty() {
            tokio::time::sleep(Duration::from_millis(200)).await;
            continue;
        }

        // Process each stream
        for stream_id in &stream_ids {
            let stream = names::events_embedded(stream_id);
            // Try to consume one event at a time with short block
            let result = consume_one_event(&conn_for_events, &stream, WORKER_ID).await;

            match result {
                Ok(Some((entry_id, event))) => {
                    process_event(&state_for_events, event, &entry_id).await;
                }
                Ok(None) => {}
                Err(e) => {
                    warn!(error = %e, stream = %stream, "event consumption error");
                }
            }
        }

        // Check debounce timers
        check_debounce_timers(&state_for_events).await;

        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

async fn consume_one_event(
    conn: &redis::aio::MultiplexedConnection,
    stream: &str,
    worker_id: &str,
) -> Result<Option<(String, Event)>> {
    let group = format!("worker-{worker_id}");
    let consumer_name = "evt-0";

    // Ensure group exists
    let mut conn = conn.clone();
    let _: std::result::Result<(), _> = redis::cmd("XGROUP")
        .arg("CREATE")
        .arg(stream)
        .arg(&group)
        .arg("0")
        .arg("MKSTREAM")
        .query_async(&mut conn)
        .await;

    let opts = redis::streams::StreamReadOptions::default()
        .group(&group, consumer_name)
        .count(1)
        .block(50);

    let result: redis::streams::StreamReadReply =
        redis::AsyncCommands::xread_options(&mut conn, &[stream], &[">"], &opts).await?;

    for stream_key in result.keys {
        for entry in stream_key.ids {
            let id = entry.id.clone();
            if let Some(redis::Value::BulkString(bytes)) = entry.map.get("data") {
                let json_str = String::from_utf8_lossy(bytes);
                if let Ok(event) = serde_json::from_str::<Event>(&json_str) {
                    // ACK
                    let _: () = redis::cmd("XACK")
                        .arg(stream)
                        .arg(&group)
                        .arg(&id)
                        .query_async(&mut conn)
                        .await?;
                    return Ok(Some((id, event)));
                }
            }
        }
    }
    Ok(None)
}

async fn process_event(state: &Arc<RwLock<WorkerState>>, event: Event, _entry_id: &str) {
    let mut ws = state.write().await;
    let now = now_ms();

    // Fan out to all activities on this stream
    let activity_ids: Vec<String> = ws
        .activities
        .iter()
        .filter(|(_, a)| a.stream_id == event.stream_id)
        .map(|(id, _)| id.clone())
        .collect();

    // Clone config to avoid borrow conflict with activities
    let config = ws.config.clone();
    for activity_id in activity_ids {
        if let Some(activity) = ws.activities.get_mut(&activity_id) {
            activity.process_event(&event, now, &config);
        }
    }

    // Check for fire signals
    let mut signals = Vec::new();
    for activity_id in ws.activities.keys().cloned().collect::<Vec<_>>() {
        if let Some(activity) = ws.activities.get_mut(&activity_id)
            && let Some(signal) = activity.take_fire_signal()
        {
            signals.push(signal);
        }
    }

    for signal in signals {
        if let Err(e) = ws.producer.publish(names::SIGNALS_FIRE, &signal).await {
            error!(error = %e, "failed to publish fire signal");
        } else {
            info!(
                activity_id = %signal.activity_id,
                goals = ?signal.firing_goal_ids,
                "published fire signal"
            );
        }
    }
}

async fn check_debounce_timers(state: &Arc<RwLock<WorkerState>>) {
    let mut ws = state.write().await;
    let now = now_ms();

    let mut signals = Vec::new();
    for activity in ws.activities.values_mut() {
        if let Some(signal) = activity.check_debounce(now) {
            signals.push(signal);
        }
    }

    for signal in signals {
        if let Err(e) = ws.producer.publish(names::SIGNALS_FIRE, &signal).await {
            error!(error = %e, "failed to publish fire signal from debounce");
        } else {
            info!(
                activity_id = %signal.activity_id,
                "published fire signal (debounce confirmed)"
            );
        }
    }
}

async fn handle_command(state: &Arc<RwLock<WorkerState>>, cmd: serde_json::Value) {
    let cmd_type = cmd.get("type").and_then(|t| t.as_str()).unwrap_or("");

    match cmd_type {
        "assign" => {
            let activity_id = cmd
                .get("activity_id")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let stream_id = cmd.get("stream_id").and_then(|v| v.as_str()).unwrap_or("");
            let goals: Vec<Goal> = cmd
                .get("goals")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();

            if activity_id.is_empty() || stream_id.is_empty() {
                warn!("invalid assign command: missing activity_id or stream_id");
                return;
            }

            let config = {
                let ws = state.read().await;
                ws.config.clone()
            };

            let instance = ActivityInstance::new(
                activity_id.to_string(),
                stream_id.to_string(),
                goals,
                &config,
            );

            let mut ws = state.write().await;
            ws.activities.insert(activity_id.to_string(), instance);
            info!(activity_id, stream_id, "activity assigned to worker");
        }
        "goal_update" => {
            let activity_id = cmd
                .get("activity_id")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let goals: Vec<Goal> = cmd
                .get("goals")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();

            let mut ws = state.write().await;
            if let Some(activity) = ws.activities.get_mut(activity_id) {
                activity.update_goals(goals);
                info!(activity_id, "goal matrix updated");
            }
        }
        other => {
            warn!(cmd_type = other, "unknown command type");
        }
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}
