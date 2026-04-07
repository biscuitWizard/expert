use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use tokio::io::{AsyncBufReadExt, BufReader};
use tracing::{info, warn};

use expert_config::Config;
use expert_redis::names;
use expert_redis::{StateStore, StreamConsumer, StreamProducer};
use expert_types::event::Event;

#[tokio::main]
async fn main() -> Result<()> {
    expert_config::init_tracing();
    let config = Config::from_env();

    let stream_id = std::env::var("STREAM_ID").unwrap_or_else(|_| "stdin-01".to_string());
    info!(stream_id, "starting stream-ingestion");

    let conn = expert_redis::connect(&config.redis_url).await?;
    let mut producer = StreamProducer::new(conn.clone(), config.stream_maxlen);
    let mut state = StateStore::new(conn.clone());

    // Spawn action consumer (logs received actions for MVP)
    let action_stream = names::actions(&stream_id);
    let action_conn = conn.clone();
    let action_group = format!("adapter-{stream_id}");
    tokio::spawn(async move {
        match StreamConsumer::new(
            action_conn,
            action_stream,
            action_group,
            "adapter-0".to_string(),
            1000,
        )
        .await
        {
            Ok(mut consumer) => loop {
                match consumer.consume::<serde_json::Value>().await {
                    Ok(Some((_id, action))) => {
                        info!(action = %action, "[MVP] received domain action (not executing)");
                        if let Err(e) = consumer.ack(&_id).await {
                            warn!(error = %e, "failed to ack action");
                        }
                    }
                    Ok(None) => {}
                    Err(e) => {
                        warn!(error = %e, "action consumer error");
                        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    }
                }
            },
            Err(e) => {
                warn!(error = %e, "failed to create action consumer");
            }
        }
    });

    // Read lines from stdin and publish as events
    let stdin = tokio::io::stdin();
    let reader = BufReader::new(stdin);
    let mut lines = reader.lines();

    info!("reading from stdin. Type lines to generate events.");

    while let Some(line) = lines.next_line().await? {
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let seq = state.incr(&names::seq_key(&stream_id)).await?;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let event = Event {
            id: uuid::Uuid::new_v4().to_string(),
            stream_id: stream_id.clone(),
            sequence: seq,
            timestamp: now,
            raw: line.clone(),
            embedding: None,
            metadata: HashMap::new(),
        };

        let stream = names::events_raw(&stream_id);
        let entry_id = producer.publish(&stream, &event).await?;
        info!(seq, entry_id, raw = %line, "published event");
    }

    info!("stdin closed, shutting down");
    Ok(())
}
