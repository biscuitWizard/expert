use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use tokio::sync::Mutex;
use tokio::time;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use tracing::{error, info, warn};

use expert_redis::{StateStore, StreamProducer};

use crate::events;

const GATEWAY_URL: &str = "wss://gateway.discord.gg/?v=10&encoding=json";

// Discord Gateway opcodes
const OP_DISPATCH: u64 = 0;
const OP_HEARTBEAT: u64 = 1;
const OP_IDENTIFY: u64 = 2;
const OP_RESUME: u64 = 6;
const OP_RECONNECT: u64 = 7;
const OP_INVALID_SESSION: u64 = 9;
const OP_HELLO: u64 = 10;
const OP_HEARTBEAT_ACK: u64 = 11;

pub async fn run_gateway(
    token: &str,
    self_user_id: &str,
    username: &str,
    stream_id: &str,
    guild_ids: &[String],
    producer: StreamProducer,
    state: StateStore,
) -> Result<()> {
    let mut session_id: Option<String> = None;
    let mut resume_url: Option<String> = None;
    let seq = Arc::new(AtomicU64::new(0));

    loop {
        let url = resume_url.as_deref().unwrap_or(GATEWAY_URL);
        info!(url, "connecting to discord gateway");

        let result = run_session(
            url,
            token,
            self_user_id,
            username,
            stream_id,
            guild_ids,
            &session_id,
            &seq,
            &producer,
            &state,
        )
        .await;

        match result {
            Ok(SessionEnd::Reconnect {
                new_session_id,
                new_resume_url,
            }) => {
                session_id = Some(new_session_id);
                resume_url = new_resume_url;
                info!("gateway requested reconnect, resuming");
                time::sleep(Duration::from_secs(1)).await;
            }
            Ok(SessionEnd::InvalidSession) => {
                session_id = None;
                resume_url = None;
                warn!("invalid session, starting fresh");
                time::sleep(Duration::from_secs(5)).await;
            }
            Err(e) => {
                error!(error = %e, "gateway session error, reconnecting");
                time::sleep(Duration::from_secs(5)).await;
            }
        }
    }
}

enum SessionEnd {
    Reconnect {
        new_session_id: String,
        new_resume_url: Option<String>,
    },
    InvalidSession,
}

#[allow(clippy::too_many_arguments)]
async fn run_session(
    url: &str,
    token: &str,
    self_user_id: &str,
    username: &str,
    stream_id: &str,
    guild_ids: &[String],
    session_id: &Option<String>,
    seq: &Arc<AtomicU64>,
    producer: &StreamProducer,
    state: &StateStore,
) -> Result<SessionEnd> {
    let (ws, _) = connect_async(url)
        .await
        .context("failed to connect to discord gateway")?;

    let (write, mut read) = ws.split();
    let write = Arc::new(Mutex::new(write));

    // Wait for HELLO
    let hello = read
        .next()
        .await
        .context("gateway closed before HELLO")?
        .context("websocket error on HELLO")?;

    let hello_data = parse_message(&hello)?;
    let op = hello_data["op"].as_u64().unwrap_or(0);
    if op != OP_HELLO {
        bail!("expected HELLO (op 10), got op {op}");
    }

    let heartbeat_interval = hello_data["d"]["heartbeat_interval"]
        .as_u64()
        .unwrap_or(41250);
    info!(heartbeat_interval, "received HELLO");

    // Send IDENTIFY or RESUME
    if let Some(sid) = session_id {
        let resume_payload = serde_json::json!({
            "op": OP_RESUME,
            "d": {
                "token": token,
                "session_id": sid,
                "seq": seq.load(Ordering::Relaxed),
            }
        });
        let mut w = write.lock().await;
        w.send(Message::Text(resume_payload.to_string().into()))
            .await
            .context("failed to send RESUME")?;
        info!("sent RESUME");
    } else {
        let identify_payload = serde_json::json!({
            "op": OP_IDENTIFY,
            "d": {
                "token": token,
                "properties": {
                    "os": "linux",
                    "browser": "expert",
                    "device": "expert",
                },
                "compress": false,
            }
        });
        let mut w = write.lock().await;
        w.send(Message::Text(identify_payload.to_string().into()))
            .await
            .context("failed to send IDENTIFY")?;
        info!("sent IDENTIFY");
    }

    // Spawn heartbeat task
    let hb_write = write.clone();
    let hb_seq = seq.clone();
    let heartbeat_handle = tokio::spawn(async move {
        let mut interval = time::interval(Duration::from_millis(heartbeat_interval));
        loop {
            interval.tick().await;
            let s = hb_seq.load(Ordering::Relaxed);
            let payload = serde_json::json!({
                "op": OP_HEARTBEAT,
                "d": if s == 0 { Value::Null } else { Value::Number(s.into()) },
            });
            let mut w = hb_write.lock().await;
            if w.send(Message::Text(payload.to_string().into()))
                .await
                .is_err()
            {
                break;
            }
        }
    });

    let mut current_session_id = session_id.clone().unwrap_or_default();
    let mut current_resume_url: Option<String> = None;
    let mut producer = producer.clone();
    let mut state = state.clone();

    // Event loop
    let result = loop {
        let msg = match read.next().await {
            Some(Ok(msg)) => msg,
            Some(Err(e)) => {
                break Err(anyhow::anyhow!("websocket error: {e}"));
            }
            None => {
                break Err(anyhow::anyhow!("gateway connection closed"));
            }
        };

        if msg.is_close() {
            break Err(anyhow::anyhow!("gateway sent close frame"));
        }

        let data = match parse_message(&msg) {
            Ok(d) => d,
            Err(_) => continue,
        };

        let op = data["op"].as_u64().unwrap_or(0);

        // Update sequence number
        if let Some(s) = data["s"].as_u64() {
            seq.store(s, Ordering::Relaxed);
        }

        match op {
            OP_DISPATCH => {
                let event_name = data["t"].as_str().unwrap_or("");

                if event_name == "READY" {
                    current_session_id = data["d"]["session_id"].as_str().unwrap_or("").to_string();
                    current_resume_url = data["d"]["resume_gateway_url"]
                        .as_str()
                        .map(|s| s.to_string());
                    info!(session_id = %current_session_id, "READY received");
                    continue;
                }

                if event_name == "RESUMED" {
                    info!("RESUMED successfully");
                    continue;
                }

                // Normalize and publish event
                if let Some(event) = events::normalize_dispatch(
                    event_name,
                    &data["d"],
                    self_user_id,
                    username,
                    stream_id,
                    guild_ids,
                    &mut state,
                )
                .await
                {
                    let stream = expert_redis::names::events_raw(stream_id);
                    match producer.publish(&stream, &event).await {
                        Ok(entry_id) => {
                            info!(
                                event_type = event_name,
                                seq = event.sequence,
                                entry_id,
                                "published discord event"
                            );
                        }
                        Err(e) => {
                            warn!(error = %e, "failed to publish discord event");
                        }
                    }
                }
            }
            OP_HEARTBEAT_ACK => {}
            OP_HEARTBEAT => {
                // Server requesting immediate heartbeat
                let s = seq.load(Ordering::Relaxed);
                let payload = serde_json::json!({
                    "op": OP_HEARTBEAT,
                    "d": if s == 0 { Value::Null } else { Value::Number(s.into()) },
                });
                let mut w = write.lock().await;
                let _ = w.send(Message::Text(payload.to_string().into())).await;
            }
            OP_RECONNECT => {
                info!("received RECONNECT opcode");
                break Ok(SessionEnd::Reconnect {
                    new_session_id: current_session_id.clone(),
                    new_resume_url: current_resume_url.clone(),
                });
            }
            OP_INVALID_SESSION => {
                let resumable = data["d"].as_bool().unwrap_or(false);
                if resumable {
                    break Ok(SessionEnd::Reconnect {
                        new_session_id: current_session_id.clone(),
                        new_resume_url: current_resume_url.clone(),
                    });
                } else {
                    break Ok(SessionEnd::InvalidSession);
                }
            }
            _ => {}
        }
    };

    heartbeat_handle.abort();
    result
}

fn parse_message(msg: &Message) -> Result<Value> {
    match msg {
        Message::Text(text) => {
            serde_json::from_str(text).context("failed to parse gateway message as JSON")
        }
        _ => bail!("unexpected non-text message"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_text_message() {
        let msg = Message::Text(r#"{"op":10,"d":{"heartbeat_interval":41250}}"#.into());
        let val = parse_message(&msg).unwrap();
        assert_eq!(val["op"], 10);
        assert_eq!(val["d"]["heartbeat_interval"], 41250);
    }

    #[test]
    fn parse_dispatch_event() {
        let msg =
            Message::Text(r#"{"op":0,"t":"MESSAGE_CREATE","s":42,"d":{"content":"hello"}}"#.into());
        let val = parse_message(&msg).unwrap();
        assert_eq!(val["op"], OP_DISPATCH);
        assert_eq!(val["t"], "MESSAGE_CREATE");
        assert_eq!(val["s"], 42);
        assert_eq!(val["d"]["content"], "hello");
    }

    #[test]
    fn parse_invalid_json() {
        let msg = Message::Text("not json".into());
        assert!(parse_message(&msg).is_err());
    }

    #[test]
    fn parse_binary_message_rejected() {
        let msg = Message::Binary(vec![1, 2, 3].into());
        assert!(parse_message(&msg).is_err());
    }

    #[test]
    fn parse_heartbeat_ack() {
        let msg = Message::Text(r#"{"op":11}"#.into());
        let val = parse_message(&msg).unwrap();
        assert_eq!(val["op"], OP_HEARTBEAT_ACK);
    }

    #[test]
    fn parse_reconnect_opcode() {
        let msg = Message::Text(r#"{"op":7,"d":null}"#.into());
        let val = parse_message(&msg).unwrap();
        assert_eq!(val["op"], OP_RECONNECT);
    }

    #[test]
    fn parse_invalid_session() {
        let msg = Message::Text(r#"{"op":9,"d":false}"#.into());
        let val = parse_message(&msg).unwrap();
        assert_eq!(val["op"], OP_INVALID_SESSION);
        assert_eq!(val["d"], false);
    }
}
