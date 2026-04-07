use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use serde::Deserialize;
use serde_json::Value;
use tracing::{info, warn};

use expert_redis::{ServiceLogger, StateStore, StreamConsumer, StreamProducer, names};
use expert_types::event::Event;

use crate::rest::DiscordRestClient;

const DEDUP_CAPACITY: usize = 1000;

#[derive(Debug, Deserialize)]
struct ActionPayload {
    tool_name: String,
    arguments: Value,
    invocation_id: Option<String>,
}

/// Run the action consumer loop: reads from `actions.{stream_id}`,
/// dispatches to Discord REST, and folds results back as events.
pub async fn run_action_consumer(
    conn: redis::aio::MultiplexedConnection,
    producer: &mut StreamProducer,
    state: &mut StateStore,
    rest: &DiscordRestClient,
    stream_id: &str,
    mut svc_log: ServiceLogger,
) -> Result<()> {
    let action_stream = names::actions(stream_id);
    let group = format!("adapter-{stream_id}");

    let mut consumer =
        StreamConsumer::new(conn, action_stream, group, "adapter-0".to_string(), 1000).await?;

    info!(stream_id, "action consumer started");

    let mut seen_ids: VecDeque<String> = VecDeque::with_capacity(DEDUP_CAPACITY);

    loop {
        match consumer.consume::<ActionPayload>().await {
            Ok(Some((id, payload))) => {
                if let Some(ref inv_id) = payload.invocation_id {
                    if seen_ids.contains(inv_id) {
                        info!(invocation_id = %inv_id, tool = %payload.tool_name, "skipping duplicate action");
                        let _ = consumer.ack(&id).await;
                        continue;
                    }
                    if seen_ids.len() >= DEDUP_CAPACITY {
                        seen_ids.pop_front();
                    }
                    seen_ids.push_back(inv_id.clone());
                }

                info!(tool = %payload.tool_name, "executing domain action");

                let result = execute_action(rest, &payload.tool_name, &payload.arguments).await;

                if let Err(e) = consumer.ack(&id).await {
                    warn!(error = %e, "failed to ack action");
                }

                let (raw, result_metadata) = match &result {
                    Ok(v) => (
                        format!("[action:{}] success", payload.tool_name),
                        serde_json::json!({ "status": "success", "result": v }),
                    ),
                    Err(e) => {
                        svc_log
                            .error(
                                format!("Action {} failed: {e}", payload.tool_name),
                                Some(serde_json::json!({
                                    "tool": payload.tool_name,
                                    "arguments": payload.arguments,
                                })),
                            )
                            .await;
                        (
                            format!("[action:{}] error: {}", payload.tool_name, e),
                            serde_json::json!({ "status": "error", "error": e.to_string() }),
                        )
                    }
                };

                let mut metadata = HashMap::new();
                metadata.insert(
                    "event_type".to_string(),
                    Value::String("action_result".to_string()),
                );
                metadata.insert(
                    "tool_name".to_string(),
                    Value::String(payload.tool_name.clone()),
                );
                metadata.insert("is_self".to_string(), Value::Bool(true));
                metadata.insert("action_result".to_string(), result_metadata);

                let seq = state.incr(&names::seq_key(stream_id)).await.unwrap_or(0);
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64;

                let event = Event {
                    id: uuid::Uuid::new_v4().to_string(),
                    stream_id: stream_id.to_string(),
                    sequence: seq,
                    timestamp: now,
                    raw,
                    embedding: None,
                    metadata,
                };

                let stream = names::events_raw(stream_id);
                if let Err(e) = producer.publish(&stream, &event).await {
                    warn!(error = %e, "failed to publish action result event");
                }
            }
            Ok(None) => {}
            Err(e) => {
                warn!(error = %e, "action consumer error");
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
        }
    }
}

async fn execute_action(
    rest: &DiscordRestClient,
    tool_name: &str,
    arguments: &Value,
) -> Result<Value> {
    match tool_name {
        "send_message" => {
            let channel_id = arg_str(arguments, "channel_id")?;
            let content = arg_str(arguments, "content")?;
            rest.send_message(channel_id, content).await
        }
        "reply_to_message" => {
            let channel_id = arg_str(arguments, "channel_id")?;
            let message_id = arg_str(arguments, "message_id")?;
            let content = arg_str(arguments, "content")?;
            rest.reply_to_message(channel_id, message_id, content).await
        }
        "send_dm" => {
            let user_id = arg_str(arguments, "user_id")?;
            let content = arg_str(arguments, "content")?;
            rest.send_dm(user_id, content).await
        }
        "react_to_message" => {
            let channel_id = arg_str(arguments, "channel_id")?;
            let message_id = arg_str(arguments, "message_id")?;
            let emoji = arg_str(arguments, "emoji")?;
            rest.react_to_message(channel_id, message_id, emoji).await?;
            Ok(serde_json::json!({"status": "reacted"}))
        }
        "join_guild" => {
            let invite_code = arg_str(arguments, "invite_code")?;
            rest.join_guild(invite_code).await
        }
        "leave_guild" => {
            let guild_id = arg_str(arguments, "guild_id")?;
            rest.leave_guild(guild_id).await?;
            Ok(serde_json::json!({"status": "left guild"}))
        }
        "send_friend_request" => {
            let username = arg_str(arguments, "username")?;
            rest.send_friend_request(username).await
        }
        "accept_friend_request" => {
            let user_id = arg_str(arguments, "user_id")?;
            rest.accept_friend_request(user_id).await?;
            Ok(serde_json::json!({"status": "friend request accepted"}))
        }
        "remove_friend" => {
            let user_id = arg_str(arguments, "user_id")?;
            rest.remove_friend(user_id).await?;
            Ok(serde_json::json!({"status": "friend removed"}))
        }
        "typing_indicator" => {
            let channel_id = arg_str(arguments, "channel_id")?;
            rest.trigger_typing(channel_id).await?;
            Ok(serde_json::json!({"status": "typing"}))
        }
        other => {
            warn!(tool = other, "unknown domain tool");
            Ok(serde_json::json!({"error": format!("unknown tool: {other}")}))
        }
    }
}

fn arg_str<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    args.get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("missing required argument: {key}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arg_str_present() {
        let args = serde_json::json!({"channel_id": "123", "content": "hello"});
        assert_eq!(arg_str(&args, "channel_id").unwrap(), "123");
        assert_eq!(arg_str(&args, "content").unwrap(), "hello");
    }

    #[test]
    fn arg_str_missing() {
        let args = serde_json::json!({"channel_id": "123"});
        assert!(arg_str(&args, "content").is_err());
    }

    #[test]
    fn arg_str_wrong_type() {
        let args = serde_json::json!({"channel_id": 123});
        assert!(arg_str(&args, "channel_id").is_err());
    }

    #[test]
    fn arg_str_null() {
        let args = serde_json::json!({"channel_id": null});
        assert!(arg_str(&args, "channel_id").is_err());
    }

    #[test]
    fn action_payload_deserializes() {
        let json = r#"{"tool_name":"send_message","arguments":{"channel_id":"123","content":"hi"},"invocation_id":"abc"}"#;
        let payload: ActionPayload = serde_json::from_str(json).unwrap();
        assert_eq!(payload.tool_name, "send_message");
        assert_eq!(payload.arguments["channel_id"], "123");
    }

    #[test]
    fn action_payload_without_invocation_id() {
        let json = r#"{"tool_name":"typing_indicator","arguments":{"channel_id":"456"}}"#;
        let payload: ActionPayload = serde_json::from_str(json).unwrap();
        assert_eq!(payload.tool_name, "typing_indicator");
        assert!(payload.invocation_id.is_none());
    }
}
