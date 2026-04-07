use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;
use tracing::warn;

use expert_redis::StateStore;
use expert_redis::names;
use expert_types::event::Event;

/// Normalize a Discord Gateway DISPATCH event into an Expert Event.
/// Returns None if the event should be filtered out.
pub async fn normalize_dispatch(
    event_name: &str,
    data: &Value,
    self_user_id: &str,
    _username: &str,
    stream_id: &str,
    guild_filter: &[String],
    state: &mut StateStore,
) -> Option<Event> {
    match event_name {
        "MESSAGE_CREATE" | "MESSAGE_UPDATE" => {
            normalize_message(
                event_name,
                data,
                self_user_id,
                stream_id,
                guild_filter,
                state,
            )
            .await
        }
        "GUILD_MEMBER_ADD" => {
            normalize_guild_member_add(data, stream_id, guild_filter, state).await
        }
        "GUILD_MEMBER_REMOVE" => {
            normalize_guild_member_remove(data, stream_id, guild_filter, state).await
        }
        "RELATIONSHIP_ADD" => normalize_relationship_add(data, stream_id, state).await,
        "RELATIONSHIP_REMOVE" => normalize_relationship_remove(data, stream_id, state).await,
        "CHANNEL_CREATE" => normalize_channel_create(data, stream_id, state).await,
        "TYPING_START" => None, // Too noisy to ingest
        _ => None,
    }
}

async fn normalize_message(
    event_name: &str,
    data: &Value,
    self_user_id: &str,
    stream_id: &str,
    guild_filter: &[String],
    state: &mut StateStore,
) -> Option<Event> {
    let guild_id = data["guild_id"].as_str().unwrap_or("").to_string();
    if !guild_filter.is_empty() && !guild_id.is_empty() && !guild_filter.contains(&guild_id) {
        return None;
    }

    let author = &data["author"];
    let author_id = author["id"].as_str().unwrap_or("unknown");
    let author_name = author
        .get("global_name")
        .and_then(|v| v.as_str())
        .or_else(|| author["username"].as_str())
        .unwrap_or("unknown");

    let content = data["content"].as_str().unwrap_or("");
    let channel_id = data["channel_id"].as_str().unwrap_or("");
    let message_id = data["id"].as_str().unwrap_or("");
    let is_self = author_id == self_user_id;

    // Build human-readable raw text
    let is_dm = guild_id.is_empty();
    let channel_name = if is_dm {
        "DM".to_string()
    } else {
        format!("#{}", channel_id)
    };

    let action = if event_name == "MESSAGE_UPDATE" {
        "edited"
    } else {
        ""
    };

    let raw = if is_dm {
        if is_self {
            format!("[DM to user] {content}")
        } else {
            format!("[DM from {author_name}] {content}")
        }
    } else if action.is_empty() {
        format!("[{channel_name}] {author_name}: {content}")
    } else {
        format!("[{channel_name}] {author_name} ({action}): {content}")
    };

    // Skip empty messages (e.g. embed-only)
    if content.is_empty()
        && data
            .get("embeds")
            .and_then(|e| e.as_array())
            .is_none_or(|a| a.is_empty())
    {
        return None;
    }

    let mut metadata = HashMap::new();
    metadata.insert(
        "event_type".to_string(),
        Value::String(if is_dm {
            "dm".to_string()
        } else {
            "message".to_string()
        }),
    );
    metadata.insert(
        "author_id".to_string(),
        Value::String(author_id.to_string()),
    );
    metadata.insert(
        "author_name".to_string(),
        Value::String(author_name.to_string()),
    );
    metadata.insert(
        "channel_id".to_string(),
        Value::String(channel_id.to_string()),
    );
    metadata.insert(
        "message_id".to_string(),
        Value::String(message_id.to_string()),
    );
    metadata.insert("is_self".to_string(), Value::Bool(is_self));

    if !guild_id.is_empty() {
        metadata.insert("guild_id".to_string(), Value::String(guild_id));
    }

    if let Some(ref_msg) = data.get("referenced_message")
        && let Some(ref_id) = ref_msg.get("id").and_then(|v| v.as_str())
    {
        metadata.insert(
            "reply_to_message_id".to_string(),
            Value::String(ref_id.to_string()),
        );
    }

    build_event(stream_id, &raw, metadata, state).await
}

async fn normalize_guild_member_add(
    data: &Value,
    stream_id: &str,
    guild_filter: &[String],
    state: &mut StateStore,
) -> Option<Event> {
    let guild_id = data["guild_id"].as_str().unwrap_or("");
    if !guild_filter.is_empty() && !guild_filter.iter().any(|g| g == guild_id) {
        return None;
    }

    let user = &data["user"];
    let user_name = user
        .get("global_name")
        .and_then(|v| v.as_str())
        .or_else(|| user["username"].as_str())
        .unwrap_or("unknown");
    let user_id = user["id"].as_str().unwrap_or("unknown");

    let raw = format!("[guild:{guild_id}] {user_name} joined the server");

    let mut metadata = HashMap::new();
    metadata.insert(
        "event_type".to_string(),
        Value::String("guild_member_add".to_string()),
    );
    metadata.insert("user_id".to_string(), Value::String(user_id.to_string()));
    metadata.insert(
        "user_name".to_string(),
        Value::String(user_name.to_string()),
    );
    metadata.insert("guild_id".to_string(), Value::String(guild_id.to_string()));

    build_event(stream_id, &raw, metadata, state).await
}

async fn normalize_guild_member_remove(
    data: &Value,
    stream_id: &str,
    guild_filter: &[String],
    state: &mut StateStore,
) -> Option<Event> {
    let guild_id = data["guild_id"].as_str().unwrap_or("");
    if !guild_filter.is_empty() && !guild_filter.iter().any(|g| g == guild_id) {
        return None;
    }

    let user = &data["user"];
    let user_name = user
        .get("global_name")
        .and_then(|v| v.as_str())
        .or_else(|| user["username"].as_str())
        .unwrap_or("unknown");
    let user_id = user["id"].as_str().unwrap_or("unknown");

    let raw = format!("[guild:{guild_id}] {user_name} left the server");

    let mut metadata = HashMap::new();
    metadata.insert(
        "event_type".to_string(),
        Value::String("guild_member_remove".to_string()),
    );
    metadata.insert("user_id".to_string(), Value::String(user_id.to_string()));
    metadata.insert(
        "user_name".to_string(),
        Value::String(user_name.to_string()),
    );
    metadata.insert("guild_id".to_string(), Value::String(guild_id.to_string()));

    build_event(stream_id, &raw, metadata, state).await
}

async fn normalize_relationship_add(
    data: &Value,
    stream_id: &str,
    state: &mut StateStore,
) -> Option<Event> {
    let rel_type = data["type"].as_u64().unwrap_or(0);
    let user = &data["user"];
    let user_name = user
        .get("global_name")
        .and_then(|v| v.as_str())
        .or_else(|| user["username"].as_str())
        .unwrap_or("unknown");
    let user_id = user["id"].as_str().unwrap_or("unknown");

    // type 1 = friend, 3 = incoming request, 4 = outgoing request
    let (raw, event_type) = match rel_type {
        1 => (format!("{user_name} is now your friend"), "friend_added"),
        3 => (
            format!("{user_name} sent you a friend request"),
            "friend_request_received",
        ),
        4 => (
            format!("You sent a friend request to {user_name}"),
            "friend_request_sent",
        ),
        _ => (
            format!("Relationship update with {user_name} (type {rel_type})"),
            "relationship_update",
        ),
    };

    let mut metadata = HashMap::new();
    metadata.insert(
        "event_type".to_string(),
        Value::String(event_type.to_string()),
    );
    metadata.insert("user_id".to_string(), Value::String(user_id.to_string()));
    metadata.insert(
        "user_name".to_string(),
        Value::String(user_name.to_string()),
    );
    metadata.insert(
        "relationship_type".to_string(),
        Value::Number(rel_type.into()),
    );

    build_event(stream_id, &raw, metadata, state).await
}

async fn normalize_relationship_remove(
    data: &Value,
    stream_id: &str,
    state: &mut StateStore,
) -> Option<Event> {
    let user_id = data["id"].as_str().unwrap_or("unknown");
    let raw = format!("Relationship with user {user_id} removed");

    let mut metadata = HashMap::new();
    metadata.insert(
        "event_type".to_string(),
        Value::String("relationship_removed".to_string()),
    );
    metadata.insert("user_id".to_string(), Value::String(user_id.to_string()));

    build_event(stream_id, &raw, metadata, state).await
}

async fn normalize_channel_create(
    data: &Value,
    stream_id: &str,
    state: &mut StateStore,
) -> Option<Event> {
    let channel_name = data["name"].as_str().unwrap_or("unknown");
    let channel_id = data["id"].as_str().unwrap_or("unknown");
    let guild_id = data["guild_id"].as_str().unwrap_or("");

    let raw = if guild_id.is_empty() {
        format!("New DM channel created: {channel_name}")
    } else {
        format!("[guild:{guild_id}] New channel created: #{channel_name}")
    };

    let mut metadata = HashMap::new();
    metadata.insert(
        "event_type".to_string(),
        Value::String("channel_create".to_string()),
    );
    metadata.insert(
        "channel_id".to_string(),
        Value::String(channel_id.to_string()),
    );
    metadata.insert(
        "channel_name".to_string(),
        Value::String(channel_name.to_string()),
    );
    if !guild_id.is_empty() {
        metadata.insert("guild_id".to_string(), Value::String(guild_id.to_string()));
    }

    build_event(stream_id, &raw, metadata, state).await
}

async fn build_event(
    stream_id: &str,
    raw: &str,
    metadata: HashMap<String, Value>,
    state: &mut StateStore,
) -> Option<Event> {
    let seq = match state.incr(&names::seq_key(stream_id)).await {
        Ok(s) => s,
        Err(e) => {
            warn!(error = %e, "failed to increment sequence counter");
            return None;
        }
    };

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    Some(Event {
        id: uuid::Uuid::new_v4().to_string(),
        stream_id: stream_id.to_string(),
        sequence: seq,
        timestamp: now,
        raw: raw.to_string(),
        embedding: None,
        metadata,
    })
}
