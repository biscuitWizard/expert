//! Integration tests for discord-adapter data contracts.
//!
//! These tests verify that the Event and action payload formats
//! used by the Discord adapter serialize/deserialize correctly
//! through Redis streams — the same path used at runtime.

use std::collections::HashMap;

use expert_redis::StreamProducer;
use expert_redis::names;
use expert_tests::*;
use expert_types::event::Event;
use expert_types::signals::ToolDefinition;
use serde::{Deserialize, Serialize};

/// Action payload format published by llm-gateway and consumed by discord-adapter.
#[derive(Debug, Serialize, Deserialize)]
struct ActionPayload {
    tool_name: String,
    arguments: serde_json::Value,
    invocation_id: Option<String>,
}

fn fake_discord_message_event(stream_id: &str) -> Event {
    let mut metadata = HashMap::new();
    metadata.insert("event_type".to_string(), serde_json::json!("message"));
    metadata.insert("author_id".to_string(), serde_json::json!("111222333"));
    metadata.insert("author_name".to_string(), serde_json::json!("TestUser"));
    metadata.insert("channel_id".to_string(), serde_json::json!("444555666"));
    metadata.insert("message_id".to_string(), serde_json::json!("777888999"));
    metadata.insert("is_self".to_string(), serde_json::json!(false));
    metadata.insert("guild_id".to_string(), serde_json::json!("100200300"));

    Event {
        id: uuid::Uuid::new_v4().to_string(),
        stream_id: stream_id.to_string(),
        sequence: 1,
        timestamp: 1000,
        raw: "[#general] TestUser: Hello everyone!".to_string(),
        embedding: None,
        metadata,
    }
}

fn fake_discord_dm_event(stream_id: &str) -> Event {
    let mut metadata = HashMap::new();
    metadata.insert("event_type".to_string(), serde_json::json!("dm"));
    metadata.insert("author_id".to_string(), serde_json::json!("111222333"));
    metadata.insert("author_name".to_string(), serde_json::json!("FriendUser"));
    metadata.insert("channel_id".to_string(), serde_json::json!("dm-channel-1"));
    metadata.insert("message_id".to_string(), serde_json::json!("msg-dm-1"));
    metadata.insert("is_self".to_string(), serde_json::json!(false));

    Event {
        id: uuid::Uuid::new_v4().to_string(),
        stream_id: stream_id.to_string(),
        sequence: 2,
        timestamp: 2000,
        raw: "[DM from FriendUser] Hey there!".to_string(),
        embedding: None,
        metadata,
    }
}

fn fake_discord_friend_request_event(stream_id: &str) -> Event {
    let mut metadata = HashMap::new();
    metadata.insert(
        "event_type".to_string(),
        serde_json::json!("friend_request_received"),
    );
    metadata.insert("user_id".to_string(), serde_json::json!("999888777"));
    metadata.insert("user_name".to_string(), serde_json::json!("NewFriend"));
    metadata.insert("relationship_type".to_string(), serde_json::json!(3));

    Event {
        id: uuid::Uuid::new_v4().to_string(),
        stream_id: stream_id.to_string(),
        sequence: 3,
        timestamp: 3000,
        raw: "NewFriend sent you a friend request".to_string(),
        embedding: None,
        metadata,
    }
}

// --- Event roundtrip tests ---

#[tokio::test]
async fn discord_message_event_roundtrip() {
    let mut conn = redis_conn().await;
    flush_redis(&mut conn).await;

    let stream_id = "discord-test";
    let stream = names::events_raw(stream_id);
    let mut producer = StreamProducer::new(conn.clone(), 1000);
    let event = fake_discord_message_event(stream_id);

    producer.publish(&stream, &event).await.unwrap();

    let mut consumer =
        expert_redis::StreamConsumer::new(conn.clone(), stream, "grp".into(), "c0".into(), 1000)
            .await
            .unwrap();

    let (_, deserialized) = consumer.consume::<Event>().await.unwrap().unwrap();

    assert_eq!(deserialized.id, event.id);
    assert_eq!(deserialized.stream_id, stream_id);
    assert_eq!(deserialized.raw, "[#general] TestUser: Hello everyone!");
    assert!(deserialized.embedding.is_none());
    assert_eq!(deserialized.metadata["event_type"], "message");
    assert_eq!(deserialized.metadata["author_id"], "111222333");
    assert_eq!(deserialized.metadata["author_name"], "TestUser");
    assert_eq!(deserialized.metadata["channel_id"], "444555666");
    assert_eq!(deserialized.metadata["guild_id"], "100200300");
    assert_eq!(deserialized.metadata["is_self"], false);
}

#[tokio::test]
async fn discord_dm_event_roundtrip() {
    let mut conn = redis_conn().await;
    flush_redis(&mut conn).await;

    let stream_id = "discord-dm-test";
    let stream = names::events_raw(stream_id);
    let mut producer = StreamProducer::new(conn.clone(), 1000);
    let event = fake_discord_dm_event(stream_id);

    producer.publish(&stream, &event).await.unwrap();

    let mut consumer =
        expert_redis::StreamConsumer::new(conn.clone(), stream, "grp".into(), "c0".into(), 1000)
            .await
            .unwrap();

    let (_, deserialized) = consumer.consume::<Event>().await.unwrap().unwrap();

    assert_eq!(deserialized.metadata["event_type"], "dm");
    assert_eq!(deserialized.raw, "[DM from FriendUser] Hey there!");
    assert!(deserialized.metadata.get("guild_id").is_none());
}

#[tokio::test]
async fn discord_friend_request_event_roundtrip() {
    let mut conn = redis_conn().await;
    flush_redis(&mut conn).await;

    let stream_id = "discord-friend-test";
    let stream = names::events_raw(stream_id);
    let mut producer = StreamProducer::new(conn.clone(), 1000);
    let event = fake_discord_friend_request_event(stream_id);

    producer.publish(&stream, &event).await.unwrap();

    let mut consumer =
        expert_redis::StreamConsumer::new(conn.clone(), stream, "grp".into(), "c0".into(), 1000)
            .await
            .unwrap();

    let (_, deserialized) = consumer.consume::<Event>().await.unwrap().unwrap();

    assert_eq!(
        deserialized.metadata["event_type"],
        "friend_request_received"
    );
    assert_eq!(deserialized.metadata["user_name"], "NewFriend");
    assert_eq!(deserialized.metadata["relationship_type"], 3);
    assert_eq!(deserialized.raw, "NewFriend sent you a friend request");
}

// --- Action payload roundtrip tests ---

#[tokio::test]
async fn action_payload_roundtrip_send_message() {
    let mut conn = redis_conn().await;
    flush_redis(&mut conn).await;

    let stream_id = "discord-action-test";
    let action_stream = names::actions(stream_id);
    let mut producer = StreamProducer::new(conn.clone(), 1000);

    let payload = ActionPayload {
        tool_name: "send_message".to_string(),
        arguments: serde_json::json!({
            "channel_id": "444555666",
            "content": "Hello from the LLM!"
        }),
        invocation_id: Some(uuid::Uuid::new_v4().to_string()),
    };

    producer.publish(&action_stream, &payload).await.unwrap();

    let mut consumer = expert_redis::StreamConsumer::new(
        conn.clone(),
        action_stream,
        format!("adapter-{stream_id}"),
        "adapter-0".into(),
        1000,
    )
    .await
    .unwrap();

    let (id, deserialized) = consumer.consume::<ActionPayload>().await.unwrap().unwrap();

    assert!(!id.is_empty());
    assert_eq!(deserialized.tool_name, "send_message");
    assert_eq!(deserialized.arguments["channel_id"], "444555666");
    assert_eq!(deserialized.arguments["content"], "Hello from the LLM!");
    assert!(deserialized.invocation_id.is_some());
}

#[tokio::test]
async fn action_payload_roundtrip_send_dm() {
    let mut conn = redis_conn().await;
    flush_redis(&mut conn).await;

    let action_stream = names::actions("discord-dm-action");
    let mut producer = StreamProducer::new(conn.clone(), 1000);

    let payload = ActionPayload {
        tool_name: "send_dm".to_string(),
        arguments: serde_json::json!({
            "user_id": "111222333",
            "content": "Hey! This is a DM."
        }),
        invocation_id: None,
    };

    producer.publish(&action_stream, &payload).await.unwrap();

    let mut consumer = expert_redis::StreamConsumer::new(
        conn.clone(),
        action_stream,
        "adapter-dm-action".into(),
        "adapter-0".into(),
        1000,
    )
    .await
    .unwrap();

    let (_, deserialized) = consumer.consume::<ActionPayload>().await.unwrap().unwrap();

    assert_eq!(deserialized.tool_name, "send_dm");
    assert_eq!(deserialized.arguments["user_id"], "111222333");
    assert!(deserialized.invocation_id.is_none());
}

#[tokio::test]
async fn action_payload_roundtrip_friend_request() {
    let mut conn = redis_conn().await;
    flush_redis(&mut conn).await;

    let action_stream = names::actions("discord-friend-action");
    let mut producer = StreamProducer::new(conn.clone(), 1000);

    let payload = ActionPayload {
        tool_name: "accept_friend_request".to_string(),
        arguments: serde_json::json!({ "user_id": "999888777" }),
        invocation_id: Some("inv-123".to_string()),
    };

    producer.publish(&action_stream, &payload).await.unwrap();

    let mut consumer = expert_redis::StreamConsumer::new(
        conn.clone(),
        action_stream,
        "adapter-friend-action".into(),
        "adapter-0".into(),
        1000,
    )
    .await
    .unwrap();

    let (_, deserialized) = consumer.consume::<ActionPayload>().await.unwrap().unwrap();

    assert_eq!(deserialized.tool_name, "accept_friend_request");
    assert_eq!(deserialized.arguments["user_id"], "999888777");
}

// --- Discord tool definition contract tests ---

#[test]
fn discord_tool_definitions_are_valid_for_llm_gateway() {
    // Verify tool definitions can be serialized the same way llm-gateway does
    let defs = discord_tool_definitions();
    let tools_json: Vec<serde_json::Value> = defs
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
        .collect();

    assert_eq!(tools_json.len(), 10);

    for tool in &tools_json {
        assert_eq!(tool["type"], "function");
        let func = &tool["function"];
        assert!(func["name"].is_string());
        assert!(func["description"].is_string());
        assert!(func["parameters"].is_object());
        assert!(func["parameters"]["type"] == "object");
    }
}

#[test]
fn discord_tool_definitions_serde_roundtrip() {
    let defs = discord_tool_definitions();
    let json = serde_json::to_string(&defs).unwrap();
    let back: Vec<ToolDefinition> = serde_json::from_str(&json).unwrap();

    assert_eq!(back.len(), defs.len());
    for (a, b) in defs.iter().zip(back.iter()) {
        assert_eq!(a.name, b.name);
        assert_eq!(a.description, b.description);
        assert_eq!(a.is_domain_tool, b.is_domain_tool);
        assert_eq!(a.parameters_schema, b.parameters_schema);
    }
}

// --- Sequence counter integration ---

#[tokio::test]
async fn sequence_counter_increments() {
    let mut conn = redis_conn().await;
    flush_redis(&mut conn).await;

    let mut state = expert_redis::StateStore::new(conn.clone());
    let key = names::seq_key("discord-seq-test");

    let s1 = state.incr(&key).await.unwrap();
    let s2 = state.incr(&key).await.unwrap();
    let s3 = state.incr(&key).await.unwrap();

    assert_eq!(s1, 1);
    assert_eq!(s2, 2);
    assert_eq!(s3, 3);
}

// --- Action result event contract ---

#[tokio::test]
async fn action_result_event_roundtrip() {
    let mut conn = redis_conn().await;
    flush_redis(&mut conn).await;

    let stream_id = "discord-result-test";
    let stream = names::events_raw(stream_id);
    let mut producer = StreamProducer::new(conn.clone(), 1000);

    let mut metadata = HashMap::new();
    metadata.insert("event_type".to_string(), serde_json::json!("action_result"));
    metadata.insert("tool_name".to_string(), serde_json::json!("send_message"));
    metadata.insert("is_self".to_string(), serde_json::json!(true));
    metadata.insert(
        "action_result".to_string(),
        serde_json::json!({"status": "success", "result": {"id": "msg-new-1"}}),
    );

    let event = Event {
        id: uuid::Uuid::new_v4().to_string(),
        stream_id: stream_id.to_string(),
        sequence: 42,
        timestamp: 5000,
        raw: "[action:send_message] success".to_string(),
        embedding: None,
        metadata,
    };

    producer.publish(&stream, &event).await.unwrap();

    let mut consumer =
        expert_redis::StreamConsumer::new(conn.clone(), stream, "grp".into(), "c0".into(), 1000)
            .await
            .unwrap();

    let (_, deserialized) = consumer.consume::<Event>().await.unwrap().unwrap();

    assert_eq!(deserialized.metadata["event_type"], "action_result");
    assert_eq!(deserialized.metadata["tool_name"], "send_message");
    assert_eq!(deserialized.metadata["is_self"], true);
    assert_eq!(deserialized.metadata["action_result"]["status"], "success");
}

/// Helper: build Discord tool definitions (mirrors discord-adapter/src/tools.rs).
/// Duplicated here because discord-adapter is a binary crate, not a library.
fn discord_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        tool(
            "send_message",
            "Send a message to a Discord channel.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "channel_id": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["channel_id", "content"]
            }),
        ),
        tool(
            "reply_to_message",
            "Reply to a specific message.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "channel_id": {"type": "string"},
                    "message_id": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["channel_id", "message_id", "content"]
            }),
        ),
        tool(
            "send_dm",
            "Send a direct message.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["user_id", "content"]
            }),
        ),
        tool(
            "react_to_message",
            "Add a reaction.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "channel_id": {"type": "string"},
                    "message_id": {"type": "string"},
                    "emoji": {"type": "string"}
                },
                "required": ["channel_id", "message_id", "emoji"]
            }),
        ),
        tool(
            "join_guild",
            "Join a guild via invite.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "invite_code": {"type": "string"}
                },
                "required": ["invite_code"]
            }),
        ),
        tool(
            "leave_guild",
            "Leave a guild.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "guild_id": {"type": "string"}
                },
                "required": ["guild_id"]
            }),
        ),
        tool(
            "send_friend_request",
            "Send a friend request.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "username": {"type": "string"}
                },
                "required": ["username"]
            }),
        ),
        tool(
            "accept_friend_request",
            "Accept a friend request.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"}
                },
                "required": ["user_id"]
            }),
        ),
        tool(
            "remove_friend",
            "Remove a friend.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"}
                },
                "required": ["user_id"]
            }),
        ),
        tool(
            "typing_indicator",
            "Show typing indicator.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "channel_id": {"type": "string"}
                },
                "required": ["channel_id"]
            }),
        ),
    ]
}

fn tool(name: &str, desc: &str, schema: serde_json::Value) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        description: desc.to_string(),
        parameters_schema: schema,
        is_domain_tool: true,
    }
}
