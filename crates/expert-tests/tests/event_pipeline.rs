use expert_redis::StreamProducer;
use expert_tests::*;
use expert_types::context::{ActivityExchange, Exchange, SelfKnowledgeNode, ToolCall};
use expert_types::signals::BotIdentity;
use expert_types::training::{TrainingBatch, TrainingBatchRequest};

#[tokio::test]
async fn test_event_serialization_roundtrip() {
    let mut conn = redis_conn().await;
    flush_redis(&mut conn).await;

    let stream = unique_stream("pipe.event");
    let mut producer = StreamProducer::new(conn.clone(), 1000);
    let event = fake_event("s1");

    producer.publish(&stream, &event).await.unwrap();

    let mut consumer =
        expert_redis::StreamConsumer::new(conn.clone(), stream, "grp".into(), "c0".into(), 1000)
            .await
            .unwrap();

    let (_, deserialized) = consumer
        .consume::<expert_types::event::Event>()
        .await
        .unwrap()
        .unwrap();

    assert_eq!(deserialized.id, event.id);
    assert_eq!(deserialized.stream_id, event.stream_id);
    assert_eq!(deserialized.sequence, event.sequence);
    assert_eq!(deserialized.raw, event.raw);
    assert_eq!(deserialized.embedding, event.embedding);
}

#[tokio::test]
async fn test_fire_signal_roundtrip() {
    let mut conn = redis_conn().await;
    flush_redis(&mut conn).await;

    let stream = unique_stream("pipe.fire");
    let mut producer = StreamProducer::new(conn.clone(), 1000);
    let signal = fake_fire_signal("act-1", "s1");

    producer.publish(&stream, &signal).await.unwrap();

    let mut consumer =
        expert_redis::StreamConsumer::new(conn.clone(), stream, "grp".into(), "c0".into(), 1000)
            .await
            .unwrap();

    let (_, deserialized) = consumer
        .consume::<expert_types::signals::FireSignal>()
        .await
        .unwrap()
        .unwrap();

    assert_eq!(deserialized.activity_id, signal.activity_id);
    assert_eq!(deserialized.stream_id, signal.stream_id);
    assert_eq!(deserialized.firing_goal_ids, signal.firing_goal_ids);
    assert_eq!(deserialized.scores, signal.scores);
    assert_eq!(deserialized.trigger_event_seq, signal.trigger_event_seq);
    assert_eq!(deserialized.timestamp, signal.timestamp);
}

#[tokio::test]
async fn test_assemble_request_roundtrip() {
    let mut conn = redis_conn().await;
    flush_redis(&mut conn).await;

    let stream = unique_stream("pipe.asm");
    let mut producer = StreamProducer::new(conn.clone(), 1000);
    let req = fake_assemble_request("act-1", "s1");

    producer.publish(&stream, &req).await.unwrap();

    let mut consumer =
        expert_redis::StreamConsumer::new(conn.clone(), stream, "grp".into(), "c0".into(), 1000)
            .await
            .unwrap();

    let (_, deserialized) = consumer
        .consume::<expert_types::signals::AssembleRequest>()
        .await
        .unwrap()
        .unwrap();

    assert_eq!(deserialized.activity_id, req.activity_id);
    assert_eq!(deserialized.stream_id, req.stream_id);
    assert_eq!(deserialized.goal_tree.len(), req.goal_tree.len());
    assert_eq!(
        deserialized.tool_definitions.len(),
        req.tool_definitions.len()
    );
    assert_eq!(
        deserialized.fire_signal.activity_id,
        req.fire_signal.activity_id
    );
}

#[tokio::test]
async fn test_training_example_roundtrip() {
    let mut conn = redis_conn().await;
    flush_redis(&mut conn).await;

    let stream = unique_stream("pipe.train");
    let mut producer = StreamProducer::new(conn.clone(), 1000);
    let example = fake_training_example();

    producer.publish(&stream, &example).await.unwrap();

    let mut consumer =
        expert_redis::StreamConsumer::new(conn.clone(), stream, "grp".into(), "c0".into(), 1000)
            .await
            .unwrap();

    let (_, deserialized) = consumer
        .consume::<expert_types::training::TrainingExample>()
        .await
        .unwrap()
        .unwrap();

    assert_eq!(deserialized.id, example.id);
    assert_eq!(deserialized.activity_id, example.activity_id);
    assert_eq!(deserialized.label, example.label);
    assert_eq!(deserialized.label_source, example.label_source);
}

#[test]
fn activity_exchange_serde_roundtrip() {
    let original = ActivityExchange {
        activity_id: "act-pipe".to_string(),
        exchange: Exchange {
            timestamp: 42,
            rendered_prompt: "rendered".to_string(),
            response: "resp".to_string(),
            tool_calls: vec![ToolCall {
                tool_name: "suppress".to_string(),
                arguments: serde_json::json!({}),
                result: None,
            }],
        },
    };

    let json = serde_json::to_string(&original).expect("serialize ActivityExchange");
    let back: ActivityExchange = serde_json::from_str(&json).expect("deserialize ActivityExchange");
    assert_eq!(back.activity_id, original.activity_id);
    assert_eq!(back.exchange.response, original.exchange.response);
}

#[test]
fn training_batch_request_serde_roundtrip() {
    let original = TrainingBatchRequest {
        request_id: "req-xyz".to_string(),
        domain: Some("test".to_string()),
        goal_id: Some("goal-1".to_string()),
        batch_size: 16,
        min_confidence: 0.75,
    };

    let json = serde_json::to_string(&original).expect("serialize TrainingBatchRequest");
    let back: TrainingBatchRequest =
        serde_json::from_str(&json).expect("deserialize TrainingBatchRequest");

    assert_eq!(back.request_id, original.request_id);
    assert_eq!(back.domain, original.domain);
    assert_eq!(back.goal_id, original.goal_id);
    assert_eq!(back.batch_size, original.batch_size);
    assert_eq!(back.min_confidence, original.min_confidence);
}

#[tokio::test]
async fn test_assemble_request_with_bot_identity_roundtrip() {
    let mut conn = redis_conn().await;
    flush_redis(&mut conn).await;

    let stream = unique_stream("pipe.asm.identity");
    let mut producer = StreamProducer::new(conn.clone(), 1000);

    let mut req = fake_assemble_request("act-id", "s1");
    req.bot_identity = Some(BotIdentity {
        username: "zero".to_string(),
        user_id: "99999".to_string(),
        display_name: Some("Zero Bot".to_string()),
    });

    producer.publish(&stream, &req).await.unwrap();

    let mut consumer =
        expert_redis::StreamConsumer::new(conn.clone(), stream, "grp".into(), "c0".into(), 1000)
            .await
            .unwrap();

    let (_, deserialized) = consumer
        .consume::<expert_types::signals::AssembleRequest>()
        .await
        .unwrap()
        .unwrap();

    let identity = deserialized.bot_identity.unwrap();
    assert_eq!(identity.username, "zero");
    assert_eq!(identity.user_id, "99999");
    assert_eq!(identity.display_name.as_deref(), Some("Zero Bot"));
}

#[tokio::test]
async fn test_assemble_request_without_bot_identity_roundtrip() {
    let mut conn = redis_conn().await;
    flush_redis(&mut conn).await;

    let stream = unique_stream("pipe.asm.noid");
    let mut producer = StreamProducer::new(conn.clone(), 1000);
    let req = fake_assemble_request("act-noid", "s1");

    producer.publish(&stream, &req).await.unwrap();

    let mut consumer =
        expert_redis::StreamConsumer::new(conn.clone(), stream, "grp".into(), "c0".into(), 1000)
            .await
            .unwrap();

    let (_, deserialized) = consumer
        .consume::<expert_types::signals::AssembleRequest>()
        .await
        .unwrap()
        .unwrap();

    assert!(deserialized.bot_identity.is_none());
}

#[tokio::test]
async fn test_self_knowledge_node_redis_roundtrip() {
    let mut conn = redis_conn().await;
    flush_redis(&mut conn).await;

    let stream = unique_stream("pipe.sk");
    let mut producer = StreamProducer::new(conn.clone(), 1000);
    let node = SelfKnowledgeNode {
        id: uuid::Uuid::new_v4().to_string(),
        category: "core_identity".to_string(),
        content: "I am Zero, an autonomous expert system.".to_string(),
        embedding: vec![0.1, 0.2, 0.3, 0.4],
        created_at: 1000,
        updated_at: 2000,
    };

    producer.publish(&stream, &node).await.unwrap();

    let mut consumer =
        expert_redis::StreamConsumer::new(conn.clone(), stream, "grp".into(), "c0".into(), 1000)
            .await
            .unwrap();

    let (_, deserialized) = consumer
        .consume::<SelfKnowledgeNode>()
        .await
        .unwrap()
        .unwrap();

    assert_eq!(deserialized.id, node.id);
    assert_eq!(deserialized.category, "core_identity");
    assert_eq!(deserialized.content, node.content);
    assert_eq!(deserialized.embedding, vec![0.1, 0.2, 0.3, 0.4]);
    assert_eq!(deserialized.created_at, 1000);
    assert_eq!(deserialized.updated_at, 2000);
}

#[tokio::test]
async fn test_self_knowledge_node_empty_embedding_roundtrip() {
    let mut conn = redis_conn().await;
    flush_redis(&mut conn).await;

    let stream = unique_stream("pipe.sk.empty");
    let mut producer = StreamProducer::new(conn.clone(), 1000);
    let node = SelfKnowledgeNode {
        id: uuid::Uuid::new_v4().to_string(),
        category: "preference".to_string(),
        content: "I like Rust.".to_string(),
        embedding: Vec::new(),
        created_at: 3000,
        updated_at: 3000,
    };

    producer.publish(&stream, &node).await.unwrap();

    let mut consumer =
        expert_redis::StreamConsumer::new(conn.clone(), stream, "grp".into(), "c0".into(), 1000)
            .await
            .unwrap();

    let (_, deserialized) = consumer
        .consume::<SelfKnowledgeNode>()
        .await
        .unwrap()
        .unwrap();

    assert_eq!(deserialized.category, "preference");
    assert!(deserialized.embedding.is_empty());
}

#[tokio::test]
async fn test_bot_identity_redis_roundtrip() {
    let mut conn = redis_conn().await;
    flush_redis(&mut conn).await;

    let stream = unique_stream("pipe.botid");
    let mut producer = StreamProducer::new(conn.clone(), 1000);
    let identity = BotIdentity {
        username: "zero".to_string(),
        user_id: "42".to_string(),
        display_name: None,
    };

    producer.publish(&stream, &identity).await.unwrap();

    let mut consumer =
        expert_redis::StreamConsumer::new(conn.clone(), stream, "grp".into(), "c0".into(), 1000)
            .await
            .unwrap();

    let (_, deserialized) = consumer.consume::<BotIdentity>().await.unwrap().unwrap();

    assert_eq!(deserialized.username, "zero");
    assert_eq!(deserialized.user_id, "42");
    assert!(deserialized.display_name.is_none());
}

#[test]
fn self_knowledge_node_serde_roundtrip() {
    let node = SelfKnowledgeNode {
        id: "sk-test".to_string(),
        category: "reflection".to_string(),
        content: "I have learned to be patient.".to_string(),
        embedding: vec![0.5; 8],
        created_at: 5000,
        updated_at: 6000,
    };

    let json = serde_json::to_string(&node).expect("serialize SelfKnowledgeNode");
    let back: SelfKnowledgeNode =
        serde_json::from_str(&json).expect("deserialize SelfKnowledgeNode");
    assert_eq!(back.id, node.id);
    assert_eq!(back.category, "reflection");
    assert_eq!(back.content, node.content);
    assert_eq!(back.embedding.len(), 8);
}

#[test]
fn bot_identity_serde_roundtrip() {
    let identity = BotIdentity {
        username: "zero".to_string(),
        user_id: "12345".to_string(),
        display_name: Some("Zero Display".to_string()),
    };

    let json = serde_json::to_string(&identity).expect("serialize BotIdentity");
    let back: BotIdentity = serde_json::from_str(&json).expect("deserialize BotIdentity");
    assert_eq!(back.username, identity.username);
    assert_eq!(back.user_id, identity.user_id);
    assert_eq!(back.display_name, identity.display_name);
}

#[test]
fn training_batch_serde_roundtrip() {
    let ex = fake_training_example();
    let original = TrainingBatch {
        request_id: "batch-req-1".to_string(),
        examples: vec![ex.clone()],
        positive_count: 1,
        negative_count: 0,
    };

    let json = serde_json::to_string(&original).expect("serialize TrainingBatch");
    let back: TrainingBatch = serde_json::from_str(&json).expect("deserialize TrainingBatch");

    assert_eq!(back.request_id, original.request_id);
    assert_eq!(back.examples.len(), 1);
    assert_eq!(back.examples[0].id, ex.id);
    assert_eq!(back.positive_count, original.positive_count);
    assert_eq!(back.negative_count, original.negative_count);
}
