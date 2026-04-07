use expert_redis::{StreamConsumer, StreamProducer};
use expert_tests::*;

#[tokio::test]
async fn test_publish_and_consume() {
    let mut conn = redis_conn().await;
    flush_redis(&mut conn).await;

    let stream = unique_stream("test.streams");
    let mut producer = StreamProducer::new(conn.clone(), 1000);
    let event = fake_event("s1");

    let id = producer.publish(&stream, &event).await.unwrap();
    assert!(!id.is_empty());

    let mut consumer = StreamConsumer::new(
        conn.clone(), stream.clone(),
        "grp".to_string(), "c0".to_string(), 1000,
    ).await.unwrap();

    let result = consumer.consume::<expert_types::event::Event>().await.unwrap();
    assert!(result.is_some());
    let (entry_id, msg) = result.unwrap();
    assert_eq!(msg.id, event.id);

    consumer.ack(&entry_id).await.unwrap();
}

#[tokio::test]
async fn test_consume_timeout() {
    let mut conn = redis_conn().await;
    flush_redis(&mut conn).await;

    let stream = unique_stream("test.empty");
    let mut consumer = StreamConsumer::new(
        conn.clone(), stream, "grp".to_string(), "c0".to_string(), 100,
    ).await.unwrap();

    let result = consumer.consume::<expert_types::event::Event>().await.unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn test_consumer_group_idempotent() {
    let mut conn = redis_conn().await;
    flush_redis(&mut conn).await;

    let stream = unique_stream("test.idem");
    let _c1 = StreamConsumer::new(
        conn.clone(), stream.clone(), "grp".to_string(), "c0".to_string(), 100,
    ).await.unwrap();

    let c2 = StreamConsumer::new(
        conn.clone(), stream, "grp".to_string(), "c1".to_string(), 100,
    ).await;
    assert!(c2.is_ok());
}

#[tokio::test]
async fn test_xrevrange() {
    let mut conn = redis_conn().await;
    flush_redis(&mut conn).await;

    let stream = unique_stream("test.xrev");
    let mut producer = StreamProducer::new(conn.clone(), 1000);

    let mut ids = Vec::new();
    for i in 0..5 {
        let mut event = fake_event("s1");
        event.sequence = i;
        event.raw = format!("event-{i}");
        let id = producer.publish(&stream, &event).await.unwrap();
        ids.push(id);
    }

    let results: Vec<(String, expert_types::event::Event)> =
        expert_redis::streams::xrevrange(&mut conn, &stream, "+", "-", 3)
            .await
            .unwrap();

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].1.raw, "event-4");
    assert_eq!(results[1].1.raw, "event-3");
    assert_eq!(results[2].1.raw, "event-2");
}

#[tokio::test]
async fn test_maxlen_trimming() {
    let mut conn = redis_conn().await;
    flush_redis(&mut conn).await;

    let stream = unique_stream("test.maxlen");
    let mut producer = StreamProducer::new(conn.clone(), 5);

    for i in 0..20 {
        let mut event = fake_event("s1");
        event.sequence = i;
        producer.publish(&stream, &event).await.unwrap();
    }

    let all: Vec<(String, expert_types::event::Event)> =
        expert_redis::streams::xrevrange(&mut conn, &stream, "+", "-", 100)
            .await
            .unwrap();

    // MAXLEN ~ is approximate, but should be roughly capped
    assert!(all.len() <= 10, "expected ~5 entries, got {}", all.len());
}

#[tokio::test]
async fn test_publish_multiple_types() {
    let mut conn = redis_conn().await;
    flush_redis(&mut conn).await;

    let stream = unique_stream("test.types");
    let mut producer = StreamProducer::new(conn.clone(), 1000);

    let signal = fake_fire_signal("act-1", "s1");
    let id = producer.publish(&stream, &signal).await.unwrap();
    assert!(!id.is_empty());

    let mut consumer = StreamConsumer::new(
        conn.clone(), stream, "grp".to_string(), "c0".to_string(), 1000,
    ).await.unwrap();

    let result = consumer.consume::<expert_types::signals::FireSignal>().await.unwrap();
    assert!(result.is_some());
    let (_, msg) = result.unwrap();
    assert_eq!(msg.activity_id, "act-1");
}
