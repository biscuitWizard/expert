use expert_redis::StateStore;
use expert_tests::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct TestData {
    name: String,
    value: i32,
}

#[tokio::test]
async fn test_json_roundtrip() {
    let mut conn = redis_conn().await;
    flush_redis(&mut conn).await;

    let mut store = StateStore::new(conn);
    let data = TestData { name: "hello".into(), value: 42 };

    store.set_json("test:json:rt", &data).await.unwrap();
    let result: Option<TestData> = store.get_json("test:json:rt").await.unwrap();
    assert_eq!(result, Some(data));
}

#[tokio::test]
async fn test_get_missing_key() {
    let mut conn = redis_conn().await;
    flush_redis(&mut conn).await;

    let mut store = StateStore::new(conn);
    let result: Option<TestData> = store.get_json("nonexistent:key").await.unwrap();
    assert!(result.is_none());
}

#[tokio::test]
async fn test_set_json_ex_expires() {
    let mut conn = redis_conn().await;
    flush_redis(&mut conn).await;

    let mut store = StateStore::new(conn);
    let data = TestData { name: "ephemeral".into(), value: 1 };

    store.set_json_ex("test:ttl", &data, 1).await.unwrap();
    let before: Option<TestData> = store.get_json("test:ttl").await.unwrap();
    assert_eq!(before, Some(data));

    tokio::time::sleep(std::time::Duration::from_millis(1200)).await;
    let after: Option<TestData> = store.get_json("test:ttl").await.unwrap();
    assert!(after.is_none());
}

#[tokio::test]
async fn test_incr() {
    let mut conn = redis_conn().await;
    flush_redis(&mut conn).await;

    let mut store = StateStore::new(conn);
    let v1 = store.incr("test:counter").await.unwrap();
    let v2 = store.incr("test:counter").await.unwrap();
    let v3 = store.incr("test:counter").await.unwrap();
    assert_eq!(v1, 1);
    assert_eq!(v2, 2);
    assert_eq!(v3, 3);
}

#[tokio::test]
async fn test_del() {
    let mut conn = redis_conn().await;
    flush_redis(&mut conn).await;

    let mut store = StateStore::new(conn);
    store.set_str("test:del", "value").await.unwrap();

    let before = store.get_str("test:del").await.unwrap();
    assert_eq!(before, Some("value".to_string()));

    store.del("test:del").await.unwrap();
    let after = store.get_str("test:del").await.unwrap();
    assert!(after.is_none());
}
