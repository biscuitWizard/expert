use expert_tests::*;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{CreateCollectionBuilder, Distance, VectorParamsBuilder};

fn qdrant_url() -> String {
    std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://127.0.0.1:6334".to_string())
}

async fn qdrant_client() -> Qdrant {
    Qdrant::from_url(&qdrant_url())
        .build()
        .expect("qdrant client failed")
}

async fn ensure_clean_collection(client: &Qdrant, name: &str, dim: u64) {
    let _ = client.delete_collection(name).await;
    client
        .create_collection(
            CreateCollectionBuilder::new(name)
                .vectors_config(VectorParamsBuilder::new(dim, Distance::Cosine)),
        )
        .await
        .expect("create collection failed");
}

#[tokio::test]
async fn test_collection_creation() {
    let client = qdrant_client().await;
    let name = format!("test_create_{}", uuid::Uuid::new_v4());

    let _ = client.delete_collection(&name).await;
    assert!(!client.collection_exists(&name).await.unwrap());

    client
        .create_collection(
            CreateCollectionBuilder::new(&name)
                .vectors_config(VectorParamsBuilder::new(8, Distance::Cosine)),
        )
        .await
        .unwrap();

    assert!(client.collection_exists(&name).await.unwrap());
    client.delete_collection(&name).await.unwrap();
}

#[tokio::test]
async fn test_episode_insert_and_search() {
    let client = qdrant_client().await;
    let dim = 8u64;
    let col_name = format!("test_episodes_{}", uuid::Uuid::new_v4());
    ensure_clean_collection(&client, &col_name, dim).await;

    let episode = fake_episode(dim as usize);

    let payload: qdrant_client::Payload = serde_json::json!({
        "activity_id": episode.activity_id,
        "goal_id": episode.goal_id,
        "domain": episode.domain,
    })
    .try_into()
    .unwrap();

    let point = qdrant_client::qdrant::PointStruct::new(
        episode.id.clone(),
        episode.embedding.clone(),
        payload,
    );

    client
        .upsert_points(qdrant_client::qdrant::UpsertPointsBuilder::new(
            &col_name,
            vec![point],
        ))
        .await
        .unwrap();

    // Small delay for indexing
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let results = client
        .search_points(
            qdrant_client::qdrant::SearchPointsBuilder::new(
                &col_name,
                episode.embedding.clone(),
                5,
            )
            .with_payload(true),
        )
        .await
        .unwrap();

    assert!(!results.result.is_empty());
    let first = &results.result[0];
    let stored_activity_id = first
        .payload
        .get("activity_id")
        .and_then(|v| v.as_str())
        .map_or("", |v| v);
    assert_eq!(stored_activity_id, "act-test");

    client.delete_collection(&col_name).await.unwrap();
}

#[tokio::test]
async fn test_goal_upsert() {
    let client = qdrant_client().await;
    let dim = 8u64;
    let col_name = format!("test_goals_{}", uuid::Uuid::new_v4());
    ensure_clean_collection(&client, &col_name, dim).await;

    let goal = fake_goal("monitor", dim as usize);

    let payload: qdrant_client::Payload = serde_json::json!({
        "name": goal.name,
        "description": goal.description,
        "version": goal.version,
    })
    .try_into()
    .unwrap();

    let point =
        qdrant_client::qdrant::PointStruct::new(goal.id.clone(), goal.embedding.clone(), payload);

    client
        .upsert_points(qdrant_client::qdrant::UpsertPointsBuilder::new(
            &col_name,
            vec![point.clone()],
        ))
        .await
        .unwrap();

    // Upsert same ID again (update)
    let updated_payload: qdrant_client::Payload = serde_json::json!({
        "name": goal.name,
        "description": "updated description",
        "version": 2,
    })
    .try_into()
    .unwrap();

    let updated_point = qdrant_client::qdrant::PointStruct::new(
        goal.id.clone(),
        goal.embedding.clone(),
        updated_payload,
    );

    client
        .upsert_points(qdrant_client::qdrant::UpsertPointsBuilder::new(
            &col_name,
            vec![updated_point],
        ))
        .await
        .unwrap();

    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let results = client
        .search_points(
            qdrant_client::qdrant::SearchPointsBuilder::new(&col_name, goal.embedding.clone(), 5)
                .with_payload(true),
        )
        .await
        .unwrap();

    assert_eq!(results.result.len(), 1, "upsert should not duplicate");
    let desc = results.result[0]
        .payload
        .get("description")
        .and_then(|v| v.as_str())
        .map_or("", |v| v);
    assert_eq!(desc, "updated description");

    client.delete_collection(&col_name).await.unwrap();
}

#[tokio::test]
async fn test_search_empty_collection() {
    let client = qdrant_client().await;
    let dim = 8u64;
    let col_name = format!("test_empty_{}", uuid::Uuid::new_v4());
    ensure_clean_collection(&client, &col_name, dim).await;

    let query = vec![0.5f32; dim as usize];
    let results = client
        .search_points(
            qdrant_client::qdrant::SearchPointsBuilder::new(&col_name, query, 5).with_payload(true),
        )
        .await
        .unwrap();

    assert!(results.result.is_empty());
    client.delete_collection(&col_name).await.unwrap();
}
