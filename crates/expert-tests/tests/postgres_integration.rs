use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;
use expert_tests::*;

fn database_url() -> String {
    std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://expert:expert_dev@127.0.0.1:5432/expert_training".to_string())
}

async fn pool() -> PgPool {
    PgPoolOptions::new()
        .max_connections(2)
        .connect(&database_url())
        .await
        .expect("failed to connect to PostgreSQL")
}

async fn run_migrations(pool: &PgPool) {
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS training_examples (
            id TEXT PRIMARY KEY,
            activity_id TEXT NOT NULL,
            stream_id TEXT NOT NULL,
            domain TEXT,
            goal_id TEXT NOT NULL,
            goal_version INTEGER NOT NULL,
            goal_embedding JSONB NOT NULL,
            event_window JSONB NOT NULL,
            window_vectors JSONB NOT NULL,
            label TEXT NOT NULL,
            label_source TEXT NOT NULL,
            label_weight REAL NOT NULL,
            reason TEXT NOT NULL,
            created_at BIGINT NOT NULL,
            used_in_batch BOOLEAN NOT NULL DEFAULT FALSE,
            confidence REAL NOT NULL,
            consensus_count INTEGER NOT NULL DEFAULT 1
        )
        "#,
    )
    .execute(pool)
    .await
    .expect("migration failed");

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_training_domain_label ON training_examples (domain, label, used_in_batch)",
    )
    .execute(pool)
    .await
    .expect("index creation failed");

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_training_goal ON training_examples (goal_id, goal_version)",
    )
    .execute(pool)
    .await
    .expect("index creation failed");
}

async fn cleanup(pool: &PgPool) {
    sqlx::query("DELETE FROM training_examples WHERE domain = 'test'")
        .execute(pool)
        .await
        .ok();
}

async fn insert_example(pool: &PgPool, ex: &expert_types::training::TrainingExample) -> Result<(), sqlx::Error> {
    let label_str = serde_json::to_string(&ex.label).unwrap();
    let source_str = serde_json::to_string(&ex.label_source).unwrap();
    let goal_emb_json = serde_json::to_value(&ex.goal_embedding).unwrap();
    let window_json = serde_json::to_value(&ex.event_window).unwrap();
    let vectors_json = serde_json::to_value(&ex.window_vectors).unwrap();

    sqlx::query(
        r#"
        INSERT INTO training_examples (
            id, activity_id, stream_id, domain, goal_id, goal_version,
            goal_embedding, event_window, window_vectors,
            label, label_source, label_weight, reason,
            created_at, used_in_batch, confidence, consensus_count
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
        ON CONFLICT (id) DO NOTHING
        "#,
    )
    .bind(&ex.id)
    .bind(&ex.activity_id)
    .bind(&ex.stream_id)
    .bind(&ex.domain)
    .bind(&ex.goal_id)
    .bind(ex.goal_version as i32)
    .bind(&goal_emb_json)
    .bind(&window_json)
    .bind(&vectors_json)
    .bind(&label_str)
    .bind(&source_str)
    .bind(ex.label_weight)
    .bind(&ex.reason)
    .bind(ex.created_at as i64)
    .bind(ex.used_in_batch)
    .bind(ex.confidence)
    .bind(ex.consensus_count as i32)
    .execute(pool)
    .await?;

    Ok(())
}

#[tokio::test]
async fn test_migrations_idempotent() {
    let pool = pool().await;
    run_migrations(&pool).await;
    run_migrations(&pool).await;
}

#[tokio::test]
async fn test_label_insert() {
    let pool = pool().await;
    run_migrations(&pool).await;
    cleanup(&pool).await;

    let example = fake_training_example();
    insert_example(&pool, &example).await.unwrap();

    let row: (String, String) = sqlx::query_as(
        "SELECT id, reason FROM training_examples WHERE id = $1",
    )
    .bind(&example.id)
    .fetch_one(&pool)
    .await
    .unwrap();

    assert_eq!(row.0, example.id);
    assert_eq!(row.1, "test reason");

    cleanup(&pool).await;
}

#[tokio::test]
async fn test_duplicate_id_ignored() {
    let pool = pool().await;
    run_migrations(&pool).await;
    cleanup(&pool).await;

    let example = fake_training_example();
    insert_example(&pool, &example).await.unwrap();
    insert_example(&pool, &example).await.unwrap(); // should not error

    let count: (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM training_examples WHERE id = $1",
    )
    .bind(&example.id)
    .fetch_one(&pool)
    .await
    .unwrap();

    assert_eq!(count.0, 1, "duplicate insert should be ignored");

    cleanup(&pool).await;
}

#[tokio::test]
async fn test_query_by_domain_and_label() {
    let pool = pool().await;
    run_migrations(&pool).await;
    cleanup(&pool).await;

    let mut ex1 = fake_training_example();
    ex1.id = uuid::Uuid::new_v4().to_string();

    let mut ex2 = fake_training_example();
    ex2.id = uuid::Uuid::new_v4().to_string();
    ex2.label = expert_types::training::Label::Negative;

    insert_example(&pool, &ex1).await.unwrap();
    insert_example(&pool, &ex2).await.unwrap();

    let positives: (i64,) = sqlx::query_as(
        "SELECT COUNT(*) FROM training_examples WHERE domain = 'test' AND label = '\"positive\"'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();

    assert_eq!(positives.0, 1);

    cleanup(&pool).await;
}
