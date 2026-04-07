use std::time::Duration;

use anyhow::Result;
use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;
use tracing::{error, info, warn};

use expert_config::Config;
use expert_redis::names;
use expert_redis::StreamConsumer;
use expert_types::training::TrainingExample;

#[tokio::main]
async fn main() -> Result<()> {
    expert_config::init_tracing();
    let config = Config::from_env();
    info!("starting training-service");

    // Connect to PostgreSQL
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&config.postgres_url)
        .await?;

    // Run migrations
    run_migrations(&pool).await?;
    info!("database migrations complete");

    let conn = expert_redis::connect(&config.redis_url).await?;

    let mut consumer = StreamConsumer::new(
        conn,
        names::LABELS_WRITE.to_string(),
        "training".to_string(),
        "training-0".to_string(),
        500,
    )
    .await?;

    info!("consuming training labels from labels.write");

    loop {
        match consumer.consume::<TrainingExample>().await {
            Ok(Some((id, example))) => {
                let _ = consumer.ack(&id).await;
                match insert_example(&pool, &example).await {
                    Ok(_) => {
                        info!(
                            example_id = %example.id,
                            label = ?example.label,
                            source = ?example.label_source,
                            "training example stored"
                        );
                    }
                    Err(e) => {
                        error!(error = %e, example_id = %example.id, "failed to store training example");
                    }
                }
            }
            Ok(None) => {}
            Err(e) => {
                warn!(error = %e, "label consumer error");
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }
    }
}

async fn run_migrations(pool: &PgPool) -> Result<()> {
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
    .await?;

    // Index for batch queries
    sqlx::query(
        r#"
        CREATE INDEX IF NOT EXISTS idx_training_domain_label
        ON training_examples (domain, label, used_in_batch)
        "#,
    )
    .execute(pool)
    .await?;

    sqlx::query(
        r#"
        CREATE INDEX IF NOT EXISTS idx_training_goal
        ON training_examples (goal_id, goal_version)
        "#,
    )
    .execute(pool)
    .await?;

    Ok(())
}

async fn insert_example(pool: &PgPool, ex: &TrainingExample) -> Result<()> {
    let label_str = serde_json::to_string(&ex.label)?;
    let source_str = serde_json::to_string(&ex.label_source)?;
    let goal_emb_json = serde_json::to_value(&ex.goal_embedding)?;
    let window_json = serde_json::to_value(&ex.event_window)?;
    let vectors_json = serde_json::to_value(&ex.window_vectors)?;

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
