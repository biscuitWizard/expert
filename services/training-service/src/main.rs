mod batch;
mod consensus;
mod train;

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

use expert_config::Config;
use expert_redis::names;
use expert_redis::{StreamConsumer, StreamProducer};
use expert_ssm::ssm::LinearSsm;
use expert_types::signals::CheckpointAvailable;
use expert_types::training::{TrainingBatchRequest, TrainingExample};

struct TrainingState {
    labels_since_last_slow: u64,
    labels_since_last_medium: u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    expert_config::init_tracing();
    let config = Config::from_env();
    info!("starting training-service");

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&config.postgres_url)
        .await?;

    run_migrations(&pool).await?;
    info!("database migrations complete");

    let conn = expert_redis::connect(&config.redis_url).await?;
    let producer = StreamProducer::new(conn.clone(), config.stream_maxlen);

    let state = Arc::new(RwLock::new(TrainingState {
        labels_since_last_slow: 0,
        labels_since_last_medium: 0,
    }));

    // Ensure checkpoint directory exists
    std::fs::create_dir_all(&config.checkpoint_dir).ok();

    // Spawn batch request consumer
    {
        let pool = pool.clone();
        let conn = conn.clone();
        let mut producer = producer.clone();
        tokio::spawn(async move {
            if let Err(e) = run_batch_consumer(conn, pool, &mut producer).await {
                error!(error = %e, "batch consumer exited");
            }
        });
    }

    // Spawn SLOW training background task
    {
        let pool = pool.clone();
        let state = state.clone();
        let config = config.clone();
        let mut producer = producer.clone();
        tokio::spawn(async move {
            run_slow_training_loop(pool, state, &config, &mut producer).await;
        });
    }

    // Spawn MEDIUM few-shot background task
    {
        let pool = pool.clone();
        let state = state.clone();
        let config = config.clone();
        let mut producer = producer.clone();
        tokio::spawn(async move {
            run_medium_fewshot_loop(pool, state, &config, &mut producer).await;
        });
    }

    // Main label consumer
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

                        let label_str = serde_json::to_string(&example.label).unwrap_or_default();
                        let source_str =
                            serde_json::to_string(&example.label_source).unwrap_or_default();
                        if let Err(e) = consensus::run_consensus(
                            &pool,
                            &example.goal_id,
                            example.goal_version as i32,
                            example.created_at as i64,
                            &label_str,
                            &source_str,
                            config.consensus_threshold,
                            config.consensus_time_window_ms,
                        )
                        .await
                        {
                            warn!(error = %e, "consensus scoring failed");
                        }

                        {
                            let mut s = state.write().await;
                            s.labels_since_last_slow += 1;
                            s.labels_since_last_medium += 1;
                        }
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

async fn run_batch_consumer(
    conn: redis::aio::MultiplexedConnection,
    pool: PgPool,
    producer: &mut StreamProducer,
) -> Result<()> {
    let mut consumer = StreamConsumer::new(
        conn,
        names::REQUESTS_TRAINING_BATCH.to_string(),
        "training-batch".to_string(),
        "batch-0".to_string(),
        500,
    )
    .await?;

    loop {
        match consumer.consume::<TrainingBatchRequest>().await {
            Ok(Some((id, req))) => {
                let _ = consumer.ack(&id).await;
                match batch::select_batch(&pool, &req).await {
                    Ok(result) => {
                        if let Err(e) = producer
                            .publish(names::RESULTS_TRAINING_BATCH, &result)
                            .await
                        {
                            error!(error = %e, "failed to publish training batch result");
                        }
                    }
                    Err(e) => {
                        error!(error = %e, request_id = %req.request_id, "batch selection failed");
                    }
                }
            }
            Ok(None) => {}
            Err(e) => {
                warn!(error = %e, "batch consumer error");
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        }
    }
}

async fn run_slow_training_loop(
    pool: PgPool,
    state: Arc<RwLock<TrainingState>>,
    config: &Config,
    producer: &mut StreamProducer,
) {
    loop {
        tokio::time::sleep(Duration::from_secs(config.training_interval_secs)).await;

        let should_train = {
            let s = state.read().await;
            s.labels_since_last_slow >= config.training_label_threshold
        };

        if !should_train {
            continue;
        }

        info!("SLOW training: starting offline retraining");

        let req = TrainingBatchRequest {
            request_id: uuid::Uuid::new_v4().to_string(),
            domain: None,
            goal_id: None,
            batch_size: config.training_batch_size,
            min_confidence: 0.3,
        };

        let batch_result = match batch::select_batch(&pool, &req).await {
            Ok(b) => b,
            Err(e) => {
                warn!(error = %e, "SLOW training: batch selection failed");
                continue;
            }
        };

        if batch_result.examples.is_empty() {
            info!("SLOW training: no examples available, skipping");
            continue;
        }

        let base_ssm = LinearSsm::new(
            config.embedding_dim,
            config.ssm_hidden_dim,
            config.ssm_max_k,
            5,
        );
        let base_ckpt = base_ssm.save_checkpoint("base");

        let checkpoint_id = uuid::Uuid::new_v4().to_string();
        let train_config = train::TrainConfig {
            epochs: config.training_epochs,
            learning_rate: config.training_learning_rate,
            gradient_clip: config.training_gradient_clip,
        };

        match train::run_training(
            &batch_result.examples,
            &base_ckpt,
            &train_config,
            &checkpoint_id,
            &[],
        ) {
            Ok(result) => {
                let path = format!("{}/{}.json", config.checkpoint_dir, checkpoint_id);
                if let Ok(json) = serde_json::to_string_pretty(&result.checkpoint) {
                    if let Err(e) = std::fs::write(&path, &json) {
                        error!(error = %e, "failed to write checkpoint file");
                        continue;
                    }
                }

                let notification = CheckpointAvailable {
                    checkpoint_id: checkpoint_id.clone(),
                    domain: result.checkpoint.domain.clone(),
                    path: path.clone(),
                    created_at: result.checkpoint.created_at,
                    timescale: Some("slow".to_string()),
                };

                if let Err(e) = producer
                    .publish(names::CHECKPOINTS_AVAILABLE, &notification)
                    .await
                {
                    error!(error = %e, "failed to publish checkpoint notification");
                } else {
                    info!(
                        checkpoint_id,
                        loss = result.final_loss,
                        examples = result.examples_seen,
                        "SLOW training: checkpoint saved and published"
                    );
                    let mut s = state.write().await;
                    s.labels_since_last_slow = 0;
                }
            }
            Err(e) => {
                error!(error = %e, "SLOW training: training failed");
            }
        }
    }
}

async fn run_medium_fewshot_loop(
    pool: PgPool,
    state: Arc<RwLock<TrainingState>>,
    config: &Config,
    producer: &mut StreamProducer,
) {
    loop {
        tokio::time::sleep(Duration::from_secs(30)).await;

        let should_train = {
            let s = state.read().await;
            s.labels_since_last_medium >= config.medium_label_threshold
        };

        if !should_train {
            continue;
        }

        info!("MEDIUM training: starting few-shot adaptation");

        let req = TrainingBatchRequest {
            request_id: uuid::Uuid::new_v4().to_string(),
            domain: None,
            goal_id: None,
            batch_size: config.medium_batch_size,
            min_confidence: 0.5,
        };

        let batch_result = match batch::select_batch(&pool, &req).await {
            Ok(b) => b,
            Err(e) => {
                warn!(error = %e, "MEDIUM training: batch selection failed");
                continue;
            }
        };

        if batch_result.examples.is_empty() {
            continue;
        }

        let base_ssm = LinearSsm::new(
            config.embedding_dim,
            config.ssm_hidden_dim,
            config.ssm_max_k,
            5,
        );
        let base_ckpt = base_ssm.save_checkpoint("base");

        let checkpoint_id = uuid::Uuid::new_v4().to_string();
        let train_config = train::TrainConfig {
            epochs: 1,
            learning_rate: config.medium_learning_rate,
            gradient_clip: config.training_gradient_clip,
        };

        match train::run_training(
            &batch_result.examples,
            &base_ckpt,
            &train_config,
            &checkpoint_id,
            &[],
        ) {
            Ok(result) => {
                let path = format!("{}/{}.json", config.checkpoint_dir, checkpoint_id);
                if let Ok(json) = serde_json::to_string_pretty(&result.checkpoint) {
                    if let Err(e) = std::fs::write(&path, &json) {
                        error!(error = %e, "failed to write medium checkpoint");
                        continue;
                    }
                }

                let notification = CheckpointAvailable {
                    checkpoint_id: checkpoint_id.clone(),
                    domain: result.checkpoint.domain.clone(),
                    path: path.clone(),
                    created_at: result.checkpoint.created_at,
                    timescale: Some("medium".to_string()),
                };

                if let Err(e) = producer
                    .publish(names::CHECKPOINTS_AVAILABLE, &notification)
                    .await
                {
                    error!(error = %e, "failed to publish medium checkpoint");
                } else {
                    info!(
                        checkpoint_id,
                        "MEDIUM training: few-shot checkpoint published"
                    );
                    let mut s = state.write().await;
                    s.labels_since_last_medium = 0;
                }
            }
            Err(e) => {
                error!(error = %e, "MEDIUM training: training failed");
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

    sqlx::query(
        r#"
        CREATE INDEX IF NOT EXISTS idx_training_consensus
        ON training_examples (goal_id, goal_version, created_at)
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
