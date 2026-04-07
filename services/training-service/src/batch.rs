use anyhow::Result;
use sqlx::{PgPool, Row};
use tracing::info;

use expert_types::training::{
    Label, LabelSource, TrainingBatch, TrainingBatchRequest, TrainingExample,
};

/// Select a balanced batch of training examples from the database.
///
/// Selects equal numbers of positive and negative examples (up to batch_size per
/// class), ordered by confidence descending. Marks selected rows as used_in_batch.
pub async fn select_batch(pool: &PgPool, req: &TrainingBatchRequest) -> Result<TrainingBatch> {
    let half = req.batch_size;

    let positives = select_by_label(
        pool,
        "positive",
        half,
        &req.domain,
        &req.goal_id,
        req.min_confidence,
    )
    .await?;
    let negatives = select_by_label(
        pool,
        "negative",
        half,
        &req.domain,
        &req.goal_id,
        req.min_confidence,
    )
    .await?;

    let positive_count = positives.len();
    let negative_count = negatives.len();

    let mut ids: Vec<String> = Vec::new();
    ids.extend(positives.iter().map(|e| e.id.clone()));
    ids.extend(negatives.iter().map(|e| e.id.clone()));

    if !ids.is_empty() {
        mark_used(pool, &ids).await?;
    }

    let mut examples = positives;
    examples.extend(negatives);

    info!(
        request_id = %req.request_id,
        positive_count,
        negative_count,
        "batch selected"
    );

    Ok(TrainingBatch {
        request_id: req.request_id.clone(),
        examples,
        positive_count,
        negative_count,
    })
}

async fn select_by_label(
    pool: &PgPool,
    label_str: &str,
    limit: usize,
    domain: &Option<String>,
    goal_id: &Option<String>,
    min_confidence: f32,
) -> Result<Vec<TrainingExample>> {
    let label_json = format!("\"{}\"", label_str);

    let rows = sqlx::query(
        r#"
        SELECT id, activity_id, stream_id, domain, goal_id, goal_version,
               goal_embedding, event_window, window_vectors,
               label, label_source, label_weight, reason,
               created_at, used_in_batch, confidence, consensus_count
        FROM training_examples
        WHERE used_in_batch = false
          AND label = $1
          AND confidence >= $2
          AND ($3::text IS NULL OR domain = $3)
          AND ($4::text IS NULL OR goal_id = $4)
        ORDER BY confidence DESC
        LIMIT $5
        "#,
    )
    .bind(&label_json)
    .bind(min_confidence)
    .bind(domain.as_deref())
    .bind(goal_id.as_deref())
    .bind(limit as i64)
    .fetch_all(pool)
    .await?;

    let examples: Vec<TrainingExample> = rows
        .into_iter()
        .map(|row| {
            let goal_emb: serde_json::Value = row.get("goal_embedding");
            let window: serde_json::Value = row.get("event_window");
            let vectors: serde_json::Value = row.get("window_vectors");
            let label_s: String = row.get("label");
            let source_s: String = row.get("label_source");

            TrainingExample {
                id: row.get("id"),
                activity_id: row.get("activity_id"),
                stream_id: row.get("stream_id"),
                domain: row.get("domain"),
                goal_id: row.get("goal_id"),
                goal_version: row.get::<i32, _>("goal_version") as u32,
                goal_embedding: serde_json::from_value(goal_emb).unwrap_or_default(),
                event_window: serde_json::from_value(window).unwrap_or_default(),
                window_vectors: serde_json::from_value(vectors).unwrap_or_default(),
                label: serde_json::from_str(&label_s).unwrap_or(Label::Positive),
                label_source: serde_json::from_str(&source_s).unwrap_or(LabelSource::Synthetic),
                label_weight: row.get("label_weight"),
                reason: row.get("reason"),
                created_at: row.get::<i64, _>("created_at") as u64,
                used_in_batch: row.get("used_in_batch"),
                confidence: row.get("confidence"),
                consensus_count: row.get::<i32, _>("consensus_count") as u32,
            }
        })
        .collect();

    Ok(examples)
}

async fn mark_used(pool: &PgPool, ids: &[String]) -> Result<()> {
    let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
    sqlx::query(
        r#"
        UPDATE training_examples
        SET used_in_batch = true
        WHERE id = ANY($1::text[])
        "#,
    )
    .bind(&id_refs)
    .execute(pool)
    .await?;

    Ok(())
}
