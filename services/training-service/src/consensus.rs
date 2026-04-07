use anyhow::Result;
use sqlx::{PgPool, Row};
use tracing::{info, warn};

struct LabelRow {
    id: String,
    label: String,
}

/// Run consensus scoring after inserting a new training example.
///
/// Looks for other examples with the same goal_id + goal_version + created_at
/// within the configured time window. If they agree on label polarity and
/// meet the count threshold, all matching rows get updated confidence and
/// consensus_count.
pub async fn run_consensus(
    pool: &PgPool,
    goal_id: &str,
    goal_version: i32,
    created_at: i64,
    _label: &str,
    _label_source: &str,
    threshold: u32,
    time_window_ms: u64,
) -> Result<()> {
    let window_start = created_at.saturating_sub(time_window_ms as i64);
    let window_end = created_at.saturating_add(time_window_ms as i64);

    let rows = sqlx::query(
        r#"
        SELECT id, label
        FROM training_examples
        WHERE goal_id = $1
          AND goal_version = $2
          AND created_at >= $3
          AND created_at <= $4
        "#,
    )
    .bind(goal_id)
    .bind(goal_version)
    .bind(window_start)
    .bind(window_end)
    .fetch_all(pool)
    .await?;

    let label_rows: Vec<LabelRow> = rows
        .iter()
        .map(|r| LabelRow {
            id: r.get("id"),
            label: r.get("label"),
        })
        .collect();

    if label_rows.len() < 2 {
        return Ok(());
    }

    let positive_count = label_rows.iter().filter(|r| is_positive(&r.label)).count();
    let negative_count = label_rows.len() - positive_count;

    let has_disagreement = positive_count > 0 && negative_count > 0;
    if has_disagreement {
        warn!(
            goal_id,
            goal_version,
            positive_count,
            negative_count,
            "consensus: disagreeing labels detected, skipping update"
        );
        return Ok(());
    }

    let agreeing_count = label_rows.len();

    if agreeing_count >= threshold as usize {
        let confidence = (0.5 + 0.15 * agreeing_count as f32).min(1.0);

        let agreeing_ids: Vec<&str> = label_rows.iter().map(|r| r.id.as_str()).collect();

        for batch in agreeing_ids.chunks(100) {
            let ids: Vec<&str> = batch.to_vec();
            sqlx::query(
                r#"
                UPDATE training_examples
                SET consensus_count = $1, confidence = $2
                WHERE id = ANY($3::text[])
                "#,
            )
            .bind(agreeing_count as i32)
            .bind(confidence)
            .bind(&ids)
            .execute(pool)
            .await?;
        }

        info!(
            goal_id,
            goal_version,
            agreeing_count,
            confidence,
            "consensus: updated confidence for agreeing labels"
        );
    }

    Ok(())
}

fn is_positive(label: &str) -> bool {
    label.contains("positive") || label.contains("Positive")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_positive() {
        assert!(is_positive("\"positive\""));
        assert!(!is_positive("\"negative\""));
    }
}
