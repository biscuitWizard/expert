use anyhow::{Result, bail};
use ndarray::{Array1, Array2};
use tracing::info;

use expert_ssm::checkpoint::SsmCheckpoint;
use expert_types::training::{Label, TrainingExample};

/// Training hyperparameters.
pub struct TrainConfig {
    pub epochs: usize,
    pub learning_rate: f32,
    pub gradient_clip: f32,
}

/// Result of a training run.
pub struct TrainResult {
    pub checkpoint: SsmCheckpoint,
    pub final_loss: f32,
    pub examples_seen: usize,
}

/// Run an offline training loop over a batch of labeled examples.
///
/// Computes forward passes through the SSM recurrence and applies gradient
/// updates to C and D (output matrices). The recurrence matrices A, B, and
/// projection are frozen in this simple training loop to maintain stability.
///
/// `goal_ids` maps row index -> goal ID for the checkpoint's output matrix.
/// For full BPTT through A/B, integrate candle autodiff in a future iteration.
pub fn run_training(
    batch: &[TrainingExample],
    base_checkpoint: &SsmCheckpoint,
    config: &TrainConfig,
    checkpoint_id: &str,
    goal_ids: &[String],
) -> Result<TrainResult> {
    if batch.is_empty() {
        bail!("empty training batch");
    }

    let projection = Array2::from_shape_vec(
        (base_checkpoint.hidden_dim, base_checkpoint.embedding_dim),
        base_checkpoint.projection.clone(),
    )?;
    let a = Array2::from_shape_vec(
        (base_checkpoint.hidden_dim, base_checkpoint.hidden_dim),
        base_checkpoint.a.clone(),
    )?;
    let input_dim =
        base_checkpoint.hidden_dim + base_checkpoint.max_k + base_checkpoint.num_scalar_features;
    let b = Array2::from_shape_vec(
        (base_checkpoint.hidden_dim, input_dim),
        base_checkpoint.b.clone(),
    )?;
    let mut c = Array2::from_shape_vec(
        (base_checkpoint.max_k, base_checkpoint.hidden_dim),
        base_checkpoint.c.clone(),
    )?;
    let mut d = Array1::from_vec(base_checkpoint.d.clone());

    let mut total_loss = 0.0f32;
    let mut count = 0usize;

    for epoch in 0..config.epochs {
        let mut epoch_loss = 0.0f32;

        for example in batch {
            let target = match example.label {
                Label::Positive => 1.0f32,
                Label::Negative => 0.0f32,
            };

            let (score, h_final) =
                forward_pass(&projection, &a, &b, &c, &d, example, base_checkpoint);

            let loss = binary_cross_entropy(score, target);
            epoch_loss += loss;
            count += 1;

            let grad_scale = (score - target) * config.learning_rate * example.label_weight;
            let goal_idx = goal_index(&example.goal_id, goal_ids, base_checkpoint.max_k);

            for j in 0..base_checkpoint.hidden_dim {
                let grad_c = grad_scale * h_final[j];
                let clipped = clip_grad(grad_c, config.gradient_clip);
                c[[goal_idx, j]] -= clipped;
            }

            let grad_d = grad_scale;
            let clipped = clip_grad(grad_d, config.gradient_clip);
            d[goal_idx] -= clipped;
        }

        total_loss = epoch_loss / batch.len() as f32;
        info!(epoch, loss = total_loss, "training epoch complete");
    }

    let ckpt = SsmCheckpoint {
        id: checkpoint_id.to_string(),
        format_version: SsmCheckpoint::current_format_version(),
        embedding_dim: base_checkpoint.embedding_dim,
        hidden_dim: base_checkpoint.hidden_dim,
        max_k: base_checkpoint.max_k,
        num_scalar_features: base_checkpoint.num_scalar_features,
        projection: projection.iter().copied().collect(),
        a: a.iter().copied().collect(),
        b: b.iter().copied().collect(),
        c: c.iter().copied().collect(),
        d: d.iter().copied().collect(),
        created_at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64,
        domain: base_checkpoint.domain.clone(),
        training_batch_id: None,
        timescale: None,
    };

    Ok(TrainResult {
        checkpoint: ckpt,
        final_loss: total_loss,
        examples_seen: count,
    })
}

fn forward_pass(
    projection: &Array2<f32>,
    a: &Array2<f32>,
    b: &Array2<f32>,
    c: &Array2<f32>,
    d: &Array1<f32>,
    example: &TrainingExample,
    base: &SsmCheckpoint,
) -> (f32, Array1<f32>) {
    let hidden_dim = base.hidden_dim;
    let max_k = base.max_k;
    let mut h = Array1::<f32>::zeros(hidden_dim);

    for (i, _event) in example.event_window.iter().enumerate() {
        let embedding = if i < example.window_vectors.len() {
            &example.window_vectors[i]
        } else {
            continue;
        };

        let emb_arr = Array1::from_vec(embedding.clone());
        let projected = projection.dot(&emb_arr);

        let mut x = Array1::<f32>::zeros(b.dim().1);
        for j in 0..hidden_dim.min(x.len()) {
            x[j] = projected[j];
        }

        // Fill cosine similarity slots with zeros (no live goal context in training replay)
        // and scalar feature slots with zeros.

        h = a.dot(&h) + b.dot(&x);
    }

    let goal_idx = 0.min(max_k.saturating_sub(1));
    let raw_score = c.row(goal_idx).dot(&h) + d[goal_idx];
    let score = sigmoid(raw_score);

    (score, h)
}

fn goal_index(goal_id: &str, goal_ids: &[String], max_k: usize) -> usize {
    goal_ids
        .iter()
        .position(|id| id == goal_id)
        .unwrap_or(0)
        .min(max_k.saturating_sub(1))
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn binary_cross_entropy(predicted: f32, target: f32) -> f32 {
    let p = predicted.clamp(1e-7, 1.0 - 1e-7);
    -(target * p.ln() + (1.0 - target) * (1.0 - p).ln())
}

fn clip_grad(grad: f32, max_norm: f32) -> f32 {
    grad.clamp(-max_norm, max_norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_binary_cross_entropy() {
        let loss_correct = binary_cross_entropy(0.9, 1.0);
        let loss_wrong = binary_cross_entropy(0.1, 1.0);
        assert!(loss_correct < loss_wrong);
    }

    #[test]
    fn test_clip_grad() {
        assert_eq!(clip_grad(2.0, 1.0), 1.0);
        assert_eq!(clip_grad(-2.0, 1.0), -1.0);
        assert_eq!(clip_grad(0.5, 1.0), 0.5);
    }
}
