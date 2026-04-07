use ndarray::{Array1, ArrayView1};
use expert_vectors::{cosine_similarity, ema_update, mahalanobis_diagonal};

/// Per-activity feature state maintained across events.
pub struct FeatureState {
    /// Per-goal EMA of cosine similarity.
    pub ema_relevance: Vec<f32>,
    /// Running stream centroid (embedding space).
    pub centroid: Array1<f32>,
    /// Diagonal inverse variance for Mahalanobis (per-dimension).
    pub inv_var: Array1<f32>,
    /// Running variance accumulator (Welford M2).
    pub var_accum: Array1<f32>,
    /// Previous centroid for drift computation.
    pub prev_centroid: Array1<f32>,
    /// Timestamp of last event (for delta_t).
    pub last_event_ts: u64,
    /// Per-goal last-seen-relevant timestamp (for silence).
    pub last_relevant_ts: Vec<u64>,
    /// Number of events processed (for centroid/variance updates).
    pub event_count: u64,
    /// EMA smoothing factor.
    pub ema_alpha: f32,
    /// Relevance threshold for considering an event "relevant" to a goal.
    pub relevance_threshold: f32,
}

impl FeatureState {
    pub fn new(embedding_dim: usize, k: usize) -> Self {
        Self {
            ema_relevance: vec![0.0; k],
            centroid: Array1::zeros(embedding_dim),
            inv_var: Array1::ones(embedding_dim),
            var_accum: Array1::zeros(embedding_dim),
            prev_centroid: Array1::zeros(embedding_dim),
            last_event_ts: 0,
            last_relevant_ts: vec![0; k],
            event_count: 0,
            ema_alpha: 0.1,
            relevance_threshold: 0.3,
        }
    }

    pub fn resize_goals(&mut self, k: usize) {
        self.ema_relevance.resize(k, 0.0);
        self.last_relevant_ts.resize(k, 0);
    }
}

/// Compute the feature vector for one event against the goal matrix.
/// Returns the concatenated feature vector: [projected_embedding, per_goal_cosines, scalars]
pub fn compute_features(
    embedding: ArrayView1<f32>,
    goal_embeddings: &[ArrayView1<f32>],
    feature_state: &mut FeatureState,
    timestamp: u64,
    projected_embedding: &Array1<f32>,
) -> Vec<f32> {
    let k = goal_embeddings.len();
    feature_state.resize_goals(k);
    feature_state.event_count += 1;

    // Per-goal cosine similarities
    let mut cosines = Vec::with_capacity(k);
    for (i, goal_emb) in goal_embeddings.iter().enumerate() {
        let cos = cosine_similarity(embedding, *goal_emb);
        cosines.push(cos);

        // Update EMA relevance
        feature_state.ema_relevance[i] =
            ema_update(feature_state.ema_relevance[i], cos, feature_state.ema_alpha);

        // Track last relevant timestamp
        if cos > feature_state.relevance_threshold {
            feature_state.last_relevant_ts[i] = timestamp;
        }
    }

    // Update stream centroid (Welford online mean)
    let old_centroid = feature_state.centroid.clone();
    expert_vectors::centroid_update(
        &mut feature_state.centroid,
        embedding,
        feature_state.event_count,
    );

    // Update variance accumulator
    expert_vectors::variance_update(
        &feature_state.centroid,
        &old_centroid,
        &mut feature_state.var_accum,
        embedding,
    );

    // Update inverse variance
    if feature_state.event_count >= 2 {
        feature_state.inv_var =
            expert_vectors::inverse_variance(feature_state.var_accum.view(), feature_state.event_count);
    }

    // Drift: distance between current and previous centroid
    let drift = if feature_state.event_count > 1 {
        let diff = &feature_state.centroid - &feature_state.prev_centroid;
        diff.dot(&diff).sqrt()
    } else {
        0.0
    };
    feature_state.prev_centroid = feature_state.centroid.clone();

    // Mahalanobis surprise
    let surprise = mahalanobis_diagonal(
        embedding,
        feature_state.centroid.view(),
        feature_state.inv_var.view(),
    );

    // Delta_t: time since last event (normalized to seconds)
    let delta_t = if feature_state.last_event_ts > 0 {
        (timestamp - feature_state.last_event_ts) as f32 / 1000.0
    } else {
        0.0
    };
    feature_state.last_event_ts = timestamp;

    // Per-goal silence: seconds since last relevant event
    let silences: Vec<f32> = feature_state
        .last_relevant_ts
        .iter()
        .map(|&ts| {
            if ts > 0 {
                (timestamp - ts) as f32 / 1000.0
            } else {
                0.0
            }
        })
        .collect();

    // Concatenate: [projected_embedding, per_goal_cosines, ema_per_goal, drift, surprise, delta_t, silences...]
    let num_scalars = 3 + k; // drift + surprise + delta_t + per-goal silence
    let mut features =
        Vec::with_capacity(projected_embedding.len() + k + num_scalars);

    features.extend(projected_embedding.iter());
    features.extend(&cosines);
    features.push(drift);
    features.push(surprise.min(10.0)); // clamp outliers
    features.push(delta_t.min(60.0));  // clamp to 60s max
    features.extend(&silences);

    features
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn make_embedding(dim: usize, val: f32) -> Array1<f32> {
        Array1::from_vec(vec![val; dim])
    }

    #[test]
    fn test_feature_vector_length() {
        let dim = 8;
        let hidden_dim = 4;
        let k = 2;
        let emb = make_embedding(dim, 0.5);
        let goal1 = make_embedding(dim, 1.0);
        let goal2 = make_embedding(dim, 0.0);
        let goals: Vec<ArrayView1<f32>> = vec![goal1.view(), goal2.view()];
        let projected = make_embedding(hidden_dim, 0.3);
        let mut state = FeatureState::new(dim, k);

        let features = compute_features(emb.view(), &goals, &mut state, 1000, &projected);
        // expected: hidden_dim + k (cosines) + 3 (drift, surprise, delta_t) + k (silences)
        let expected_len = hidden_dim + k + 3 + k;
        assert_eq!(features.len(), expected_len);
    }

    #[test]
    fn test_cosines_in_range() {
        let dim = 8;
        let k = 2;
        let emb = make_embedding(dim, 0.5);
        let goal = make_embedding(dim, 1.0);
        let goals = vec![goal.view(), goal.view()];
        let projected = make_embedding(4, 0.3);
        let mut state = FeatureState::new(dim, k);

        let features = compute_features(emb.view(), &goals, &mut state, 1000, &projected);
        // Cosines start at index hidden_dim=4
        let eps = 1e-5;
        for i in 4..4 + k {
            let cos = features[i];
            assert!(cos >= -1.0 - eps && cos <= 1.0 + eps, "cosine {cos} out of range");
        }
    }

    #[test]
    fn test_delta_t_computation() {
        let dim = 4;
        let k = 1;
        let emb = make_embedding(dim, 0.5);
        let goal = make_embedding(dim, 1.0);
        let goals = vec![goal.view()];
        let projected = make_embedding(2, 0.3);
        let mut state = FeatureState::new(dim, k);

        // First event: delta_t should be 0
        let f1 = compute_features(emb.view(), &goals, &mut state, 1000, &projected);
        let delta_t_idx = 2 + k + 2; // projected(2) + cosines(1) + drift + surprise = idx 4
        let delta_t_1 = f1[delta_t_idx];
        assert_eq!(delta_t_1, 0.0, "first event delta_t should be 0");

        // Second event 2500ms later: delta_t should be 2.5
        let f2 = compute_features(emb.view(), &goals, &mut state, 3500, &projected);
        let delta_t_2 = f2[delta_t_idx];
        assert!((delta_t_2 - 2.5).abs() < 0.01, "delta_t should be 2.5, got {delta_t_2}");
    }

    #[test]
    fn test_ema_moves_toward_value() {
        let dim = 4;
        let k = 1;
        let emb = make_embedding(dim, 1.0);
        let goal = make_embedding(dim, 1.0);
        let goals = vec![goal.view()];
        let projected = make_embedding(2, 0.3);
        let mut state = FeatureState::new(dim, k);

        compute_features(emb.view(), &goals, &mut state, 1000, &projected);
        let ema_after_one = state.ema_relevance[0];
        assert!(ema_after_one > 0.0, "EMA should be positive for identical vectors");

        compute_features(emb.view(), &goals, &mut state, 2000, &projected);
        let ema_after_two = state.ema_relevance[0];
        assert!(ema_after_two > ema_after_one, "EMA should increase with repeated relevant events");
    }

    #[test]
    fn test_resize_goals() {
        let mut state = FeatureState::new(4, 2);
        assert_eq!(state.ema_relevance.len(), 2);
        assert_eq!(state.last_relevant_ts.len(), 2);

        state.resize_goals(5);
        assert_eq!(state.ema_relevance.len(), 5);
        assert_eq!(state.last_relevant_ts.len(), 5);

        state.resize_goals(1);
        assert_eq!(state.ema_relevance.len(), 1);
        assert_eq!(state.last_relevant_ts.len(), 1);
    }

    #[test]
    fn test_event_count_increments() {
        let dim = 4;
        let k = 1;
        let emb = make_embedding(dim, 0.5);
        let goal = make_embedding(dim, 1.0);
        let goals = vec![goal.view()];
        let projected = make_embedding(2, 0.3);
        let mut state = FeatureState::new(dim, k);

        assert_eq!(state.event_count, 0);
        compute_features(emb.view(), &goals, &mut state, 1000, &projected);
        assert_eq!(state.event_count, 1);
        compute_features(emb.view(), &goals, &mut state, 2000, &projected);
        assert_eq!(state.event_count, 2);
    }
}
