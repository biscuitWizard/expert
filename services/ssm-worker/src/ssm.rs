use ndarray::{Array1, Array2, ArrayView1};

/// Trait abstracting the SSM recurrence so implementations can be swapped.
pub trait SsmCore: Send + Sync {
    fn update(&mut self, input: &[f32], k: usize) -> Vec<f32>;
    fn state(&self) -> &[f32];
    fn load_state(&mut self, state: &[f32]);
    fn reset(&mut self);
}

/// Minimal linear SSM: h_t = A * h_{t-1} + B * x_t
/// Per-goal scores: sigmoid(C_i . h_t + D_i)
pub struct LinearSsm {
    pub projection: Array2<f32>, // [hidden_dim, embedding_dim]
    pub a: Array2<f32>,          // [hidden_dim, hidden_dim]
    pub b: Array2<f32>,          // [hidden_dim, input_dim]
    pub c: Array2<f32>,          // [max_k, hidden_dim]
    pub d: Array1<f32>,          // [max_k]
    pub h: Array1<f32>,          // [hidden_dim]
    hidden_dim: usize,
    input_dim: usize,
}

impl LinearSsm {
    /// Create a new LinearSsm with default initialization.
    /// `embedding_dim`: raw embedding size (4096 for Qwen3-Embedding-8B)
    /// `hidden_dim`: SSM hidden state dimension (256)
    /// `max_k`: maximum number of goals supported
    /// `num_scalar_features`: additional scalar features beyond cosines
    pub fn new(
        embedding_dim: usize,
        hidden_dim: usize,
        max_k: usize,
        num_scalar_features: usize,
    ) -> Self {
        // input_dim = hidden_dim (projected) + max_k (cosines) + num_scalar_features
        let input_dim = hidden_dim + max_k + num_scalar_features;

        // Projection: [hidden_dim, embedding_dim], small random init
        let projection = Array2::from_shape_fn((hidden_dim, embedding_dim), |(_i, _j)| {
            (rand_f32() - 0.5) * 2.0 / (embedding_dim as f32).sqrt()
        });

        // A: scaled identity for stable recurrence
        let mut a = Array2::zeros((hidden_dim, hidden_dim));
        for i in 0..hidden_dim {
            a[[i, i]] = 0.95; // slow decay
        }

        // B: Xavier init
        let b_scale = (2.0 / (hidden_dim + input_dim) as f32).sqrt();
        let b = Array2::from_shape_fn((hidden_dim, input_dim), |_| {
            (rand_f32() - 0.5) * 2.0 * b_scale
        });

        // C: Xavier init
        let c_scale = (2.0 / (max_k + hidden_dim) as f32).sqrt();
        let c = Array2::from_shape_fn((max_k, hidden_dim), |_| {
            (rand_f32() - 0.5) * 2.0 * c_scale
        });

        // D: zero bias
        let d = Array1::zeros(max_k);

        let h = Array1::zeros(hidden_dim);

        Self {
            projection,
            a,
            b,
            c,
            d,
            h,
            hidden_dim,
            input_dim,
        }
    }

    /// Project a raw embedding from embedding_dim to hidden_dim.
    pub fn project_embedding(&self, embedding: ArrayView1<f32>) -> Array1<f32> {
        self.projection.dot(&embedding)
    }
}

impl SsmCore for LinearSsm {
    fn update(&mut self, input: &[f32], k: usize) -> Vec<f32> {
        let x = ArrayView1::from(input);
        // Pad or truncate input to expected input_dim
        let mut padded = Array1::zeros(self.input_dim);
        let copy_len = input.len().min(self.input_dim);
        padded
            .slice_mut(ndarray::s![..copy_len])
            .assign(&x.slice(ndarray::s![..copy_len]));

        // h_t = A * h_{t-1} + B * x_t
        let new_h = self.a.dot(&self.h) + self.b.dot(&padded);
        self.h = new_h;

        // Per-goal scores: sigmoid(C_i . h_t + D_i) for i in 0..k
        let actual_k = k.min(self.c.nrows());
        let mut scores = Vec::with_capacity(actual_k);
        for i in 0..actual_k {
            let row = self.c.row(i);
            let logit = row.dot(&self.h) + self.d[i];
            scores.push(sigmoid(logit));
        }
        scores
    }

    fn state(&self) -> &[f32] {
        self.h.as_slice().unwrap()
    }

    fn load_state(&mut self, state: &[f32]) {
        let len = state.len().min(self.hidden_dim);
        self.h
            .slice_mut(ndarray::s![..len])
            .assign(&ArrayView1::from(&state[..len]));
    }

    fn reset(&mut self) {
        self.h = Array1::zeros(self.hidden_dim);
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ssm() -> LinearSsm {
        LinearSsm::new(64, 16, 4, 5)
    }

    #[test]
    fn test_new_dimensions() {
        let ssm = make_ssm();
        // input_dim = hidden_dim + max_k + num_scalar_features = 16 + 4 + 5 = 25
        assert_eq!(ssm.projection.shape(), &[16, 64]);
        assert_eq!(ssm.a.shape(), &[16, 16]);
        assert_eq!(ssm.b.shape(), &[16, 25]);
        assert_eq!(ssm.c.shape(), &[4, 16]);
        assert_eq!(ssm.d.len(), 4);
        assert_eq!(ssm.h.len(), 16);
    }

    #[test]
    fn test_update_output_length() {
        let mut ssm = make_ssm();
        let input = vec![0.1; 25];
        let scores = ssm.update(&input, 3);
        assert_eq!(scores.len(), 3);
    }

    #[test]
    fn test_update_clamps_to_max_k() {
        let mut ssm = make_ssm();
        let input = vec![0.1; 25];
        let scores = ssm.update(&input, 100);
        assert_eq!(scores.len(), 4); // max_k = 4
    }

    #[test]
    fn test_sigmoid_bounds() {
        let mut ssm = make_ssm();
        let input = vec![1.0; 25];
        for _ in 0..10 {
            let scores = ssm.update(&input, 4);
            for &s in &scores {
                assert!(s > 0.0 && s < 1.0, "score {s} out of sigmoid range");
            }
        }
    }

    #[test]
    fn test_state_roundtrip() {
        let mut ssm = make_ssm();
        let input = vec![0.5; 25];
        ssm.update(&input, 4);

        let saved = ssm.state().to_vec();
        assert_eq!(saved.len(), 16);

        let mut ssm2 = make_ssm();
        ssm2.load_state(&saved);
        let restored = ssm2.state().to_vec();
        assert_eq!(saved, restored);
    }

    #[test]
    fn test_reset_zeros_state() {
        let mut ssm = make_ssm();
        let input = vec![1.0; 25];
        ssm.update(&input, 4);
        assert!(ssm.state().iter().any(|&v| v != 0.0));

        ssm.reset();
        assert!(ssm.state().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_project_embedding() {
        let ssm = make_ssm();
        let emb = Array1::from_vec(vec![0.1; 64]);
        let projected = ssm.project_embedding(emb.view());
        assert_eq!(projected.len(), 16);
    }

    #[test]
    fn test_update_with_short_input_pads() {
        let mut ssm = make_ssm();
        let short = vec![0.5; 5];
        let scores = ssm.update(&short, 4);
        assert_eq!(scores.len(), 4);
        for &s in &scores {
            assert!(s > 0.0 && s < 1.0);
        }
    }
}

/// Simple deterministic pseudo-random for weight init.
/// Not cryptographic, just needs variety across dimensions.
fn rand_f32() -> f32 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEED: AtomicU64 = AtomicU64::new(0x517cc1b727220a95);
    let mut s = SEED.fetch_add(1, Ordering::Relaxed);
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    s = s.wrapping_mul(0x2545F4914F6CDD1D);
    (s >> 40) as f32 / (1u64 << 24) as f32
}
