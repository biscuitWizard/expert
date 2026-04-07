use ndarray::{Array1, ArrayView1};

pub fn cosine_similarity(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
    let dot = a.dot(&b);
    let norm_a = a.dot(&a).sqrt();
    let norm_b = b.dot(&b).sqrt();
    let denom = norm_a * norm_b;
    if denom < 1e-12 { 0.0 } else { dot / denom }
}

pub fn ema_update(old: f32, new: f32, alpha: f32) -> f32 {
    alpha * new + (1.0 - alpha) * old
}

/// Update a running centroid (mean) in place. `count` is the number of samples
/// *after* including `new_vec` (i.e., caller should increment before calling).
pub fn centroid_update(centroid: &mut Array1<f32>, new_vec: ArrayView1<f32>, count: u64) {
    let n = count as f32;
    centroid.zip_mut_with(&new_vec, |c, &v| {
        *c += (v - *c) / n;
    });
}

/// Mahalanobis distance with diagonal covariance (independent dimensions).
/// `inv_var` is the element-wise reciprocal of the per-dimension variance.
/// Falls back to Euclidean distance scaled by 1/dim if any variance is zero.
pub fn mahalanobis_diagonal(
    x: ArrayView1<f32>,
    mean: ArrayView1<f32>,
    inv_var: ArrayView1<f32>,
) -> f32 {
    let diff = &x - &mean;
    let weighted = &diff * &diff * &inv_var;
    weighted.sum().sqrt()
}

/// Update running variance using Welford's online algorithm.
/// `count` is the sample count *after* including `new_vec`.
pub fn variance_update(
    mean: &Array1<f32>,
    old_mean: &Array1<f32>,
    var_accum: &mut Array1<f32>,
    new_vec: ArrayView1<f32>,
) {
    let delta_old = &new_vec - &old_mean.view();
    let delta_new = &new_vec - &mean.view();
    var_accum.zip_mut_with(&(&delta_old * &delta_new), |v, &d| {
        *v += d;
    });
}

/// Compute the inverse variance from the accumulated M2 and sample count.
/// Clamps to a minimum variance to avoid division by zero.
pub fn inverse_variance(var_accum: ArrayView1<f32>, count: u64) -> Array1<f32> {
    let n = count.max(2) as f32;
    let min_var = 1e-6;
    var_accum.mapv(|m2| {
        let var = (m2 / (n - 1.0)).max(min_var);
        1.0 / var
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_cosine_identical() {
        let a = array![1.0, 0.0, 0.0];
        let b = array![1.0, 0.0, 0.0];
        assert!((cosine_similarity(a.view(), b.view()) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = array![1.0, 0.0];
        let b = array![0.0, 1.0];
        assert!(cosine_similarity(a.view(), b.view()).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = array![0.0, 0.0];
        let b = array![1.0, 1.0];
        assert_eq!(cosine_similarity(a.view(), b.view()), 0.0);
    }

    #[test]
    fn test_ema() {
        let result = ema_update(0.5, 1.0, 0.1);
        assert!((result - 0.55).abs() < 1e-6);
    }

    #[test]
    fn test_centroid_update() {
        let mut centroid = array![1.0, 2.0];
        let new = array![3.0, 4.0];
        centroid_update(&mut centroid, new.view(), 2);
        assert!((centroid[0] - 2.0).abs() < 1e-6);
        assert!((centroid[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_mahalanobis_uniform_variance() {
        let x = array![3.0, 0.0];
        let mean = array![0.0, 0.0];
        let inv_var = array![1.0, 1.0];
        let d = mahalanobis_diagonal(x.view(), mean.view(), inv_var.view());
        assert!((d - 3.0).abs() < 1e-6);
    }
}
