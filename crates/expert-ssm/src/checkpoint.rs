use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::ssm::LinearSsm;

const FORMAT_VERSION: u32 = 1;

/// Serializable snapshot of all SSM weight matrices.
/// Written to disk by the training service, loaded by ssm-workers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SsmCheckpoint {
    pub id: String,
    pub format_version: u32,
    pub embedding_dim: usize,
    pub hidden_dim: usize,
    pub max_k: usize,
    pub num_scalar_features: usize,
    pub projection: Vec<f32>,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub c: Vec<f32>,
    pub d: Vec<f32>,
    pub created_at: u64,
    pub domain: Option<String>,
    pub training_batch_id: Option<String>,
    pub timescale: Option<String>,
}

impl SsmCheckpoint {
    pub fn current_format_version() -> u32 {
        FORMAT_VERSION
    }

    pub fn validate(&self) -> Result<()> {
        if self.format_version != FORMAT_VERSION {
            bail!(
                "unsupported checkpoint format version {} (expected {})",
                self.format_version,
                FORMAT_VERSION
            );
        }
        let expected_proj = self.hidden_dim * self.embedding_dim;
        if self.projection.len() != expected_proj {
            bail!(
                "projection size mismatch: got {}, expected {}",
                self.projection.len(),
                expected_proj
            );
        }
        let expected_a = self.hidden_dim * self.hidden_dim;
        if self.a.len() != expected_a {
            bail!(
                "A matrix size mismatch: got {}, expected {}",
                self.a.len(),
                expected_a
            );
        }
        let input_dim = self.hidden_dim + self.max_k + self.num_scalar_features;
        let expected_b = self.hidden_dim * input_dim;
        if self.b.len() != expected_b {
            bail!(
                "B matrix size mismatch: got {}, expected {}",
                self.b.len(),
                expected_b
            );
        }
        let expected_c = self.max_k * self.hidden_dim;
        if self.c.len() != expected_c {
            bail!(
                "C matrix size mismatch: got {}, expected {}",
                self.c.len(),
                expected_c
            );
        }
        if self.d.len() != self.max_k {
            bail!(
                "D vector size mismatch: got {}, expected {}",
                self.d.len(),
                self.max_k
            );
        }
        Ok(())
    }
}

impl LinearSsm {
    pub fn save_checkpoint(&self, id: &str) -> SsmCheckpoint {
        let (rows_a, _) = self.a.dim();
        let hidden_dim = rows_a;
        let (_, cols_proj) = self.projection.dim();
        let embedding_dim = cols_proj;
        let (rows_c, _) = self.c.dim();
        let max_k = rows_c;
        let (_, cols_b) = self.b.dim();
        let num_scalar_features = cols_b.saturating_sub(hidden_dim + max_k);

        SsmCheckpoint {
            id: id.to_string(),
            format_version: FORMAT_VERSION,
            embedding_dim,
            hidden_dim,
            max_k,
            num_scalar_features,
            projection: self.projection.iter().copied().collect(),
            a: self.a.iter().copied().collect(),
            b: self.b.iter().copied().collect(),
            c: self.c.iter().copied().collect(),
            d: self.d.iter().copied().collect(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            domain: None,
            training_batch_id: None,
            timescale: None,
        }
    }

    pub fn load_checkpoint(&mut self, ckpt: &SsmCheckpoint) -> Result<()> {
        ckpt.validate()?;

        let (rows_a, _) = self.a.dim();
        if ckpt.hidden_dim != rows_a {
            bail!(
                "hidden_dim mismatch: checkpoint has {}, SSM has {}",
                ckpt.hidden_dim,
                rows_a
            );
        }
        let (_, cols_proj) = self.projection.dim();
        if ckpt.embedding_dim != cols_proj {
            bail!(
                "embedding_dim mismatch: checkpoint has {}, SSM has {}",
                ckpt.embedding_dim,
                cols_proj
            );
        }

        use ndarray::{Array1, Array2};

        self.projection = Array2::from_shape_vec(
            (ckpt.hidden_dim, ckpt.embedding_dim),
            ckpt.projection.clone(),
        )?;
        self.a = Array2::from_shape_vec((ckpt.hidden_dim, ckpt.hidden_dim), ckpt.a.clone())?;
        let input_dim = ckpt.hidden_dim + ckpt.max_k + ckpt.num_scalar_features;
        self.b = Array2::from_shape_vec((ckpt.hidden_dim, input_dim), ckpt.b.clone())?;
        self.c = Array2::from_shape_vec((ckpt.max_k, ckpt.hidden_dim), ckpt.c.clone())?;
        self.d = Array1::from_vec(ckpt.d.clone());

        self.h = Array1::zeros(ckpt.hidden_dim);

        Ok(())
    }

    pub fn from_checkpoint(ckpt: &SsmCheckpoint) -> Result<Self> {
        ckpt.validate()?;
        let input_dim = ckpt.hidden_dim + ckpt.max_k + ckpt.num_scalar_features;

        use ndarray::{Array1, Array2};

        Ok(Self {
            projection: Array2::from_shape_vec(
                (ckpt.hidden_dim, ckpt.embedding_dim),
                ckpt.projection.clone(),
            )?,
            a: Array2::from_shape_vec((ckpt.hidden_dim, ckpt.hidden_dim), ckpt.a.clone())?,
            b: Array2::from_shape_vec((ckpt.hidden_dim, input_dim), ckpt.b.clone())?,
            c: Array2::from_shape_vec((ckpt.max_k, ckpt.hidden_dim), ckpt.c.clone())?,
            d: Array1::from_vec(ckpt.d.clone()),
            h: Array1::zeros(ckpt.hidden_dim),
            input_dim,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_roundtrip() {
        let ssm = LinearSsm::new(64, 16, 4, 5);
        let ckpt = ssm.save_checkpoint("test-1");

        assert_eq!(ckpt.format_version, FORMAT_VERSION);
        assert_eq!(ckpt.embedding_dim, 64);
        assert_eq!(ckpt.hidden_dim, 16);
        assert_eq!(ckpt.max_k, 4);
        assert_eq!(ckpt.num_scalar_features, 5);

        let ssm2 = LinearSsm::from_checkpoint(&ckpt).unwrap();
        assert_eq!(ssm2.a.dim(), ssm.a.dim());
        assert_eq!(ssm2.b.dim(), ssm.b.dim());
        assert_eq!(ssm2.c.dim(), ssm.c.dim());
        assert_eq!(ssm2.d.len(), ssm.d.len());
    }

    #[test]
    fn test_checkpoint_load_into_existing() {
        let ssm_src = LinearSsm::new(64, 16, 4, 5);
        let ckpt = ssm_src.save_checkpoint("test-load");

        let mut ssm_dst = LinearSsm::new(64, 16, 4, 5);
        ssm_dst.load_checkpoint(&ckpt).unwrap();

        assert_eq!(ssm_dst.a.as_slice().unwrap(), ssm_src.a.as_slice().unwrap());
        assert_eq!(ssm_dst.b.as_slice().unwrap(), ssm_src.b.as_slice().unwrap());
        assert_eq!(ssm_dst.c.as_slice().unwrap(), ssm_src.c.as_slice().unwrap());
        assert_eq!(ssm_dst.d.as_slice().unwrap(), ssm_src.d.as_slice().unwrap());
    }

    #[test]
    fn test_checkpoint_serde_json_roundtrip() {
        let ssm = LinearSsm::new(64, 16, 4, 5);
        let ckpt = ssm.save_checkpoint("serde-test");
        let json = serde_json::to_string(&ckpt).unwrap();
        let ckpt2: SsmCheckpoint = serde_json::from_str(&json).unwrap();
        assert_eq!(ckpt.id, ckpt2.id);
        assert_eq!(ckpt.a, ckpt2.a);
    }

    #[test]
    fn test_checkpoint_format_version_mismatch() {
        let mut ckpt = LinearSsm::new(64, 16, 4, 5).save_checkpoint("bad-version");
        ckpt.format_version = 999;
        assert!(ckpt.validate().is_err());
    }

    #[test]
    fn test_checkpoint_dimension_mismatch() {
        let ckpt = LinearSsm::new(64, 16, 4, 5).save_checkpoint("dim-test");
        let mut ssm_wrong = LinearSsm::new(32, 8, 4, 5);
        assert!(ssm_wrong.load_checkpoint(&ckpt).is_err());
    }
}
