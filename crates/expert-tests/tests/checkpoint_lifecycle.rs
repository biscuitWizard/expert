//! SSM checkpoint and signal serde / validation tests (no external services).

use expert_ssm::checkpoint::SsmCheckpoint;
use expert_ssm::ssm::LinearSsm;
use expert_types::signals::CheckpointAvailable;

#[test]
fn ssm_checkpoint_json_roundtrip() {
    let ssm = LinearSsm::new(64, 16, 4, 5);
    let ckpt = ssm.save_checkpoint("roundtrip-1");

    let json = serde_json::to_string(&ckpt).expect("serialize SsmCheckpoint");
    let back: SsmCheckpoint = serde_json::from_str(&json).expect("deserialize SsmCheckpoint");

    assert_eq!(back.id, ckpt.id);
    assert_eq!(back.format_version, ckpt.format_version);
    assert_eq!(back.embedding_dim, ckpt.embedding_dim);
    assert_eq!(back.hidden_dim, ckpt.hidden_dim);
    assert_eq!(back.max_k, ckpt.max_k);
    assert_eq!(back.num_scalar_features, ckpt.num_scalar_features);
    assert_eq!(back.projection, ckpt.projection);
    assert_eq!(back.a, ckpt.a);
    assert_eq!(back.b, ckpt.b);
    assert_eq!(back.c, ckpt.c);
    assert_eq!(back.d, ckpt.d);
}

#[test]
fn linear_ssm_save_checkpoint_from_checkpoint_weights_match() {
    let ssm = LinearSsm::new(64, 16, 4, 5);
    let ckpt = ssm.save_checkpoint("weights-1");

    let ssm2 = LinearSsm::from_checkpoint(&ckpt).expect("from_checkpoint");

    assert_eq!(
        ssm2.projection.as_slice().unwrap(),
        ssm.projection.as_slice().unwrap()
    );
    assert_eq!(ssm2.a.as_slice().unwrap(), ssm.a.as_slice().unwrap());
    assert_eq!(ssm2.b.as_slice().unwrap(), ssm.b.as_slice().unwrap());
    assert_eq!(ssm2.c.as_slice().unwrap(), ssm.c.as_slice().unwrap());
    assert_eq!(ssm2.d.as_slice().unwrap(), ssm.d.as_slice().unwrap());
}

#[test]
fn checkpoint_available_serde_includes_timescale() {
    let msg = CheckpointAvailable {
        checkpoint_id: "ckpt-abc".to_string(),
        domain: Some("domain-x".to_string()),
        path: "s3://bucket/key".to_string(),
        created_at: 1_700_000_000_000,
        timescale: Some("slow".to_string()),
    };

    let json = serde_json::to_string(&msg).expect("serialize CheckpointAvailable");
    assert!(json.contains("slow"), "expected timescale in JSON: {json}");

    let back: CheckpointAvailable =
        serde_json::from_str(&json).expect("deserialize CheckpointAvailable");
    assert_eq!(back.timescale, Some("slow".to_string()));
    assert_eq!(back.checkpoint_id, msg.checkpoint_id);
    assert_eq!(back.path, msg.path);
}

#[test]
fn ssm_checkpoint_validate_rejects_wrong_format_version() {
    let mut ckpt = LinearSsm::new(64, 16, 4, 5).save_checkpoint("bad-ver");
    ckpt.format_version = 999;

    let err = ckpt
        .validate()
        .expect_err("wrong format_version should fail");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("format version") || msg.contains("999"),
        "unexpected error: {msg}"
    );
}
