//! Unit tests for the pipeline's pure decisions.

use super::*;

#[test]
#[ignore = "pending implementation of resolve_reference_mode"]
fn should_pick_bfloat16_when_the_weights_fit_in_seventy_percent_of_available_memory() {
    let resolved = resolve_reference_mode(
        ReferenceMode::Auto,
        18 * 1024 * 1024 * 1024,
        28 * 1024 * 1024 * 1024,
    );

    assert_eq!(resolved, ReferenceMode::Bfloat16);
}

#[test]
#[ignore = "pending implementation of resolve_reference_mode"]
fn should_fall_back_to_the_uniform_baseline_when_bfloat16_does_not_fit() {
    let resolved = resolve_reference_mode(
        ReferenceMode::Auto,
        54 * 1024 * 1024 * 1024,
        28 * 1024 * 1024 * 1024,
    );

    assert_eq!(resolved, ReferenceMode::Uniform4Bit);
}

#[test]
#[ignore = "pending implementation of resolve_reference_mode"]
fn should_pass_an_explicit_reference_mode_through_unchanged() {
    let resolved = resolve_reference_mode(ReferenceMode::Bfloat16, u64::MAX, 1);

    assert_eq!(resolved, ReferenceMode::Bfloat16);
}

#[test]
#[ignore = "pending implementation of estimate_bfloat16_weight_bytes"]
fn should_count_fp8_tensors_twice_when_estimating_the_bfloat16_footprint() {
    // A shard whose header declares one F8_E4M3 tensor of 1024 bytes: the
    // estimate is the file size plus 1024.
    let directory = tempfile::tempdir().expect("a temporary directory should be creatable");
    let header = r#"{"w": {"dtype": "F8_E4M3", "shape": [32, 32], "data_offsets": [0, 1024]}}"#;
    let mut shard_bytes = Vec::new();
    shard_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    shard_bytes.extend_from_slice(header.as_bytes());
    shard_bytes.extend_from_slice(&[0_u8; 1024]);
    std::fs::write(directory.path().join("model.safetensors"), &shard_bytes)
        .expect("the fixture shard should be writable");

    let estimate =
        estimate_bfloat16_weight_bytes(directory.path()).expect("the estimate should succeed");

    assert_eq!(estimate, shard_bytes.len() as u64 + 1024);
}
