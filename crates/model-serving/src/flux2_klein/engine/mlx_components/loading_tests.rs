//! Beside-source tests for the pure FLUX.2 Klein load-accounting units.

#[test]
fn should_sum_the_three_component_payload_byte_counts() {
    assert_eq!(
        super::flux2_klein_total_artifact_payload_bytes(2_000, 3_000, 5_000),
        Ok(10_000)
    );
}

#[test]
fn should_fail_payload_accounting_when_the_u64_sum_overflows() {
    assert_eq!(
        super::flux2_klein_total_artifact_payload_bytes(u64::MAX, 1, 0),
        Err("FLUX.2 Klein artifact payload accounting overflowed".to_owned())
    );
}

#[test]
fn should_fail_payload_accounting_at_the_second_addition() {
    // The first pair fits; the overflow surfaces when the VAE payload joins,
    // so every link in the checked chain must be overflow-safe.
    assert_eq!(
        super::flux2_klein_total_artifact_payload_bytes(u64::MAX - 1, 1, 1),
        Err("FLUX.2 Klein artifact payload accounting overflowed".to_owned())
    );
}

#[test]
fn should_accept_the_maximum_payload_without_overflow() {
    assert_eq!(
        super::flux2_klein_total_artifact_payload_bytes(u64::MAX, 0, 0),
        Ok(u64::MAX)
    );
}

#[test]
fn should_sum_the_three_component_shard_counts() {
    assert_eq!(super::flux2_klein_model_shard_count(2, 3, 5), Ok(10));
}

#[test]
fn should_fail_shard_accounting_when_the_usize_sum_overflows() {
    assert_eq!(
        super::flux2_klein_model_shard_count(usize::MAX, 1, 0),
        Err("FLUX.2 Klein artifact shard accounting overflowed".to_owned())
    );
}

#[test]
fn should_accept_the_maximum_shard_count_without_overflow() {
    assert_eq!(
        super::flux2_klein_model_shard_count(usize::MAX, 0, 0),
        Ok(usize::MAX)
    );
}
