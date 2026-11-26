//! Beside-source coverage for the load accounting: the memory floor sum and the metadata of the
//! two load outcomes. None of it needs a GPU.

use model_primitives::performance_attribution::PerformanceAttributionOutcome;

use super::{
    MAXIMUM_FAILURE_DESCRIPTION_CHARACTERS, failed_load_metadata, memory_floor_bytes,
    successful_load_metadata,
};
use crate::official_profile::{
    QWEN_IMAGE_21_ACTIVATION_HEADROOM_BYTES, qwen_image_21_official_model_id,
};

const ACTIVATION_HEADROOM_BYTES: u64 = QWEN_IMAGE_21_ACTIVATION_HEADROOM_BYTES as u64;
const REVIEWED_REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

#[test]
fn should_add_the_activation_headroom_to_the_three_component_payloads() {
    let floor_bytes = memory_floor_bytes([5_000, 700, 30]);

    assert_eq!(floor_bytes, Some(5_730 + ACTIVATION_HEADROOM_BYTES));
}

#[test]
fn should_floor_an_empty_artifact_at_the_activation_headroom() {
    assert_eq!(
        memory_floor_bytes([0, 0, 0]),
        Some(ACTIVATION_HEADROOM_BYTES)
    );
}

#[test]
fn should_accept_a_floor_that_lands_exactly_on_the_largest_representable_size() {
    let payload_bytes = u64::MAX - ACTIVATION_HEADROOM_BYTES;

    assert_eq!(memory_floor_bytes([payload_bytes, 0, 0]), Some(u64::MAX));
}

#[test]
fn should_reject_payloads_whose_sum_overflows() {
    assert_eq!(memory_floor_bytes([u64::MAX, 1, 0]), None);
    assert_eq!(
        memory_floor_bytes([u64::MAX / 2 + 1, u64::MAX / 2 + 1, 0]),
        None
    );
}

#[test]
fn should_reject_a_sum_that_overflows_only_once_the_headroom_is_added() {
    let payload_bytes = u64::MAX - ACTIVATION_HEADROOM_BYTES + 1;

    assert_eq!(memory_floor_bytes([payload_bytes, 0, 0]), None);
}

#[test]
fn should_describe_a_failed_load_with_no_payload_and_the_bounded_reason() {
    let failure_metadata = failed_load_metadata(REVIEWED_REVISION, "a rejected artifact");

    assert_eq!(
        failure_metadata.outcome,
        PerformanceAttributionOutcome::Failed
    );
    assert_eq!(
        failure_metadata.model_id.as_deref(),
        Some(qwen_image_21_official_model_id())
    );
    assert_eq!(
        failure_metadata.model_revision.as_deref(),
        Some(REVIEWED_REVISION)
    );
    assert_eq!(failure_metadata.total_artifact_payload_bytes, None);
    assert_eq!(failure_metadata.resident_model_payload_bytes, Some(0));
    assert_eq!(failure_metadata.model_shard_count, None);
    assert_eq!(
        failure_metadata.failure_description.as_deref(),
        Some("a rejected artifact")
    );
}

#[test]
fn should_cut_a_long_failure_description_at_the_character_bound() {
    let long_description = "é".repeat(MAXIMUM_FAILURE_DESCRIPTION_CHARACTERS + 40);

    let failure_metadata = failed_load_metadata(REVIEWED_REVISION, &long_description);

    let bounded_description = failure_metadata
        .failure_description
        .expect("a failed load always carries a description");
    assert_eq!(
        bounded_description.chars().count(),
        MAXIMUM_FAILURE_DESCRIPTION_CHARACTERS
    );
}

#[test]
fn should_keep_an_empty_failure_description_empty() {
    let failure_metadata = failed_load_metadata(REVIEWED_REVISION, "");

    assert_eq!(failure_metadata.failure_description.as_deref(), Some(""));
}

#[test]
fn should_describe_an_accepted_load_with_the_payload_total_and_three_shards() {
    let success_metadata =
        successful_load_metadata(REVIEWED_REVISION, 9_000 + ACTIVATION_HEADROOM_BYTES);

    assert_eq!(
        success_metadata.outcome,
        PerformanceAttributionOutcome::Success
    );
    assert_eq!(success_metadata.total_artifact_payload_bytes, Some(9_000));
    assert_eq!(success_metadata.resident_model_payload_bytes, Some(0));
    assert_eq!(success_metadata.model_shard_count, Some(3));
    assert_eq!(success_metadata.failure_description, None);
}

#[test]
fn should_leave_the_payload_total_unknown_when_the_floor_is_below_the_headroom() {
    let success_metadata =
        successful_load_metadata(REVIEWED_REVISION, ACTIVATION_HEADROOM_BYTES - 1);

    assert_eq!(success_metadata.total_artifact_payload_bytes, None);
}
