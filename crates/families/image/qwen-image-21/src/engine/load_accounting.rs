//! The numbers a Qwen-Image-2.1 model load reports: the memory floor the artifact needs and the
//! metadata of the load's performance report. Kept apart from the component owner so the owner
//! only sequences the load, and so this arithmetic is checked without a GPU.

use model_primitives::performance_attribution::{
    ModelLoadingPerformanceAttributionMetadata, PerformanceAttributionOutcome,
};

use crate::official_profile::{
    QWEN_IMAGE_21_ACTIVATION_HEADROOM_BYTES, qwen_image_21_official_model_id,
};

/// Bound on the public failure descriptions that can carry component detail.
pub(super) const MAXIMUM_FAILURE_DESCRIPTION_CHARACTERS: usize = 512;

/// The artifact's own weight bytes plus the measured activation headroom, or `None` when the sum
/// does not fit in 64 bits. The worker may raise the live ceiling from here but never below it.
pub(super) fn memory_floor_bytes(component_payload_bytes: [u64; 3]) -> Option<u64> {
    let mut floor_bytes = 0_u64;
    for payload_bytes in component_payload_bytes {
        floor_bytes = floor_bytes.checked_add(payload_bytes)?;
    }
    floor_bytes.checked_add(QWEN_IMAGE_21_ACTIVATION_HEADROOM_BYTES as u64)
}

/// Metadata for a load whose artifact validation was rejected. The description is cut to
/// `MAXIMUM_FAILURE_DESCRIPTION_CHARACTERS` characters.
pub(super) fn failed_load_metadata(
    model_revision: &str,
    error_description: &str,
) -> ModelLoadingPerformanceAttributionMetadata {
    let bounded_failure = error_description
        .chars()
        .take(MAXIMUM_FAILURE_DESCRIPTION_CHARACTERS)
        .collect();
    ModelLoadingPerformanceAttributionMetadata {
        outcome: PerformanceAttributionOutcome::Failed,
        model_id: Some(qwen_image_21_official_model_id().to_owned()),
        model_revision: Some(model_revision.to_owned()),
        prefill_transient_observation_completed: false,
        prefill_observed_transient_high_water_bytes: 0,
        total_artifact_payload_bytes: None,
        resident_model_payload_bytes: Some(0),
        model_shard_count: None,
        mlx_active_memory_bytes: None,
        mlx_buffer_cache_bytes: None,
        mlx_peak_memory_bytes: None,
        failure_description: Some(bounded_failure),
    }
}

/// Metadata for an accepted load. The payload total is the memory floor without its activation
/// headroom, and no weights are resident yet because they map at the first request.
pub(super) fn successful_load_metadata(
    model_revision: &str,
    memory_floor_bytes: u64,
) -> ModelLoadingPerformanceAttributionMetadata {
    let total_artifact_payload_bytes =
        memory_floor_bytes.checked_sub(QWEN_IMAGE_21_ACTIVATION_HEADROOM_BYTES as u64);
    // The reviewed artifact stores each component as one weight file, so the shard count
    // is the component count.
    let model_shard_count = Some(3_usize);
    ModelLoadingPerformanceAttributionMetadata {
        outcome: PerformanceAttributionOutcome::Success,
        model_id: Some(qwen_image_21_official_model_id().to_owned()),
        model_revision: Some(model_revision.to_owned()),
        prefill_transient_observation_completed: false,
        prefill_observed_transient_high_water_bytes: 0,
        total_artifact_payload_bytes,
        resident_model_payload_bytes: Some(0),
        model_shard_count,
        mlx_active_memory_bytes: None,
        mlx_buffer_cache_bytes: None,
        mlx_peak_memory_bytes: None,
        failure_description: None,
    }
}

#[cfg(test)]
#[path = "load_accounting_tests.rs"]
mod load_accounting_tests;
