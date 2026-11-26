//! Unit contracts for the startup pipeline's pure pieces: the metadata a failed
//! model-loading attribution is recorded with.

use super::*;

const ONE_GIGABYTE_BYTES: u64 = 1_073_741_824;

fn reviewed_ornith_artifact(
    total_payload_bytes: u64,
    shard_count: usize,
) -> ReviewedArtifactMetadata {
    ReviewedArtifactMetadata {
        model_id: "ornith-1.0".to_owned(),
        model_revision: "revision-7".to_owned(),
        total_payload_bytes,
        shard_count,
    }
}

#[test]
fn should_leave_the_artifact_identity_out_when_the_artifact_never_validated() {
    let failed_loading_metadata = failed_model_loading_metadata(None, "artifact validation failed");

    assert_eq!(
        failed_loading_metadata.outcome,
        PerformanceAttributionOutcome::Failed
    );
    assert_eq!(failed_loading_metadata.model_id, None);
    assert_eq!(failed_loading_metadata.model_revision, None);
    assert_eq!(failed_loading_metadata.total_artifact_payload_bytes, None);
    assert_eq!(failed_loading_metadata.model_shard_count, None);
    assert_eq!(
        failed_loading_metadata.failure_description.as_deref(),
        Some("artifact validation failed")
    );
}

#[test]
fn should_carry_the_reviewed_artifact_identity_into_a_failed_loading() {
    let reviewed_artifact = reviewed_ornith_artifact(ONE_GIGABYTE_BYTES, 3);

    let failed_loading_metadata =
        failed_model_loading_metadata(Some(&reviewed_artifact), "Qwen3.5 engine selection failed");

    assert_eq!(
        failed_loading_metadata.model_id.as_deref(),
        Some("ornith-1.0")
    );
    assert_eq!(
        failed_loading_metadata.model_revision.as_deref(),
        Some("revision-7")
    );
    assert_eq!(
        failed_loading_metadata.total_artifact_payload_bytes,
        Some(ONE_GIGABYTE_BYTES)
    );
    assert_eq!(failed_loading_metadata.model_shard_count, Some(3));
    assert_eq!(
        failed_loading_metadata.failure_description.as_deref(),
        Some("Qwen3.5 engine selection failed")
    );
}

#[test]
fn should_report_an_empty_artifact_as_zero_rather_than_unknown() {
    let reviewed_artifact = reviewed_ornith_artifact(0, 0);

    let failed_loading_metadata =
        failed_model_loading_metadata(Some(&reviewed_artifact), "tokenizer initialization failed");

    assert_eq!(
        failed_loading_metadata.total_artifact_payload_bytes,
        Some(0)
    );
    assert_eq!(failed_loading_metadata.model_shard_count, Some(0));
}

#[test]
fn should_never_claim_memory_figures_or_a_finished_prefill_for_a_failed_loading() {
    let reviewed_artifact = reviewed_ornith_artifact(ONE_GIGABYTE_BYTES, 1);

    let failed_loading_metadata = failed_model_loading_metadata(
        Some(&reviewed_artifact),
        "prompt-processing chunk configuration failed",
    );

    assert!(!failed_loading_metadata.prefill_transient_observation_completed);
    assert_eq!(
        failed_loading_metadata.prefill_observed_transient_high_water_bytes,
        0
    );
    assert_eq!(failed_loading_metadata.resident_model_payload_bytes, None);
    assert_eq!(failed_loading_metadata.mlx_active_memory_bytes, None);
    assert_eq!(failed_loading_metadata.mlx_buffer_cache_bytes, None);
    assert_eq!(failed_loading_metadata.mlx_peak_memory_bytes, None);
}
