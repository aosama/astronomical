use std::time::Duration;

use astronomical_ipc_protocol::RequestId;
use astronomical_model_serving::PersistentPromptCacheDiskStoreConfig;

use super::{
    RepresentativeGenerationMeasurement, SPECULATIVE_PREFILL_KEEP_PERCENTAGE,
    SPECULATIVE_PREFILL_SELECTION_CHUNK_TOKEN_COUNT, build_loaded_representative_engine,
    generate_representative_measurement, prepare_representative_prompt,
};

const REPRESENTATIVE_OUTPUT_TOKEN_COUNT: u16 = 1;
const ZERO_THRESHOLD: u64 = 0;
const MINIMUM_PERSISTENT_STATE_WRITE_COUNT: u64 = 1;
const MINIMUM_PUBLISHED_TARGET_STATE_FILE_COUNT: usize = 1;
// The on-disk layout under the active-model cache directory is a stable cache
// format contract, so the journey deletes from the real target-state directory
// exactly as an out-of-band cleanup tool would.
const SPECULATIVE_PREFILL_TARGET_STATES_DIRECTORY_NAME: &str = "speculative_prefill_target_states";

#[tokio::test]
#[ignore = "proves a sparse target-state file deleted out-of-band fails open as a cold restore under the configured MLX ceiling"]
async fn should_fail_open_when_a_persisted_target_state_vanishes_between_requests() {
    tokio::time::timeout(Duration::from_secs(115), async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let target_model_directory =
            crate::serving_acceptance::support::configured_resident_sparse_moe_model_directory();
        let (draft_model_directory, draft_model_id) =
            crate::serving_acceptance::support::configured_speculative_prefill_draft_model(
                &target_model_directory,
            );
        let representative_prompt = prepare_representative_prompt(&target_model_directory);
        let persistent_prompt_cache_root_directory =
            tempfile::tempdir().expect("the acceptance should create a shared SSD cache root");
        let target_persistent_prompt_cache_directory =
            persistent_prompt_cache_root_directory.path().join("target");
        let target_states_directory = target_persistent_prompt_cache_directory
            .join(SPECULATIVE_PREFILL_TARGET_STATES_DIRECTORY_NAME);
        let persistent_prompt_cache_disk_store_config = PersistentPromptCacheDiskStoreConfig::new(
            target_persistent_prompt_cache_directory,
            persistent_prompt_cache_root_directory.path().to_path_buf(),
            crate::common::configured_model_artifact_prompt_cache_maximum_size_bytes(),
        );
        let mlx_memory_limits = crate::common::sample_serving_acceptance_mlx_memory_limits().await;
        // One engine across both requests is the regression's defining condition:
        // a fresh engine rescans the cache directory at startup and would never
        // hold the stale in-memory index entry that the vanished file exposes.
        let mut loaded_representative_engine = build_loaded_representative_engine(
            &target_model_directory,
            &draft_model_directory,
            &draft_model_id,
            true,
            SPECULATIVE_PREFILL_KEEP_PERCENTAGE,
            SPECULATIVE_PREFILL_SELECTION_CHUNK_TOKEN_COUNT,
            Some(persistent_prompt_cache_disk_store_config),
            mlx_memory_limits,
        )
        .await;

        eprintln!(
            "[speculative-prefill-vanished-target-state] status=progress phase=cold_target_state_population prompt_tokens={} ETA_seconds=115",
            representative_prompt.prompt_token_ids.len(),
        );
        let cold_measurement = generate_representative_measurement(
            &mut loaded_representative_engine,
            &representative_prompt,
            REPRESENTATIVE_OUTPUT_TOKEN_COUNT,
            RequestId::new(95_300),
        )
        .await;
        assert_cold_request_publishes_a_target_state_artifact(
            &cold_measurement,
            &target_states_directory,
        );

        remove_all_target_state_files(&target_states_directory);
        eprintln!(
            "[speculative-prefill-vanished-target-state] status=progress phase=vanished_target_state_recovery prompt_tokens={}",
            representative_prompt.prompt_token_ids.len(),
        );
        let recovery_measurement = generate_representative_measurement(
            &mut loaded_representative_engine,
            &representative_prompt,
            REPRESENTATIVE_OUTPUT_TOKEN_COUNT,
            RequestId::new(95_301),
        )
        .await;
        assert_vanished_target_state_restores_as_a_miss(&recovery_measurement);

        eprintln!(
            "[speculative-prefill-vanished-target-state] status=success recovered_output_tokens={} republished_target_state_write_count={}",
            recovery_measurement.generated_token_ids.len(),
            recovery_measurement.speculative_prefill_target_persistent_state_write_count,
        );
    })
    .await
    .expect("the vanished target-state acceptance should finish within 115 seconds");
}

fn assert_cold_request_publishes_a_target_state_artifact(
    cold_measurement: &RepresentativeGenerationMeasurement,
    target_states_directory: &std::path::Path,
) {
    assert_eq!(
        cold_measurement.speculative_prefill_fallback_count,
        ZERO_THRESHOLD
    );
    assert_eq!(
        cold_measurement.speculative_prefill_target_persistent_state_restored_token_count,
        ZERO_THRESHOLD,
        "the cold request has no prior sparse target state to restore"
    );
    assert!(
        cold_measurement.speculative_prefill_target_persistent_state_write_count
            >= MINIMUM_PERSISTENT_STATE_WRITE_COUNT,
        "the cold request must publish reusable sparse target state before it can vanish"
    );
    assert!(
        count_target_state_files(target_states_directory)
            >= MINIMUM_PUBLISHED_TARGET_STATE_FILE_COUNT,
        "the sparse target-state artifact should exist on disk before the out-of-band deletion"
    );
}

fn assert_vanished_target_state_restores_as_a_miss(
    recovery_measurement: &RepresentativeGenerationMeasurement,
) {
    assert_eq!(
        recovery_measurement.generated_token_ids.len(),
        usize::from(REPRESENTATIVE_OUTPUT_TOKEN_COUNT),
        "the request whose sparse target state vanished must complete instead of failing"
    );
    assert_eq!(
        recovery_measurement.speculative_prefill_target_persistent_state_restored_token_count,
        ZERO_THRESHOLD,
        "the vanished sparse target state must restore as a miss, never as stale prefix work"
    );
    assert_eq!(
        recovery_measurement.speculative_prefill_fallback_count, ZERO_THRESHOLD,
        "the vanished sparse target state must fail open without degrading to a fallback"
    );
    assert_eq!(
        recovery_measurement.restored_target_persistent_prompt_cache_token_count, ZERO_THRESHOLD,
        "sparse target reuse must remain separate from exact dense target cache reuse"
    );
    assert!(
        recovery_measurement.speculative_prefill_target_persistent_state_write_count
            >= MINIMUM_PERSISTENT_STATE_WRITE_COUNT,
        "the recovery request should republish sparse target state so the next chat can reuse it"
    );
}

fn count_target_state_files(target_states_directory: &std::path::Path) -> usize {
    std::fs::read_dir(target_states_directory)
        .expect("the sparse target-state directory should exist after a cold request")
        .map(|entry| entry.expect("the sparse target-state directory should be readable"))
        .filter(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|extension| extension == "safetensors")
        })
        .count()
}

fn remove_all_target_state_files(target_states_directory: &std::path::Path) {
    for entry in std::fs::read_dir(target_states_directory)
        .expect("the sparse target-state directory should exist for the out-of-band deletion")
    {
        let entry = entry.expect("the sparse target-state directory should be readable");
        if entry
            .path()
            .extension()
            .is_some_and(|extension| extension == "safetensors")
        {
            std::fs::remove_file(entry.path()).expect(
                "the out-of-band deletion of a sparse target-state artifact should succeed",
            );
        }
    }
    assert_eq!(
        count_target_state_files(target_states_directory),
        0,
        "the out-of-band deletion must leave no sparse target-state artifacts behind"
    );
}
