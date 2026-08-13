use std::time::Duration;

use astronomical_config::AstronomicalConfig;
use astronomical_ipc_protocol::RequestId;
use astronomical_model_serving::PersistentPromptCacheDiskStoreConfig;

use super::speculative_prefill::{prepare_representative_prompt, run_representative_generation};

#[tokio::test]
#[ignore = "loads the configured target and drafter and proves persistent SpecPrefill works while retaining every selectable token"]
async fn should_complete_persistent_speculative_prefill_when_keep_percentage_is_full() {
    tokio::time::timeout(Duration::from_secs(115), async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let target_model_directory = crate::common::configured_ornith_model_artifact_directory();
        let astronomical_config = AstronomicalConfig::load_from_default_location()
            .expect("the standard Astronomical configuration should load");
        let configured_speculative_prefill = astronomical_config
            .speculative_prefill()
            .expect("the configured SpecPrefill policy should resolve");
        let draft_model_id = configured_speculative_prefill
            .draft_model_id()
            .expect("the configured SpecPrefill policy should name a drafter")
            .to_owned();
        let draft_model_directory = astronomical_config
            .find_configured_model_directory_by_id(&draft_model_id)
            .expect("configured drafter model discovery should complete")
            .expect("the configured drafter should be available");
        let representative_prompt = prepare_representative_prompt(&target_model_directory);
        let mlx_memory_limits =
            crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
        let persistent_prompt_cache_directory = tempfile::tempdir()
            .expect("the full-keep qualification should create a cache directory");
        let persistent_prompt_cache_config = PersistentPromptCacheDiskStoreConfig::new(
            persistent_prompt_cache_directory.path().join("target"),
            persistent_prompt_cache_directory.path().to_path_buf(),
            crate::common::configured_model_artifact_prompt_cache_maximum_size_bytes(),
        );

        eprintln!(
            "[speculative-prefill-full-keep] status=progress phase=generation prompt_tokens={} ETA_seconds=115",
            representative_prompt.prompt_token_ids.len(),
        );
        let full_keep_measurement = run_representative_generation(
            &target_model_directory,
            &draft_model_directory,
            &draft_model_id,
            &representative_prompt,
            true,
            1,
            100,
            RequestId::new(95_090),
            Some(persistent_prompt_cache_config),
            mlx_memory_limits,
        )
        .await;

        assert_eq!(full_keep_measurement.generated_token_ids.len(), 1);
        assert_eq!(full_keep_measurement.speculative_prefill_fallback_count, 0);
        assert_eq!(
            full_keep_measurement.expert_weight_disk_page_load_count,
            0,
            "a resident target that fits beside the drafter must not demand-load expert pages",
        );
        assert_eq!(
            full_keep_measurement.speculative_prefill_draft_scored_suffix_token_count,
            representative_prompt.prompt_token_ids.len() as u64,
        );
        assert!(
            full_keep_measurement.speculative_prefill_selected_token_count > 0,
            "a successful full-keep request must publish its target selection"
        );
        assert_eq!(
            full_keep_measurement.speculative_prefill_target_persistent_state_write_count,
            1,
            "a successful persistent request must publish restorable target state"
        );
    })
    .await
    .expect("the full-keep persistent SpecPrefill qualification must finish within 115 seconds");
}
