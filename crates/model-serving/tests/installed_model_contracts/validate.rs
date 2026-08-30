use std::fs;

use astronomical_model_serving::Qwen3_5ArtifactValidator;
#[cfg(feature = "direct-mlx")]
use astronomical_model_serving::{
    InferenceEngine, PerformanceAttribution, PerformanceAttributionLog, Qwen3_5Engine,
    Qwen3_5FeedForwardArchitecture, Qwen3_5InferenceRequest, Qwen3_5PromptProcessingChunkSizer,
    Qwen3_5Tokenizer,
};

/// Links or copies a regular file from source to target, falling back to copy
/// when hard linking is not permitted (e.g. cross-volume or APFS restricted).
/// Skips directories and symlinks — the artifact validator only needs regular files.
fn link_or_copy_file(source: &std::path::Path, target: &std::path::Path) {
    if !source.is_file() {
        return;
    }
    if fs::hard_link(source, target).is_err() {
        fs::copy(source, target).unwrap_or_else(|error| {
            panic!("the test should copy {source:?} to {target:?}: {error}")
        });
    }
}

#[test]
#[ignore = "requires model_directories to discover the Ornith 1.5 acceptance artifact"]
fn should_validate_a_text_only_artifact_without_the_vision_sidecar() {
    let model_directory = crate::common::configured_large_sparse_moe_model_directory();
    let text_artifact_directory = tempfile::tempdir()
        .expect("the acceptance test should create a temporary text artifact directory");
    // Link all files from the model directory except the vision sidecar and
    // materialize an index that no longer assigns tensors to the omitted file.
    for directory_entry in fs::read_dir(&model_directory)
        .expect("the test should read the model directory")
        .filter_map(Result::ok)
    {
        let entry_path = directory_entry.path();
        let file_name = directory_entry.file_name();
        let file_name_str = file_name.to_string_lossy();
        if file_name_str == "optiq" {
            // Skip the vision sidecar directory entirely for a text-only test.
            continue;
        }
        if entry_path.is_dir() {
            // Skip subdirectories (e.g. .cache, refs) — the text-only
            // artifact validator only needs files, not nested directories.
            continue;
        }
        let target_path = text_artifact_directory.path().join(&*file_name_str);
        if file_name_str == "model.safetensors.index.json" {
            let mut index_document = serde_json::from_slice::<serde_json::Value>(
                &fs::read(&entry_path).expect("the test should read the source shard index"),
            )
            .expect("the source shard index should parse");
            index_document["weight_map"]
                .as_object_mut()
                .expect("the source shard index should contain a weight map")
                .retain(|tensor_name, _| !tensor_name.starts_with("vision_tower."));
            fs::write(
                &target_path,
                serde_json::to_vec(&index_document)
                    .expect("the text-only shard index should serialize"),
            )
            .expect("the test should write the text-only shard index");
            continue;
        }
        link_or_copy_file(&entry_path, &target_path);
    }

    let mut validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(text_artifact_directory.path(), 20_480)
        .expect("a text-only artifact should validate without the vision sidecar");

    let shard_count = validated_artifact.shard_count();
    let total_payload_bytes = validated_artifact.total_payload_bytes();
    let model_shard_file_names = validated_artifact
        .shard_index()
        .model_shard_file_names()
        .to_vec();
    assert_eq!(shard_count, model_shard_file_names.len());
    assert_eq!(
        total_payload_bytes,
        validated_artifact.shard_index().total_payload_bytes()
    );
    let model_shard_source_ids = validated_artifact
        .source_ids_for_file_names(&model_shard_file_names)
        .expect("all retained language shards should resolve to source identities");
    let shard_files = validated_artifact
        .take_safetensors_sources(&model_shard_source_ids)
        .expect("all retained language shard descriptors should transfer safely");
    assert_eq!(shard_files.len(), shard_count);
}

#[test]
#[ignore = "requires model_directories to discover the Ornith 1.5 acceptance artifact"]
fn should_validate_a_vision_enabled_artifact_with_the_vision_sidecar() {
    let model_directory = crate::common::configured_large_sparse_moe_model_directory();
    let vision_artifact_directory = tempfile::tempdir()
        .expect("the acceptance test should create a temporary vision artifact directory");
    // Link all files from the model directory including the optiq vision sidecar.
    for directory_entry in fs::read_dir(&model_directory)
        .expect("the test should read the model directory")
        .filter_map(Result::ok)
    {
        let entry_path = directory_entry.path();
        let file_name = directory_entry.file_name();
        let file_name_str = file_name.to_string_lossy();
        let target_path = vision_artifact_directory.path().join(&*file_name_str);
        if entry_path.is_dir() {
            // Skip non-model directories like .cache, .git, refs.
            // Only copy the optiq vision sidecar directory.
            if file_name_str == "optiq" {
                fs::create_dir_all(&target_path).expect("the test should create subdirectories");
                for sub_entry in fs::read_dir(&entry_path)
                    .expect("the test should read the subdirectory")
                    .filter_map(Result::ok)
                {
                    let sub_target = target_path.join(sub_entry.file_name());
                    link_or_copy_file(&sub_entry.path(), &sub_target);
                }
            }
            // Skip other directories — the artifact validator only needs
            // model files and the optiq vision sidecar, not .cache/refs/etc.
        } else {
            link_or_copy_file(&entry_path, &target_path);
        }
    }

    let mut validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(vision_artifact_directory.path(), 20_480)
        .expect("a vision-enabled artifact should validate with the vision sidecar");

    // The vision sidecar should be present and transferable through opaque source identities.
    let vision_sidecar_file_names = validated_artifact
        .shard_index()
        .vision_sidecar_file_names()
        .to_vec();
    let vision_sidecar_source_ids = validated_artifact
        .source_ids_for_file_names(&vision_sidecar_file_names)
        .expect("the vision sidecars should resolve to source identities");
    assert!(
        !validated_artifact
            .take_safetensors_sources(&vision_sidecar_source_ids)
            .expect("the vision sidecars should be available for transfer")
            .is_empty()
    );
}

#[test]
#[ignore = "requires model_directories to discover the Ornith 1.5 acceptance artifact"]
fn should_ignore_non_execution_files_in_the_configured_local_artifact() {
    let model_directory = crate::common::configured_large_sparse_moe_model_directory();

    Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 20_480)
        .expect("non-execution repository files should not block text-model validation");
}

#[test]
#[ignore = "requires model_directories to discover the large sparse MoE installed model"]
fn should_validate_the_large_sparse_moe_artifact_quantization_width() {
    let model_directory = crate::common::configured_large_sparse_moe_model_directory();

    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the large sparse MoE installed model should validate");

    // Shard count must be positive and consistent with the index.
    let shard_count = validated_artifact.shard_count();
    assert!(
        shard_count > 0,
        "shard count must be positive, got {shard_count}"
    );
    assert_eq!(
        shard_count,
        validated_artifact
            .shard_index()
            .model_shard_file_names()
            .len(),
        "shard count must match model shard file names"
    );
    assert!(validated_artifact.total_payload_bytes() > 0);
    if validated_artifact.has_separate_vision_sidecar() {
        assert!(
            validated_artifact.supports_image_input(),
            "a vision sidecar must advertise image input"
        );
    }
    assert!(
        [2, 3, 4, 5, 6, 8].contains(&validated_artifact.config().default_quantization_bits()),
        "the large sparse MoE installed model must declare a supported affine bit width"
    );
    assert!(
        [32u32, 64, 128].contains(
            &validated_artifact
                .config()
                .default_quantization_group_size()
        ),
        "the large sparse MoE installed model must declare a supported affine group size"
    );
}
#[test]
#[ignore = "requires model_directories to discover the large sparse MoE installed model"]
fn should_validate_the_large_sparse_moe_artifact() {
    fn expected_model_id() -> &'static str {
        crate::common::large_sparse_moe_model_id()
    }

    let model_directory = crate::serving_acceptance::support::large_sparse_moe_model_directory();

    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the large sparse MoE installed model should validate");

    // Structural validity: positive shard count, positive payload, architecture from config.
    assert!(validated_artifact.shard_count() > 0);
    assert!(validated_artifact.total_payload_bytes() > 0);
    assert_eq!(validated_artifact.model_id(), expected_model_id());
    assert_eq!(
        validated_artifact.shard_count(),
        validated_artifact
            .shard_index()
            .model_shard_file_names()
            .len()
    );
    assert!(validated_artifact.config().layer_count() > 0);
    assert!(validated_artifact.config().expert_count() > 0);
    assert!(validated_artifact.config().experts_per_token() > 0);
    assert!(
        [2, 3, 4, 5, 6, 8].contains(&validated_artifact.config().default_quantization_bits()),
        "default affine bit width must be a supported MLX width"
    );
    assert!(
        [32u32, 64, 128].contains(
            &validated_artifact
                .config()
                .default_quantization_group_size()
        ),
        "default affine group size must be a supported MLX group size"
    );
    if validated_artifact.has_separate_vision_sidecar() {
        assert!(validated_artifact.supports_image_input());
    }
}

#[test]
#[ignore = "requires model_directories to discover a complete supported depth-one MTP artifact"]
fn should_validate_a_configured_depth_one_mtp_artifact() {
    let model_directory =
        crate::serving_acceptance::support::configured_depth_one_mtp_model_directory();

    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the discovered depth-one MTP artifact should validate");

    let mtp_layer_count = validated_artifact.config().mtp_layer_count();
    if validated_artifact
        .mtp_artifact_capability()
        .is_mtp_capable()
    {
        assert!(
            mtp_layer_count >= 1,
            "an MTP-capable inventory must declare at least one MTP layer in config"
        );
    }
}

#[cfg(feature = "direct-mlx")]
#[tokio::test]
#[ignore = "loads the configured target and smallest compatible SpecPrefill draft artifacts"]
async fn should_stop_when_the_configured_drafter_prefix_restore_fails() {
    use std::time::Duration;

    use astronomical_ipc_protocol::{
        RequestId, SpeculativePrefillRuntimeState, WorkerSpeculativePrefillConfiguration,
    };
    use tokio::time::timeout;

    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let target_model_directory_path = crate::common::configured_large_sparse_moe_model_directory();
    let (draft_model_directory_path, configured_draft_model_id) =
        crate::serving_acceptance::support::configured_speculative_prefill_draft_model(
            &target_model_directory_path,
        );
    eprintln!(
        "[speculative-prefill-artifact] status=progress phase=draft_selected draft_model_id={configured_draft_model_id}"
    );

    eprintln!("[speculative-prefill-artifact] status=progress phase=artifact_validation");
    let validated_target_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&target_model_directory_path, 20_480)
        .expect("the configured Ornith target artifact should validate");
    let validated_draft_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&draft_model_directory_path, 20_480)
        .expect("the configured Ornith draft artifact should validate");
    assert_eq!(
        Qwen3_5Tokenizer::token_identifier_mapping_digest(
            validated_target_artifact
                .tokenizer_bytes()
                .expect("the target should retain tokenizer bytes"),
        )
        .expect("the target tokenizer mapping should digest"),
        Qwen3_5Tokenizer::token_identifier_mapping_digest(
            validated_draft_artifact
                .tokenizer_bytes()
                .expect("the draft should retain tokenizer bytes"),
        )
        .expect("the draft tokenizer mapping should digest"),
        "the target and draft must share one token-to-identifier mapping"
    );
    assert_eq!(
        validated_draft_artifact
            .config()
            .feed_forward_architecture(),
        Qwen3_5FeedForwardArchitecture::Dense
    );
    let draft_model_revision = validated_draft_artifact.revision().to_owned();
    let target_tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_target_artifact)
        .expect("the configured Ornith target tokenizer should initialize");
    let target_think_end_token_id = target_tokenizer.think_end_token_id();
    let target_image_pad_token_id = target_tokenizer.image_pad_token_id();
    let mlx_memory_limits = crate::common::sample_serving_acceptance_mlx_memory_limits().await;
    let performance_attribution_directory = tempfile::tempdir()
        .expect("the acceptance should create a performance-attribution directory");
    let performance_attribution_log_path = performance_attribution_directory
        .path()
        .join("performance-attribution.jsonl");
    let target_model_id = validated_target_artifact.model_id().to_owned();
    let speculative_prefill_configuration = WorkerSpeculativePrefillConfiguration {
        enabled: true,
        target_model_id: Some(target_model_id),
        draft_model_id: Some(configured_draft_model_id.clone()),
        draft_model_directory: Some(draft_model_directory_path.clone()),
        minimum_prompt_tokens: 8,
        keep_percentage: 50,
        selection_chunck_token_count: 4,
        mandatory_trailing_token_count: 4,
        lookahead_token_count: 2,
        importance_pooling_kernel_token_count: 3,
    };
    let mut qwen3_5_engine = Qwen3_5Engine::new_with_runtime_chunking_and_speculative_prefill_and_performance_attribution(
        validated_target_artifact,
        mlx_memory_limits.active_memory_limit_bytes(),
        mlx_memory_limits.allocator_cache_memory_limit_bytes(),
        None,
        Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(16)
            .expect("the acceptance prefill chunk size should be valid"),
        target_think_end_token_id,
        target_model_directory_path,
        crate::common::standard_worker_chunking_configuration(),
        true,
        true,
        speculative_prefill_configuration,
        PerformanceAttribution::enabled(),
        PerformanceAttributionLog::open(&performance_attribution_log_path, true)
            .expect("the acceptance performance-attribution log should open"),
    )
    .expect("the target and draft engine settings should be valid");

    timeout(Duration::from_secs(115), async {
        eprintln!("[speculative-prefill-artifact] status=progress phase=model_load");
        let engine_load_result = qwen3_5_engine
            .load()
            .await
            .expect("the target and compatible draft language model should load together");
        assert_eq!(
            engine_load_result.speculative_prefill_runtime_state(),
            SpeculativePrefillRuntimeState::Active
        );
        assert_eq!(
            engine_load_result.speculative_prefill_draft_model_id(),
            Some(configured_draft_model_id.as_str())
        );
        assert_eq!(
            engine_load_result.speculative_prefill_draft_model_revision(),
            Some(draft_model_revision.as_str())
        );

        eprintln!("[speculative-prefill-artifact] status=progress phase=speculative_generation");
        let request_id = RequestId::new(95_001);
        qwen3_5_engine
            .start_generation(
                Qwen3_5InferenceRequest::new(request_id, std::iter::repeat_n(846, 32).collect(), 1)
                    .with_image_pad_token_id(target_image_pad_token_id)
                    .with_performance_attribution(PerformanceAttribution::enabled()),
            )
            .await
            .expect("the target should accept a short speculative-prefill request");
        qwen3_5_engine
            .force_next_speculative_prefill_draft_prefix_restore_failure_for_tests(request_id)
            .await
            .expect("the acceptance should arm one draft-prefix cache miss");
        let generation_error = loop {
            match qwen3_5_engine.decode_next_token(request_id).await {
                Err(generation_error) => break generation_error,
                Ok(astronomical_model_serving::GeneratedToken::PrefillProgress { .. })
                | Ok(astronomical_model_serving::GeneratedToken::PromptProcessingPhaseStarted {
                    ..
                })
                | Ok(astronomical_model_serving::GeneratedToken::GenerationPreparationStarted {
                    ..
                }) => continue,
                Ok(successful_generation_event) => panic!(
                    "the armed drafter restore miss must stop before successful generation: {successful_generation_event:?}"
                ),
            }
        };
        assert!(matches!(
            generation_error,
            astronomical_model_serving::InferenceEngineError::InvalidRequest { ref reason }
                if reason.contains("drafter persistent-state restoration")
                    && reason.contains("request was stopped")
                    && reason.contains("without a target-only retry")
        ));
        eprintln!("[speculative-prefill-artifact] status=stopped_as_configured");
    })
    .await
    .expect("target and draft acceptance must finish within 115 seconds");
}
