use std::fs;

use astronomical_model_serving::Qwen3_5ArtifactValidator;
#[cfg(feature = "direct-mlx")]
use astronomical_model_serving::{
    GeneratedToken, InferenceEngine, PerformanceAttribution, PerformanceAttributionLog,
    Qwen3_5Engine, Qwen3_5FeedForwardArchitecture, Qwen3_5InferenceRequest,
    Qwen3_5PromptProcessingChunkSizer, Qwen3_5Tokenizer,
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
#[ignore = "requires model_directories to discover the Ornith 1.5 qualification artifact"]
fn should_validate_a_text_only_artifact_without_the_vision_sidecar() {
    let model_directory = crate::common::configured_ornith_model_artifact_directory();
    let text_artifact_directory = tempfile::tempdir()
        .expect("the qualification test should create a temporary text artifact directory");
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
#[ignore = "requires model_directories to discover the Ornith 1.5 qualification artifact"]
fn should_validate_a_vision_enabled_artifact_with_the_vision_sidecar() {
    let model_directory = crate::common::configured_ornith_model_artifact_directory();
    let vision_artifact_directory = tempfile::tempdir()
        .expect("the qualification test should create a temporary vision artifact directory");
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
#[ignore = "requires model_directories to discover the Ornith 1.5 qualification artifact"]
fn should_ignore_non_execution_files_in_the_configured_local_artifact() {
    let model_directory = crate::common::configured_ornith_model_artifact_directory();

    Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 20_480)
        .expect("non-execution repository files should not block text-model validation");
}

#[test]
#[ignore = "requires model_directories to discover the Ornith 1.5 oQ6e artifact"]
fn should_validate_a_standard_six_bit_qwen3_5_artifact() {
    let model_directory = crate::common::configured_ornith_model_artifact_directory();

    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the standard MLX six-bit Ornith artifact should validate");

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
    assert!(!validated_artifact.has_separate_vision_sidecar());
    // The named 6-bit artifact declares 6-bit quantization.
    assert_eq!(validated_artifact.config().default_quantization_bits(), 6);
}

#[cfg(feature = "direct-mlx")]
#[tokio::test]
#[ignore = "loads and decodes the configured Ornith 1.5 oQ6e artifact"]
async fn should_load_and_decode_affine_six_bit_qwen3_5_moe_through_bounded_expert_paging() {
    use std::time::Duration;

    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = crate::common::configured_ornith_model_artifact_directory();
    eprintln!("affine-six-bit-paging status=progress phase=artifact_validation");
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the affine six-bit Ornith artifact should validate before engine loading");
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the affine six-bit tokenizer should load for image-pad-token resolution");
    let image_pad_token_id = tokenizer.image_pad_token_id();
    let mlx_memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
    let mut affine_six_bit_engine = Qwen3_5Engine::new_with_prompt_processing_chunk_sizer(
        validated_artifact,
        mlx_memory_limits.active_memory_limit_bytes(),
        mlx_memory_limits.allocator_cache_memory_limit_bytes(),
        None,
        Qwen3_5PromptProcessingChunkSizer::for_fixed_prompt_processing_chunk_size_tokens(2_048)
            .expect("the six-bit qualification prefill chunk size should be valid"),
        image_pad_token_id,
        model_directory,
        crate::common::standard_worker_chunking_configuration(),
        false,
        crate::common::disabled_worker_speculative_prefill_configuration(),
    )
    .expect("the affine six-bit paged engine settings should be valid");
    eprintln!("affine-six-bit-paging status=progress phase=model_load");
    tokio::time::timeout(Duration::from_secs(110), affine_six_bit_engine.load())
        .await
        .expect("affine six-bit model load must finish within 110 seconds")
        .expect("affine six-bit dense weights and paged experts should load");
    // Use tokenizer-derived control tokens for the prompt construction.
    let im_start = tokenizer.im_start_token_id();
    let im_end = tokenizer.im_end_token_id();
    let think_start = tokenizer.think_start_token_id();
    let newline_id: u32 = 198;
    let assistant_id: u32 = 74_455;
    let space_id: u32 = 271;
    affine_six_bit_engine
        .start_generation(
            Qwen3_5InferenceRequest::new(
                astronomical_ipc_protocol::RequestId::new(90_001),
                std::iter::repeat_n(846, 636)
                    .chain([
                        im_start,
                        846,
                        newline_id,
                        im_end,
                        newline_id,
                        im_start,
                        assistant_id,
                        newline_id,
                        think_start,
                        space_id,
                    ])
                    .collect(),
                1,
            )
            .with_image_pad_token_id(image_pad_token_id),
        )
        .await
        .expect("the affine six-bit paged engine should accept a short request");
    eprintln!("affine-six-bit-paging status=progress phase=decode");
    let generated_token = tokio::time::timeout(Duration::from_secs(110), async {
        loop {
            match affine_six_bit_engine
                .decode_next_token(astronomical_ipc_protocol::RequestId::new(90_001))
                .await
                .expect("affine six-bit paged decode should succeed")
            {
                GeneratedToken::PrefillProgress { .. } => continue,
                generated_token => return generated_token,
            }
        }
    })
    .await
    .expect("affine six-bit prefill and decode must finish within 110 seconds");
    assert!(matches!(
        generated_token,
        GeneratedToken::TokenId { .. } | GeneratedToken::EndOfSequence
    ));
}

#[test]
#[ignore = "requires model_directories to discover the Ornith 1.5 oQ8e artifact"]
fn should_validate_the_ornith_1_5_35b_a3b_eight_bit_artifact() {
    const EXPECTED_MODEL_ID: &str = crate::common::ORNITH_MODEL_SWAP_SOURCE_MODEL_ID;

    let model_directory = super::ornith_1_5_35b_a3b_eight_bit_model_directory();

    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the complete Ornith 1.5 35B A3B eight-bit artifact should validate");

    // Structural validity: positive shard count, positive payload, architecture from config.
    assert!(validated_artifact.shard_count() > 0);
    assert!(validated_artifact.total_payload_bytes() > 0);
    assert_eq!(validated_artifact.model_id(), EXPECTED_MODEL_ID);
    assert!(validated_artifact.config().layer_count() > 0);
    assert!(validated_artifact.config().expert_count() > 0);
    assert!(validated_artifact.config().experts_per_token() > 0);
    assert_eq!(validated_artifact.config().default_quantization_bits(), 8);
    // Group size must be a valid MLX affine group size.
    assert!(
        [32u32, 64, 128].contains(
            &validated_artifact
                .config()
                .default_quantization_group_size()
        )
    );
    assert!(validated_artifact.supports_image_input());
    assert!(!validated_artifact.has_separate_vision_sidecar());
}

#[test]
#[ignore = "requires model_directories to discover a complete supported depth-one MTP artifact"]
fn should_validate_a_configured_depth_one_mtp_artifact() {
    let model_directory = super::configured_depth_one_mtp_model_artifact_directory();

    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the discovered depth-one MTP artifact should validate");

    assert!(
        validated_artifact
            .mtp_artifact_capability()
            .is_mtp_capable(),
        "qualification requires the complete supported MTP tensor inventory"
    );
    assert_eq!(validated_artifact.config().mtp_layer_count(), 1);
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
    let target_model_directory_path = crate::common::configured_ornith_model_artifact_directory();
    let (draft_model_directory_path, configured_draft_model_id) =
        super::configured_speculative_prefill_draft_model_artifact(&target_model_directory_path);
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
    let mlx_memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
    let performance_attribution_directory = tempfile::tempdir()
        .expect("the qualification should create a performance-attribution directory");
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
            .expect("the qualification prefill chunk size should be valid"),
        target_think_end_token_id,
        target_model_directory_path,
        crate::common::standard_worker_chunking_configuration(),
        true,
        true,
        speculative_prefill_configuration,
        PerformanceAttribution::enabled(),
        PerformanceAttributionLog::open(&performance_attribution_log_path, true)
            .expect("the qualification performance-attribution log should open"),
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
            .expect("the qualification should arm one draft-prefix cache miss");
        let generation_error = qwen3_5_engine
            .decode_next_token(request_id)
            .await
            .expect_err("the configured drafter restore failure must stop the request");
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
    .expect("target and draft qualification must finish within 115 seconds");
}
