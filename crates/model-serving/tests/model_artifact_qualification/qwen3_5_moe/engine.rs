use std::time::Duration;

use astronomical_ipc_protocol::{ExpertMemoryMode, RequestId};
use astronomical_model_serving::{
    GeneratedToken, InferenceEngine, Qwen3_5ArtifactValidator, Qwen3_5Engine,
    Qwen3_5InferenceRequest, Qwen3_5PrefillChunckSizer, Qwen3_5Tokenizer,
};
use tokio::time::timeout;

use super::ORNITH_IMAGE_PAD_TOKEN_ID;

const LOCAL_AI_PROMPT_TOKEN_IDS: [u32; 20] = [
    248_045, 846, 198, 657, 799, 14_542, 8_495, 314, 2_136, 14_791, 13, 248_046, 198, 248_045,
    74_455, 198, 248_068, 271, 248_069, 271,
];
const LIVE_CONTEXT_TELEMETRY_PROMPT_TOKEN_COUNT: usize = 2_049;
const LIVE_MEMORY_LIMIT_INITIAL_BYTES: usize = 10_000_000_000;
const LIVE_MEMORY_LIMIT_RAISED_BYTES: u64 = 16_000_000_000;
const LIVE_MEMORY_LIMIT_CALIBRATION_PROMPT_TOKEN_COUNT: usize = 5_925;
const LIVE_MEMORY_LIMIT_REGRESSION_PROMPT_TOKEN_COUNT: usize = 66_224;

#[tokio::test]
#[ignore = "loads and generates with the complete pinned 22 GB Ornith artifact"]
async fn should_generate_the_certified_greedy_continuation_through_the_engine_trait() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = crate::common::configured_ornith_model_artifact_directory();
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the pinned Ornith artifact should validate before engine loading");
    let mlx_memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
    let mut qwen3_5_engine = Qwen3_5Engine::new_with_prefill_chunck_sizer(
        validated_artifact,
        mlx_memory_limits.active_memory_limit_bytes(),
        mlx_memory_limits.allocator_cache_memory_limit_bytes(),
        None,
        Qwen3_5PrefillChunckSizer::for_fixed_prefill_chunck_tokens(16)
            .expect("the test prefill_chunck_tokens should be valid"),
        ORNITH_IMAGE_PAD_TOKEN_ID,
        model_directory.to_path_buf(),
        crate::common::standard_worker_chunking_configuration(),
        false,
        crate::common::disabled_worker_speculative_prefill_configuration(),
    )
    .expect("the bounded Ornith engine settings should be valid");
    qwen3_5_engine
        .load()
        .await
        .expect("the engine should materialize the complete Ornith model");
    let request_id = RequestId::new(1_000);
    qwen3_5_engine
        .start_generation(
            Qwen3_5InferenceRequest::new(
                request_id,
                super::super::qwen3_5::SAY_HI_PROMPT_TOKEN_IDS.to_vec(),
                super::CERTIFIED_SAY_HI_GREEDY_TOKEN_COUNT,
            )
            .with_image_pad_token_id(ORNITH_IMAGE_PAD_TOKEN_ID),
        )
        .await
        .expect("the engine should accept one greedy generation request");

    let mut generated_token_ids = Vec::new();
    while generated_token_ids.len() < super::CERTIFIED_SAY_HI_GREEDY_TOKEN_IDS.len() {
        match qwen3_5_engine
            .decode_next_token(request_id)
            .await
            .expect("each engine boundary should advance the request")
        {
            GeneratedToken::TokenId { token_id, .. } => generated_token_ids.push(token_id),
            GeneratedToken::PrefillProgress { .. } => {}
            GeneratedToken::PromptProcessingPhaseStarted { .. } => {}
            GeneratedToken::EndOfSequence => break,
        }
    }

    assert_eq!(
        generated_token_ids,
        super::CERTIFIED_SAY_HI_GREEDY_TOKEN_IDS
    );
}

#[tokio::test]
#[ignore = "loads and samples with the complete pinned 22 GB Ornith artifact"]
async fn should_generate_the_certified_sampled_continuation_through_the_engine_trait() {
    let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
    let model_directory = crate::common::configured_ornith_model_artifact_directory();
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(&model_directory, 20_480)
        .expect("the pinned Ornith artifact should validate before engine loading");
    let mlx_memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
    let mut qwen3_5_engine = Qwen3_5Engine::new_with_prefill_chunck_sizer(
        validated_artifact,
        mlx_memory_limits.active_memory_limit_bytes(),
        mlx_memory_limits.allocator_cache_memory_limit_bytes(),
        None,
        Qwen3_5PrefillChunckSizer::for_fixed_prefill_chunck_tokens(16)
            .expect("the test prefill_chunck_tokens should be valid"),
        ORNITH_IMAGE_PAD_TOKEN_ID,
        model_directory.to_path_buf(),
        crate::common::standard_worker_chunking_configuration(),
        false,
        crate::common::disabled_worker_speculative_prefill_configuration(),
    )
    .expect("the bounded Ornith engine settings should be valid");
    qwen3_5_engine
        .load()
        .await
        .expect("the engine should materialize the complete Ornith model");
    let request_id = RequestId::new(1_001);
    qwen3_5_engine
        .start_generation(
            Qwen3_5InferenceRequest::new_sampling(
                request_id,
                LOCAL_AI_PROMPT_TOKEN_IDS.to_vec(),
                16,
                700,
                900,
                Some(1_234),
            )
            .with_image_pad_token_id(ORNITH_IMAGE_PAD_TOKEN_ID),
        )
        .await
        .expect("the engine should accept one deterministically sampled request");

    let mut generated_token_ids = Vec::new();
    while generated_token_ids.len() < 16 {
        match qwen3_5_engine
            .decode_next_token(request_id)
            .await
            .expect("each engine boundary should advance the sampled request")
        {
            GeneratedToken::TokenId { token_id, .. } => generated_token_ids.push(token_id),
            GeneratedToken::PrefillProgress { .. } => {}
            GeneratedToken::PromptProcessingPhaseStarted { .. } => {}
            GeneratedToken::EndOfSequence => break,
        }
    }

    assert_eq!(
        generated_token_ids,
        vec![
            3_833, 14_542, 8_495, 314, 2_136, 14_791, 369, 2_972, 38_545, 4_718, 795, 11_992, 321,
            4_610, 159_034, 271,
        ]
    );
}

#[tokio::test]
#[ignore = "loads Ornith and verifies live context telemetry without adaptive RAM growth admission"]
async fn should_report_live_context_telemetry_without_adaptive_ram_growth_guard() {
    timeout(Duration::from_secs(115), async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let model_directory = crate::common::configured_ornith_model_artifact_directory();
        let validated_artifact = Qwen3_5ArtifactValidator::new()
            .validate(&model_directory, 20_480)
            .expect("the pinned Ornith artifact should validate before engine loading");
        let mlx_memory_limits = crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
        let mut qwen3_5_engine =
            Qwen3_5Engine::new_with_runtime_chunking_and_speculative_prefill_and_performance_attribution(
                validated_artifact,
                mlx_memory_limits.active_memory_limit_bytes(),
                mlx_memory_limits.allocator_cache_memory_limit_bytes(),
                None,
                Qwen3_5PrefillChunckSizer::for_fixed_prefill_chunck_tokens(2_048)
                    .expect("the production prefill chunck size should be valid"),
                ORNITH_IMAGE_PAD_TOKEN_ID,
                model_directory,
                crate::common::standard_worker_chunking_configuration(),
                false,
                false,
                crate::common::disabled_worker_speculative_prefill_configuration(),
                astronomical_model_serving::PerformanceAttribution::disabled(),
                astronomical_model_serving::PerformanceAttributionLog::disabled(),
            )
            .expect("the disabled adaptive-guard engine settings should be valid");
        eprintln!("[live-context-telemetry 0/2] status=progress phase=model_load");
        qwen3_5_engine
            .load()
            .await
            .expect("the engine should materialize the complete Ornith model");
        let request_id = RequestId::new(1_004);
        qwen3_5_engine
            .start_generation(
                Qwen3_5InferenceRequest::new(
                    request_id,
                    vec![198; LIVE_CONTEXT_TELEMETRY_PROMPT_TOKEN_COUNT],
                    1,
                )
                .with_image_pad_token_id(ORNITH_IMAGE_PAD_TOKEN_ID),
            )
            .await
            .expect("the telemetry request should be admitted");

        eprintln!("[live-context-telemetry 1/2] status=progress phase=prefill");
        match qwen3_5_engine
            .decode_next_token(request_id)
            .await
            .expect("the first prefill chunck should complete")
        {
            GeneratedToken::PrefillProgress {
                mlx_memory_telemetry: Some(mlx_memory_telemetry),
                ..
            } => {
                assert!(
                    mlx_memory_telemetry
                        .active_memory_breakdown
                        .context_state_payload_bytes
                        > 0,
                    "an active request must report its live context payload even when adaptive admission is disabled"
                );
                eprintln!("[live-context-telemetry 2/2] status=success");
            }
            other_generation_event => panic!(
                "the first advance should report live prefill telemetry, got {other_generation_event:?}"
            ),
        }
    })
    .await
    .expect("the live-context telemetry regression must finish within 115 seconds");
}

#[tokio::test]
#[ignore = "loads a configured depth-one MTP checkpoint and verifies demand-only expert paging"]
async fn should_keep_sparse_experts_paged_even_when_idle_memory_is_sufficient() {
    timeout(Duration::from_secs(120), async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let model_directory = super::configured_depth_one_mtp_model_artifact_directory();
        assert!(
            model_directory.is_dir(),
            "the configured depth-one MTP checkpoint must be available"
        );
        eprintln!("[one-expert-cache 0/4] status=progress phase=artifact_validation");
        let validated_artifact = Qwen3_5ArtifactValidator::new()
            .validate(&model_directory, 20_480)
            .expect("the configured depth-one MTP artifact should validate");
        let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
            .expect("the tokenizer should expose validated control tokens");
        let think_end_token_id = tokenizer.think_end_token_id();
        let image_pad_token_id = tokenizer.image_pad_token_id();
        let mlx_memory_limits = crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
        let mut qwen3_5_engine = Qwen3_5Engine::new_with_runtime_chunking_and_speculative_prefill_and_performance_attribution(
            validated_artifact,
            mlx_memory_limits.active_memory_limit_bytes(),
            mlx_memory_limits.allocator_cache_memory_limit_bytes(),
            None,
            Qwen3_5PrefillChunckSizer::for_fixed_prefill_chunck_tokens(16)
                .expect("the test prefill_chunck_tokens should be valid"),
            think_end_token_id,
            model_directory,
            crate::common::standard_worker_chunking_configuration(),
            false,
            false,
            crate::common::disabled_worker_speculative_prefill_configuration(),
            astronomical_model_serving::PerformanceAttribution::disabled(),
            astronomical_model_serving::PerformanceAttributionLog::disabled(),
        )
        .expect("the one-expert-cache engine settings should be valid");

        eprintln!("[one-expert-cache 1/4] status=progress phase=model_load ETA_seconds=90");
        qwen3_5_engine
            .load()
            .await
            .expect("the one-expert-cache model should load");
        let request_id = RequestId::new(1_003);
        eprintln!("[one-expert-cache 2/4] status=progress phase=short_generation");
        qwen3_5_engine
            .start_generation(
                Qwen3_5InferenceRequest::new(
                    request_id,
                    super::super::qwen3_5::SAY_HI_PROMPT_TOKEN_IDS.to_vec(),
                    2,
                )
                .with_image_pad_token_id(image_pad_token_id),
            )
            .await
            .expect("the demand-paged model should accept a short request");
        loop {
            match qwen3_5_engine
                .decode_next_token(request_id)
                .await
                .expect("the demand-paged model should generate through one-expert pages")
            {
                GeneratedToken::PrefillProgress { .. } => {}
                GeneratedToken::PromptProcessingPhaseStarted { .. } => {}
                GeneratedToken::TokenId { .. } | GeneratedToken::EndOfSequence => break,
            }
        }
        let generation_finalization = qwen3_5_engine
            .cancel_generation(request_id)
            .await
            .expect("the one-expert-cache request should finalize cleanly");

        eprintln!("[one-expert-cache 3/4] status=progress phase=paging_assertion");
        assert_eq!(
            generation_finalization.expert_memory_mode(),
            Some(ExpertMemoryMode::Paged),
            "one-expert-only caching must never promote unrouted experts into complete-layer residency"
        );
        eprintln!("[one-expert-cache 4/4] status=success");
    })
    .await
    .expect("the one-expert-cache regression must finish within 120 seconds");
}

#[tokio::test]
#[ignore = "loads a configured MTP model and exercises a live memory-limit increase during real prefill"]
async fn should_use_the_raised_live_memory_limit_for_adaptive_expert_eviction() {
    timeout(Duration::from_secs(120), async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let model_directory = super::configured_depth_one_mtp_model_artifact_directory();
        let validated_artifact = Qwen3_5ArtifactValidator::new()
            .validate(&model_directory, 20_480)
            .expect("the configured depth-one MTP artifact should validate");
        let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
            .expect("the tokenizer should expose validated control tokens");
        let think_end_token_id = tokenizer.think_end_token_id();
        let image_pad_token_id = tokenizer.image_pad_token_id();
        let mut qwen3_5_engine =
            Qwen3_5Engine::new_with_runtime_chunking_and_speculative_prefill_and_performance_attribution(
                validated_artifact,
                LIVE_MEMORY_LIMIT_INITIAL_BYTES,
                LIVE_MEMORY_LIMIT_INITIAL_BYTES,
                None,
                Qwen3_5PrefillChunckSizer::for_fixed_prefill_chunck_tokens(2_048)
                    .expect("the production prefill chunck size should be valid"),
                think_end_token_id,
                model_directory,
                crate::common::standard_worker_chunking_configuration(),
                true,
                false,
                crate::common::disabled_worker_speculative_prefill_configuration(),
                astronomical_model_serving::PerformanceAttribution::disabled(),
                astronomical_model_serving::PerformanceAttributionLog::disabled(),
            )
            .expect("the live memory-limit regression engine settings should be valid");

        eprintln!("[live-memory-limit-eviction 0/4] status=progress phase=model_load");
        qwen3_5_engine
            .load()
            .await
            .expect("the configured MTP model should load under the initial memory limit");
        let calibration_request_id = RequestId::new(1_005);
        qwen3_5_engine
            .start_generation(
                Qwen3_5InferenceRequest::new(
                    calibration_request_id,
                    vec![198; LIVE_MEMORY_LIMIT_CALIBRATION_PROMPT_TOKEN_COUNT],
                    1,
                )
                .with_image_pad_token_id(image_pad_token_id),
            )
            .await
            .expect("the calibration request should be admitted");
        while matches!(
            qwen3_5_engine
                .decode_next_token(calibration_request_id)
                .await
                .expect("the calibration request should measure prefill growth"),
            GeneratedToken::PrefillProgress { .. }
        ) {}
        qwen3_5_engine
            .update_mlx_memory_limit(LIVE_MEMORY_LIMIT_RAISED_BYTES)
            .await
            .expect("the idle engine should accept the raised live memory limit");

        let request_id = RequestId::new(1_006);
        qwen3_5_engine
            .start_generation(
                Qwen3_5InferenceRequest::new(
                    request_id,
                    vec![198; LIVE_MEMORY_LIMIT_REGRESSION_PROMPT_TOKEN_COUNT],
                    1,
                )
                .with_image_pad_token_id(image_pad_token_id),
            )
            .await
            .expect("the long request should fit after expert reclamation");
        let retained_expert_payload_bytes_after_request_admission = qwen3_5_engine
            .collect_mlx_memory_telemetry()
            .await
            .expect("request-admission memory telemetry should be available")
            .expect("the loaded model should report request-admission memory telemetry")
            .active_memory_breakdown
            .expert_payload_bytes;

        eprintln!("[live-memory-limit-eviction 2/4] status=progress phase=prefill");
        for _completed_prefill_chunck_count in 0..8 {
            assert!(
                matches!(
                    qwen3_5_engine
                        .decode_next_token(request_id)
                        .await
                        .expect("adaptive prefill should use the raised live memory limit"),
                    GeneratedToken::PrefillProgress { .. }
                ),
                "the 66,224-token request should still be prefilling"
            );
        }
        let retained_expert_payload_bytes_after_adaptive_prefill = qwen3_5_engine
            .collect_mlx_memory_telemetry()
            .await
            .expect("adaptive-prefill memory telemetry should be available")
            .expect("the loaded model should report adaptive-prefill memory telemetry")
            .active_memory_breakdown
            .expert_payload_bytes;

        assert_eq!(
            retained_expert_payload_bytes_after_adaptive_prefill,
            retained_expert_payload_bytes_after_request_admission,
            "adaptive prefill must not evict experts against the obsolete 10 GB ceiling after the live ceiling rises to 16 GB"
        );
        eprintln!("[live-memory-limit-eviction 4/4] status=success");
    })
    .await
    .expect("the live memory-limit eviction regression must finish within 120 seconds");
}
