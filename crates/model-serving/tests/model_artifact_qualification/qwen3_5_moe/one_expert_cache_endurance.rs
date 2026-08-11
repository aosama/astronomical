use std::time::Duration;

use astronomical_ipc_protocol::RequestId;
use astronomical_model_serving::{
    GeneratedToken, InferenceEngine, PerformanceAttribution, PerformanceAttributionLog,
    Qwen3_5ArtifactValidator, Qwen3_5Engine, Qwen3_5InferenceRequest, Qwen3_5PrefillChunckSizer,
};
use tokio::time::timeout;

const ENDURANCE_TIMEOUT: Duration = Duration::from_secs(120);
const INPUT_TOKEN_COUNT: usize = 54_885;
const PREFILL_CHUNCK_TOKENS: u32 = 2_048;
const DETERMINISTIC_PROMPT_TOKEN_ID: u32 = 198;
const IMAGE_PAD_TOKEN_ID: u32 = 248_069;
#[tokio::test]
#[ignore = "loads Ornith-1.0-35B-8bit and verifies consecutive one-expert-cache requests"]
async fn should_serve_consecutive_large_requests_when_adaptive_growth_is_disabled() {
    timeout(ENDURANCE_TIMEOUT, async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let model_directory =
            crate::common::configured_model_directory_by_id("Ornith-1.0-35B-8bit")
                .expect("the local Ornith eight-bit checkpoint should be configured");
        let validated_artifact = Qwen3_5ArtifactValidator::new()
            .validate(&model_directory, 5_000)
            .expect("the local Ornith eight-bit artifact should validate");
        let mlx_memory_limits = crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
        let mut qwen3_5_engine =
            Qwen3_5Engine::new_with_runtime_chunking_and_speculative_prefill_and_performance_attribution(
                validated_artifact,
                mlx_memory_limits.active_memory_limit_bytes(),
                mlx_memory_limits.allocator_cache_memory_limit_bytes(),
                None,
                Qwen3_5PrefillChunckSizer::for_fixed_prefill_chunck_tokens(
                    PREFILL_CHUNCK_TOKENS,
                )
                .expect("the production prefill chunck size should be valid"),
                IMAGE_PAD_TOKEN_ID,
                model_directory,
                crate::common::standard_worker_chunking_configuration(),
                false,
                false,
                crate::common::disabled_worker_speculative_prefill_configuration(),
                PerformanceAttribution::disabled(),
                PerformanceAttributionLog::disabled(),
            )
            .expect("the one-expert-cache engine settings should be valid");

        eprintln!("[one-expert-cache-endurance 0/3] status=progress phase=model_load");
        qwen3_5_engine
            .load()
            .await
            .expect("the Ornith eight-bit model should load");
        let expert_payload_bytes_after_load = qwen3_5_engine
            .collect_mlx_memory_telemetry()
            .await
            .expect("load-time memory telemetry should be available")
            .expect("the loaded model should report memory telemetry")
            .active_memory_breakdown
            .expert_payload_bytes;
        run_large_request(&mut qwen3_5_engine, RequestId::new(41_001), 1).await;
        let expert_payload_bytes_after_first_request = qwen3_5_engine
            .collect_mlx_memory_telemetry()
            .await
            .expect("post-request memory telemetry should be available")
            .expect("the loaded model should report post-request memory telemetry")
            .active_memory_breakdown
            .expert_payload_bytes;
        assert!(
            expert_payload_bytes_after_first_request <= expert_payload_bytes_after_load,
            "without measured transient evidence, request finalization must not increase expert residency from {expert_payload_bytes_after_load} to {expert_payload_bytes_after_first_request} bytes"
        );
        run_large_request(&mut qwen3_5_engine, RequestId::new(41_002), 2).await;
        eprintln!("[one-expert-cache-endurance 3/3] status=success");
    })
    .await
    .expect("the consecutive one-expert-cache regression must finish within 120 seconds");
}

async fn run_large_request(
    qwen3_5_engine: &mut Qwen3_5Engine,
    request_id: RequestId,
    request_number: usize,
) {
    eprintln!(
        "[one-expert-cache-endurance {request_number}/3] status=progress phase=large_request input_tokens={INPUT_TOKEN_COUNT}"
    );
    qwen3_5_engine
        .start_generation(
            Qwen3_5InferenceRequest::new(
                request_id,
                vec![DETERMINISTIC_PROMPT_TOKEN_ID; INPUT_TOKEN_COUNT],
                1,
            )
            .with_image_pad_token_id(IMAGE_PAD_TOKEN_ID),
        )
        .await
        .expect("the one-expert-cache engine should admit the large request");
    loop {
        match qwen3_5_engine
            .decode_next_token(request_id)
            .await
            .expect("one-expert caching must not exhaust Metal memory during prefill")
        {
            GeneratedToken::PrefillProgress {
                processed_token_count,
                ..
            } => eprintln!(
                "[one-expert-cache-endurance {request_number}/3] status=progress phase=prefill processed_tokens={processed_token_count}"
            ),
            GeneratedToken::PromptProcessingPhaseStarted { .. } => {}
            GeneratedToken::TokenId { .. } | GeneratedToken::EndOfSequence => return,
        }
    }
}
