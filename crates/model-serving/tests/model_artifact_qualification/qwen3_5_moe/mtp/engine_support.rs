use std::path::{Path, PathBuf};

use astronomical_ipc_protocol::RequestId;
use astronomical_model_serving::{
    DEFAULT_FULL_ATTENTION_KV_STATE_GROWTH_TOKENS, GeneratedToken, InferenceEngine,
    PerformanceAttribution, PerformanceAttributionLog, Qwen3_5ArtifactValidator, Qwen3_5Engine,
    Qwen3_5InferenceRequest, Qwen3_5PrefillChunckSizer, Qwen3_5Tokenizer,
};

use super::IMAGE_PAD_TOKEN_ID;

pub(super) async fn load_mtp_test_engine(
    model_directory: &Path,
    mtp_enabled: bool,
) -> (Qwen3_5Engine, tempfile::TempDir, PathBuf) {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 20_480)
        .expect("the local oQ4e MTP artifact should validate before engine loading");
    let think_end_token_id = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the MTP tokenizer should expose validated control tokens")
        .think_end_token_id();
    let mlx_memory_limits =
        crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;
    let temporary_log_directory =
        tempfile::tempdir().expect("the MTP test should create an attribution directory");
    let performance_attribution_log_path = temporary_log_directory
        .path()
        .join("performance-attribution.jsonl");
    let performance_attribution_log =
        PerformanceAttributionLog::open(&performance_attribution_log_path, true)
            .expect("the MTP test should open its attribution log");
    let qwen3_5_engine = Qwen3_5Engine::new_with_prefill_chunck_sizer_and_performance_attribution(
        validated_artifact,
        mlx_memory_limits.active_memory_limit_bytes(),
        mlx_memory_limits.allocator_cache_memory_limit_bytes(),
        None,
        Qwen3_5PrefillChunckSizer::for_fixed_prefill_chunck_tokens(16)
            .expect("the MTP test prefill_chunck_tokens should be valid"),
        think_end_token_id,
        model_directory.to_path_buf(),
        DEFAULT_FULL_ATTENTION_KV_STATE_GROWTH_TOKENS,
        true,
        mtp_enabled,
        PerformanceAttribution::disabled(),
        performance_attribution_log,
    )
    .expect("the oQ4e MTP engine settings should be valid");
    (
        qwen3_5_engine,
        temporary_log_directory,
        performance_attribution_log_path,
    )
}

pub(super) async fn generate_with_mtp_engine(
    model_directory: &Path,
    output_token_count: u16,
    force_next_mtp_draft_rejection: bool,
) -> (Vec<u32>, serde_json::Value) {
    let (mut qwen3_5_engine, _temporary_log_directory, performance_attribution_log_path) =
        load_mtp_test_engine(model_directory, true).await;
    qwen3_5_engine
        .load()
        .await
        .expect("the engine should materialize the oQ4e MTP model");
    let request_id = RequestId::new(36_001);
    qwen3_5_engine
        .start_generation(
            Qwen3_5InferenceRequest::new(
                request_id,
                super::super::super::qwen3_5::SAY_HI_PROMPT_TOKEN_IDS.to_vec(),
                output_token_count,
            )
            .with_image_pad_token_id(IMAGE_PAD_TOKEN_ID)
            .with_performance_attribution(PerformanceAttribution::enabled()),
        )
        .await
        .expect("the engine should accept one MTP greedy generation request");
    if force_next_mtp_draft_rejection {
        qwen3_5_engine
            .force_next_mtp_draft_rejection_for_tests(request_id)
            .await
            .expect("the engine should arm one deterministic MTP rejection");
    }

    let mut generated_token_ids = Vec::with_capacity(output_token_count as usize);
    while generated_token_ids.len() < output_token_count as usize {
        match qwen3_5_engine
            .decode_next_token(request_id)
            .await
            .expect("each MTP engine boundary should advance the request")
        {
            GeneratedToken::TokenId { token_id, .. } => generated_token_ids.push(token_id),
            GeneratedToken::PrefillProgress { .. } => {}
            GeneratedToken::EndOfSequence => break,
        }
    }
    drop(qwen3_5_engine);
    let performance_attribution_jsonl = std::fs::read_to_string(&performance_attribution_log_path)
        .expect("the completed MTP request should write attribution");
    let performance_attribution_json = serde_json::from_str(performance_attribution_jsonl.trim())
        .expect("the MTP attribution should be valid JSON");
    (generated_token_ids, performance_attribution_json)
}

pub(super) fn performance_counter_amount(
    performance_attribution_json: &serde_json::Value,
    performance_counter_identifier: &str,
) -> u64 {
    performance_attribution_json["counters"]
        .as_array()
        .and_then(|performance_counters| {
            performance_counters.iter().find(|performance_counter| {
                performance_counter["counter"] == performance_counter_identifier
            })
        })
        .and_then(|performance_counter| performance_counter["amount"].as_u64())
        .unwrap_or(0)
}

pub(super) fn generation_report_for_request(
    performance_attribution_log_path: &Path,
    request_id: RequestId,
) -> serde_json::Value {
    std::fs::read_to_string(performance_attribution_log_path)
        .expect("the MTP attribution log should be readable")
        .lines()
        .map(|performance_attribution_line| {
            serde_json::from_str::<serde_json::Value>(performance_attribution_line)
                .expect("each MTP attribution record should be valid JSON")
        })
        .find(|performance_attribution_report| {
            performance_attribution_report["request_id"] == request_id.value()
        })
        .expect("the completed MTP request should have an attribution report")
}
