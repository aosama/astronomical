use std::path::{Path, PathBuf};

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use astronomical_model_serving::{
    DEFAULT_FULL_ATTENTION_KV_STATE_GROWTH_TOKENS, GeneratedToken, InferenceEngine,
    PerformanceAttribution, PerformanceAttributionLog, Qwen3_5ArtifactValidator, Qwen3_5Engine,
    Qwen3_5InferenceRequest, Qwen3_5PrefillChunckSizer, Qwen3_5Tokenizer,
};

pub(super) async fn load_mtp_test_engine(
    model_directory: &Path,
    mtp_enabled: bool,
    attribute_model_loading: bool,
) -> (Qwen3_5Engine, tempfile::TempDir, PathBuf) {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 20_480)
        .expect("the configured MTP artifact should validate before engine loading");
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
        if attribute_model_loading {
            PerformanceAttribution::enabled()
        } else {
            PerformanceAttribution::disabled()
        },
        performance_attribution_log,
    )
    .expect("the configured MTP engine settings should be valid");
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
    let configured_mtp_artifact_test_inputs = configured_mtp_artifact_test_inputs(model_directory);
    let (mut qwen3_5_engine, _temporary_log_directory, performance_attribution_log_path) =
        load_mtp_test_engine(model_directory, true, false).await;
    qwen3_5_engine
        .load()
        .await
        .expect("the engine should materialize the configured MTP model");
    let request_id = RequestId::new(36_001);
    qwen3_5_engine
        .start_generation(
            Qwen3_5InferenceRequest::new(
                request_id,
                configured_mtp_artifact_test_inputs.short_prompt_token_ids,
                output_token_count,
            )
            .with_image_pad_token_id(configured_mtp_artifact_test_inputs.image_pad_token_id)
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
            GeneratedToken::TokenId {
                token_id,
                generation_finalization,
                ..
            } => {
                generated_token_ids.push(token_id);
                if generation_finalization.is_some() {
                    break;
                }
            }
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

pub(super) struct ConfiguredMtpArtifactTestInputs {
    pub(super) short_prompt_token_ids: Vec<u32>,
    pub(super) injected_feedback_token_ids: Vec<u32>,
    pub(super) image_pad_token_id: u32,
    pub(super) end_of_sequence_token_ids: Vec<u32>,
}

pub(super) fn configured_mtp_artifact_test_inputs(
    model_directory: &Path,
) -> ConfiguredMtpArtifactTestInputs {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 20_480)
        .expect("the configured MTP artifact should validate before preparing test input");
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the configured MTP tokenizer should prepare artifact-compatible test input");
    let short_prompt_request = tokenizer
        .prepare_chat(
            &ChatGenerationCommand {
                request_id: RequestId::new(36_000),
                model: validated_artifact.model_id().to_owned(),
                messages: vec![ChatMessage::User {
                    content: "Say hi.".to_owned(),
                    images: Vec::new(),
                }],
                tools: Vec::new(),
                tool_choice: ChatToolChoice::None,
                settings: ChatGenerationSettings {
                    max_output_tokens: 128,
                    temperature_thousandths: Some(0),
                    top_p_thousandths: Some(1_000),
                    seed: None,
                    thinking_budget: None,
                },
            },
            false,
        )
        .expect("the configured MTP tokenizer should prepare the short qualification prompt");
    let injected_feedback_token_ids = tokenizer
        .encode_model_visible_correction("Continue with the corrected context.", false)
        .expect("the configured MTP tokenizer should encode injected model feedback");
    ConfiguredMtpArtifactTestInputs {
        short_prompt_token_ids: short_prompt_request.input_token_ids().to_vec(),
        injected_feedback_token_ids,
        image_pad_token_id: tokenizer.image_pad_token_id(),
        end_of_sequence_token_ids: validated_artifact
            .config()
            .end_of_sequence_token_ids()
            .to_vec(),
    }
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

pub(super) fn performance_operation_occurrence_count(
    performance_attribution_json: &serde_json::Value,
    performance_operation_identifier: &str,
) -> u64 {
    performance_attribution_json["operations"]
        .as_array()
        .and_then(|performance_operations| {
            performance_operations.iter().find(|performance_operation| {
                performance_operation["operation"] == performance_operation_identifier
            })
        })
        .and_then(|performance_operation| performance_operation["occurrence_count"].as_u64())
        .unwrap_or(0)
}

pub(super) fn assert_terminal_only_speculative_prefill_attribution(
    generation_report: &serde_json::Value,
    mtp_enabled: bool,
    completed_prefill_chunck_tokens: &[usize],
) {
    let counter_identifiers = [
        "speculative_prefill_target_only_prefix_chunck_count",
        "speculative_prefill_target_only_prefix_token_count",
        "speculative_prefill_terminal_capture_chunck_count",
        "speculative_prefill_terminal_mtp_history_token_count",
    ];
    if !mtp_enabled {
        let serialized_counters = generation_report["counters"]
            .as_array()
            .expect("the target-only report should contain counters");
        for counter_identifier in counter_identifiers {
            assert!(
                serialized_counters.iter().all(|serialized_counter| {
                    serialized_counter["counter"] != counter_identifier
                }),
                "target-only generation must omit zero speculative-prefill counters",
            );
        }
        assert_eq!(
            performance_operation_occurrence_count(
                generation_report,
                "mtp_prompt_history_initialization_span",
            ),
            0,
        );
        return;
    }

    let (terminal_mtp_history_token_count, target_only_prefix_chuncks) =
        completed_prefill_chunck_tokens
            .split_last()
            .expect("the representative prompt should complete at least one prefill chunk");
    let target_only_prefix_token_count = target_only_prefix_chuncks.iter().sum::<usize>();
    for (counter_identifier, expected_amount) in [
        (counter_identifiers[0], target_only_prefix_chuncks.len()),
        (counter_identifiers[1], target_only_prefix_token_count),
        (counter_identifiers[2], 1),
        (counter_identifiers[3], *terminal_mtp_history_token_count),
    ] {
        assert_eq!(
            performance_counter_amount(generation_report, counter_identifier),
            expected_amount as u64,
            "the speculative-prefill counter must match completed production chunks",
        );
    }
    assert_eq!(
        performance_operation_occurrence_count(
            generation_report,
            "mtp_prompt_history_initialization_span",
        ),
        1,
        "MTP prompt history must initialize exactly once regardless of prefix chunk count",
    );
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
