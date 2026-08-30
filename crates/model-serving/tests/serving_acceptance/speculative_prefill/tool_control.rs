use std::time::Duration;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, ChatToolDefinition,
    RequestId,
};
use astronomical_model_serving::{
    PersistentPromptCacheDiskStoreConfig, Qwen3_5ArtifactValidator, Qwen3_5OutputEvent,
    Qwen3_5RequestOutput, Qwen3_5Tokenizer, Qwen3_5ToolCall,
};

use super::{
    RepresentativePrompt, SPECULATIVE_PREFILL_KEEP_PERCENTAGE, run_representative_generation,
};

const REPRESENTATIVE_TOOL_PROMPT_TOKEN_COUNT: usize = 8_192;
pub(super) const REPRESENTATIVE_TOOL_OUTPUT_TOKEN_COUNT: u16 = 256;
const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

#[tokio::test]
#[ignore = "proves target-only and protected SpecPrefill tool-call correctness with an 8K Romeo and Juliet prompt"]
async fn should_preserve_a_schema_valid_tool_call_while_fully_prefilling_control_context() {
    tokio::time::timeout(Duration::from_secs(115), async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let target_model_directory = crate::common::configured_large_sparse_moe_model_directory();
        let (draft_model_directory, draft_model_id) =
            crate::serving_acceptance::support::configured_speculative_prefill_draft_model(&target_model_directory);
        let validated_target_artifact = Qwen3_5ArtifactValidator::new()
            .validate(&target_model_directory, REPRESENTATIVE_TOOL_OUTPUT_TOKEN_COUNT as u32)
            .expect("the target artifact should validate for tool acceptance");
        let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_target_artifact)
            .expect("the target tokenizer should load for tool acceptance");
        let declared_tools = literary_analysis_tools();
        let representative_tool_prompt = prepare_representative_tool_prompt(
            &tokenizer,
            validated_target_artifact.model_id(),
            &declared_tools,
            None,
        );
        let mlx_memory_limits =
            crate::common::sample_serving_acceptance_mlx_memory_limits().await;
        let persistent_prompt_cache_root_directory = tempfile::tempdir()
            .expect("the tool acceptance should create an empty SSD cache root");
        let persistent_prompt_cache_disk_store_config = PersistentPromptCacheDiskStoreConfig::new(
            persistent_prompt_cache_root_directory.path().join("target"),
            persistent_prompt_cache_root_directory.path().to_path_buf(),
            crate::common::configured_model_artifact_prompt_cache_maximum_size_bytes(),
        );

        eprintln!(
            "[speculative-prefill-tool-control] status=progress phase=target_only prompt_tokens={} output_tokens={}",
            representative_tool_prompt.prompt_token_ids.len(),
            REPRESENTATIVE_TOOL_OUTPUT_TOKEN_COUNT,
        );
        let target_only_measurement = run_representative_generation(
            &target_model_directory,
            &draft_model_directory,
            &draft_model_id,
            &representative_tool_prompt,
            false,
            REPRESENTATIVE_TOOL_OUTPUT_TOKEN_COUNT,
            SPECULATIVE_PREFILL_KEEP_PERCENTAGE,
            RequestId::new(95_300),
            None,
            mlx_memory_limits,
        )
        .await;
        let target_only_tool_call = parse_one_tool_call(
            &tokenizer,
            &declared_tools,
            &target_only_measurement.generated_token_ids,
        );

        eprintln!(
            "[speculative-prefill-tool-control] status=progress phase=protected_speculative_prefill prompt_tokens={} output_tokens={}",
            representative_tool_prompt.prompt_token_ids.len(),
            REPRESENTATIVE_TOOL_OUTPUT_TOKEN_COUNT,
        );
        let speculative_prefill_measurement = run_representative_generation(
            &target_model_directory,
            &draft_model_directory,
            &draft_model_id,
            &representative_tool_prompt,
            true,
            REPRESENTATIVE_TOOL_OUTPUT_TOKEN_COUNT,
            SPECULATIVE_PREFILL_KEEP_PERCENTAGE,
            RequestId::new(95_301),
            Some(persistent_prompt_cache_disk_store_config.clone()),
            mlx_memory_limits,
        )
        .await;
        let speculative_prefill_tool_call = parse_one_tool_call(
            &tokenizer,
            &declared_tools,
            &speculative_prefill_measurement.generated_token_ids,
        );

        assert_eq!(
            speculative_prefill_measurement.speculative_prefill_ordinary_control_span_token_count,
            representative_tool_prompt.ordinary_target_prefill_control_span_token_count as u64,
        );
        assert_representative_tool_prompt_selection_boundaries(
            &representative_tool_prompt,
            &speculative_prefill_measurement,
        );
        assert_eq!(
            speculative_prefill_measurement.speculative_prefill_draft_scored_suffix_token_count,
            representative_tool_prompt.prompt_token_ids.len() as u64,
            "the cold drafter must score the complete rendered prompt",
        );
        assert_schema_valid_literary_analysis_tool_call(&target_only_tool_call);
        assert_schema_valid_literary_analysis_tool_call(&speculative_prefill_tool_call);
        assert!(
            speculative_prefill_measurement
                .speculative_prefill_target_persistent_state_write_count
                >= 1
        );
        assert_eq!(
            speculative_prefill_tool_call.function_name,
            target_only_tool_call.function_name,
        );

        eprintln!(
            "[speculative-prefill-tool-control] status=progress phase=warm_process_independent_target_restore"
        );
        let warm_speculative_prefill_measurement = run_representative_generation(
            &target_model_directory,
            &draft_model_directory,
            &draft_model_id,
            &representative_tool_prompt,
            true,
            REPRESENTATIVE_TOOL_OUTPUT_TOKEN_COUNT,
            SPECULATIVE_PREFILL_KEEP_PERCENTAGE,
            RequestId::new(95_302),
            Some(persistent_prompt_cache_disk_store_config),
            mlx_memory_limits,
        )
        .await;
        let warm_speculative_prefill_tool_call = parse_one_tool_call(
            &tokenizer,
            &declared_tools,
            &warm_speculative_prefill_measurement.generated_token_ids,
        );
        eprintln!(
            "[speculative-prefill-tool-control] status=warm_reuse target_sparse_restored_tokens={} target_exact_restored_tokens={} drafter_restored_tokens={} control_tokens={}",
            warm_speculative_prefill_measurement
                .speculative_prefill_target_persistent_state_restored_token_count,
            warm_speculative_prefill_measurement
                .restored_target_persistent_prompt_cache_token_count,
            warm_speculative_prefill_measurement
                .speculative_prefill_draft_persistent_prefix_restored_token_count,
            representative_tool_prompt.ordinary_target_prefill_control_span_token_count,
        );
        assert_schema_valid_literary_analysis_tool_call(&warm_speculative_prefill_tool_call);
        assert!(
            warm_speculative_prefill_measurement
                .speculative_prefill_target_persistent_state_restored_token_count
                >= representative_tool_prompt.ordinary_target_prefill_control_span_token_count
                    as u64
        );
        assert_eq!(
            warm_speculative_prefill_measurement
                .speculative_prefill_draft_persistent_prefix_restored_token_count,
            0,
            "an exact target-state hit must bypass unnecessary drafter work",
        );
        assert_eq!(
            warm_speculative_prefill_measurement.speculative_prefill_drafter_eligible_token_count,
            0,
            "an exact target-state hit must report no drafter input that was never received",
        );
        eprintln!("[speculative-prefill-tool-control] status=success");
    })
    .await
    .expect("the protected tool-control acceptance should finish within 115 seconds");
}

pub(super) fn prepare_representative_tool_prompt(
    tokenizer: &Qwen3_5Tokenizer,
    target_model_id: &str,
    declared_tools: &[ChatToolDefinition],
    sampling_seed: Option<u64>,
) -> RepresentativePrompt {
    let mut repeated_source_material = String::new();
    let prepared_tool_request = loop {
        if !repeated_source_material.is_empty() {
            repeated_source_material.push_str("\n\n");
        }
        repeated_source_material.push_str(ROMEO_AND_JULIET_SOURCE);
        let prepared_tool_request = tokenizer
            .prepare_chat(
                &ChatGenerationCommand {
                    request_id: RequestId::new(95_299),
                    model: target_model_id.to_owned(),
                    messages: vec![
                        ChatMessage::System {
                            content: "Use the declared tool and return its required fields.".to_owned(),
                        },
                        ChatMessage::User {
                            content: format!(
                                "Call record_literary_analysis now. Record the play's central conflict and classify its outcome as tragic. Use this source material as evidence.\n\n{repeated_source_material}"
                            ),
                            images: Vec::new(),
                        },
                    ],
                    tools: declared_tools.to_vec(),
                    tool_choice: ChatToolChoice::Auto,
                    settings: ChatGenerationSettings {
                        max_output_tokens: REPRESENTATIVE_TOOL_OUTPUT_TOKEN_COUNT,
                        temperature_thousandths: None,
                        top_p_thousandths: None,
                        seed: None,
                        thinking_budget: Some(256),
                    },
                    qwen_thinking_channel_seed: None,
                },
                false,
            )
            .expect("the representative tool prompt should prepare");
        if prepared_tool_request.input_token_ids().len() >= REPRESENTATIVE_TOOL_PROMPT_TOKEN_COUNT {
            break prepared_tool_request;
        }
    };
    let complete_prompt_token_ids = prepared_tool_request.input_token_ids();
    let assistant_suffix_start = complete_prompt_token_ids
        .iter()
        .rposition(|prompt_token_id| *prompt_token_id == tokenizer.im_end_token_id())
        .expect("the tool prompt should contain the assistant suffix marker");
    let assistant_suffix_token_ids = &complete_prompt_token_ids[assistant_suffix_start..];
    let retained_prompt_prefix_token_count = REPRESENTATIVE_TOOL_PROMPT_TOKEN_COUNT
        .checked_sub(assistant_suffix_token_ids.len())
        .expect("the assistant suffix should fit the tool prompt budget");
    let ordinary_target_prefill_control_span_token_count =
        prepared_tool_request.ordinary_target_prefill_control_span_token_count();
    assert!(
        ordinary_target_prefill_control_span_token_count > 0
            && ordinary_target_prefill_control_span_token_count
                < retained_prompt_prefix_token_count
    );
    let mut prompt_token_ids =
        complete_prompt_token_ids[..retained_prompt_prefix_token_count].to_vec();
    prompt_token_ids.extend_from_slice(assistant_suffix_token_ids);
    assert_eq!(
        prompt_token_ids.len(),
        REPRESENTATIVE_TOOL_PROMPT_TOKEN_COUNT
    );
    RepresentativePrompt {
        prompt_token_ids,
        image_pad_token_id: tokenizer.image_pad_token_id(),
        processed_visual_images: Vec::new(),
        ordinary_target_prefill_control_span_token_count,
        sampling_temperature_thousandths: 1_000,
        sampling_top_p_thousandths: 1_000,
        sampling_seed,
    }
}

pub(super) fn literary_analysis_tools() -> Vec<ChatToolDefinition> {
    vec![ChatToolDefinition {
        name: "record_literary_analysis".to_owned(),
        description: Some("Record a structured literary analysis.".to_owned()),
        parameters_json: r#"{"type":"object","properties":{"central_conflict":{"type":"string"},"outcome":{"type":"string","enum":["tragic","comic"]}},"required":["central_conflict","outcome"],"additionalProperties":false}"#.to_owned(),
    }]
}

pub(super) fn parse_one_tool_call(
    tokenizer: &Qwen3_5Tokenizer,
    declared_tools: &[ChatToolDefinition],
    generated_token_ids: &[u32],
) -> Qwen3_5ToolCall {
    let tool_calls = parse_tool_calls(tokenizer, declared_tools, generated_token_ids);
    assert_eq!(
        tool_calls.len(),
        1,
        "the model should emit exactly one declared tool call"
    );
    tool_calls
        .into_iter()
        .next()
        .expect("one validated tool call should remain")
}

pub(super) fn parse_tool_calls(
    tokenizer: &Qwen3_5Tokenizer,
    declared_tools: &[ChatToolDefinition],
    generated_token_ids: &[u32],
) -> Vec<Qwen3_5ToolCall> {
    let mut request_output = Qwen3_5RequestOutput::new(tokenizer, declared_tools, false, None)
        .expect("the tool output parser should initialize");
    let mut output_events = Vec::new();
    for generated_token_id in generated_token_ids {
        output_events.extend(
            request_output
                .push_token(*generated_token_id)
                .expect("the generated tool output should parse"),
        );
    }
    output_events.extend(
        request_output
            .finish()
            .expect("the generated tool output should finish parsing"),
    );
    output_events
        .into_iter()
        .filter_map(|output_event| match output_event {
            Qwen3_5OutputEvent::ToolCall(tool_call) => Some(tool_call),
            Qwen3_5OutputEvent::ReasoningDelta(_)
            | Qwen3_5OutputEvent::TextDelta(_)
            | Qwen3_5OutputEvent::ModelVisibleCorrection { .. } => None,
        })
        .collect::<Vec<_>>()
}

pub(super) fn assert_schema_valid_literary_analysis_tool_call(tool_call: &Qwen3_5ToolCall) {
    assert_eq!(tool_call.function_name, "record_literary_analysis");
    let arguments_document = serde_json::from_str::<serde_json::Value>(&tool_call.arguments_json)
        .expect("tool arguments should be valid JSON");
    let arguments_object = arguments_document
        .as_object()
        .expect("tool arguments should be an object");
    assert_eq!(arguments_object.len(), 2);
    assert!(
        arguments_object
            .get("central_conflict")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|central_conflict| !central_conflict.trim().is_empty())
    );
    assert_eq!(
        arguments_object
            .get("outcome")
            .and_then(serde_json::Value::as_str),
        Some("tragic")
    );
}

fn assert_representative_tool_prompt_selection_boundaries(
    representative_tool_prompt: &RepresentativePrompt,
    speculative_prefill_measurement: &super::RepresentativeGenerationMeasurement,
) {
    const SELECTION_CHUNCK_TOKEN_COUNT: usize = 32;
    const MANDATORY_TRAILING_TOKEN_COUNT: usize = 512;

    let final_generation_kickoff_position = representative_tool_prompt
        .prompt_token_ids
        .len()
        .checked_sub(1)
        .expect("the tool prompt must contain its final generation-kickoff token");
    let selectable_conversation_token_count = final_generation_kickoff_position
        .checked_sub(representative_tool_prompt.ordinary_target_prefill_control_span_token_count)
        .expect("the selectable conversation must follow the complete control span");
    let selectable_conversation_chunck_count =
        selectable_conversation_token_count.div_ceil(SELECTION_CHUNCK_TOKEN_COUNT);
    let percentage_derived_conversation_chunck_budget = (selectable_conversation_chunck_count
        * SPECULATIVE_PREFILL_KEEP_PERCENTAGE as usize)
        .div_ceil(100);
    let mandatory_trailing_start_position =
        selectable_conversation_token_count.saturating_sub(MANDATORY_TRAILING_TOKEN_COUNT);
    let first_mandatory_trailing_chunck_index =
        mandatory_trailing_start_position / SELECTION_CHUNCK_TOKEN_COUNT;
    let mandatory_trailing_chunck_count =
        selectable_conversation_chunck_count.saturating_sub(first_mandatory_trailing_chunck_index);
    let retained_conversation_chunck_count =
        percentage_derived_conversation_chunck_budget.max(mandatory_trailing_chunck_count);
    let final_selectable_chunck_token_count = selectable_conversation_token_count.saturating_sub(
        selectable_conversation_chunck_count
            .saturating_sub(1)
            .saturating_mul(SELECTION_CHUNCK_TOKEN_COUNT),
    );
    let expected_selected_conversation_token_count = retained_conversation_chunck_count
        .saturating_sub(1)
        .saturating_mul(SELECTION_CHUNCK_TOKEN_COUNT)
        .saturating_add(final_selectable_chunck_token_count);
    let complete_ordered_target_positions = (0..representative_tool_prompt
        .ordinary_target_prefill_control_span_token_count)
        .chain(
            speculative_prefill_measurement
                .speculative_prefill_selected_token_positions
                .iter()
                .copied(),
        )
        .chain(std::iter::once(final_generation_kickoff_position))
        .collect::<Vec<_>>();

    assert_eq!(
        representative_tool_prompt.ordinary_target_prefill_control_span_token_count
            ..final_generation_kickoff_position,
        representative_tool_prompt.ordinary_target_prefill_control_span_token_count
            ..representative_tool_prompt.ordinary_target_prefill_control_span_token_count
                + selectable_conversation_token_count,
    );
    assert_eq!(
        speculative_prefill_measurement.speculative_prefill_selected_token_count,
        expected_selected_conversation_token_count as u64,
        "the real sparse target selection must use only the percentage-derived conversation budget plus mandatory trailing chunks",
    );
    assert_eq!(
        speculative_prefill_measurement
            .speculative_prefill_selected_token_positions
            .len(),
        expected_selected_conversation_token_count,
    );
    assert!(
        speculative_prefill_measurement
            .speculative_prefill_selected_token_positions
            .windows(2)
            .all(|adjacent_selected_positions| {
                adjacent_selected_positions[0] < adjacent_selected_positions[1]
            }),
        "selected conversation positions must remain strictly ordered",
    );
    assert!(
        speculative_prefill_measurement
            .speculative_prefill_selected_token_positions
            .iter()
            .all(|selected_conversation_position| {
                *selected_conversation_position
                    >= representative_tool_prompt.ordinary_target_prefill_control_span_token_count
                    && *selected_conversation_position < final_generation_kickoff_position
            }),
        "only conversation positions before generation kickoff may enter sparse selection",
    );
    assert_eq!(complete_ordered_target_positions[0], 0);
    assert_eq!(
        complete_ordered_target_positions
            [representative_tool_prompt.ordinary_target_prefill_control_span_token_count - 1],
        representative_tool_prompt.ordinary_target_prefill_control_span_token_count - 1,
    );
    assert_eq!(
        complete_ordered_target_positions.last().copied(),
        Some(final_generation_kickoff_position),
    );
    assert!(
        complete_ordered_target_positions
            .windows(2)
            .all(|adjacent_target_positions| {
                adjacent_target_positions[0] < adjacent_target_positions[1]
            }),
        "complete target positions must contain the dense control span, selected conversation, and final kickoff exactly once in order",
    );
    assert!(
        speculative_prefill_measurement.speculative_prefill_selected_token_count
            < selectable_conversation_token_count as u64,
        "the real acceptance must exercise sparse rather than full conversation retention",
    );
}
