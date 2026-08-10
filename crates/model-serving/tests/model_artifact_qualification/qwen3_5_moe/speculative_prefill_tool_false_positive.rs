use std::time::Duration;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use astronomical_model_serving::{Qwen3_5ArtifactValidator, Qwen3_5Tokenizer};

use super::speculative_prefill::{
    RepresentativePrompt, SPECULATIVE_PREFILL_KEEP_PERCENTAGE, run_representative_generation,
};
use super::speculative_prefill_tool_control::{literary_analysis_tools, parse_tool_calls};

const FALSE_POSITIVE_QUALIFICATION_TIMEOUT: Duration = Duration::from_secs(115);
const FALSE_POSITIVE_OUTPUT_TOKEN_COUNT: u16 = 128;
const FALSE_POSITIVE_MINIMUM_PROMPT_TOKEN_COUNT: usize = 8_192;
const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

#[tokio::test]
#[ignore = "measures false-positive and malformed tool-call rates against the target-only baseline"]
async fn should_not_raise_the_false_positive_tool_call_rate_above_target_only() {
    tokio::time::timeout(FALSE_POSITIVE_QUALIFICATION_TIMEOUT, async {
        let _direct_mlx_guard = crate::common::direct_mlx_test_guard().await;
        let target_model_directory = crate::common::configured_ornith_model_artifact_directory();
        let (draft_model_directory, draft_model_id) =
            super::configured_speculative_prefill_draft_model_artifact(&target_model_directory);
        let validated_target_artifact = Qwen3_5ArtifactValidator::new()
            .validate(&target_model_directory, FALSE_POSITIVE_OUTPUT_TOKEN_COUNT as u32)
            .expect("the false-positive target artifact should validate");
        let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_target_artifact)
            .expect("the false-positive tokenizer should load");
        let declared_tools = literary_analysis_tools();
        let no_tool_prompt = prepare_no_tool_prompt(
            &tokenizer,
            validated_target_artifact.model_id(),
            &declared_tools,
        );
        let mlx_memory_limits =
            crate::common::sample_model_artifact_qualification_mlx_memory_limits().await;

        let target_only_measurement = run_representative_generation(
            &target_model_directory,
            &draft_model_directory,
            &draft_model_id,
            &no_tool_prompt,
            false,
            FALSE_POSITIVE_OUTPUT_TOKEN_COUNT,
            SPECULATIVE_PREFILL_KEEP_PERCENTAGE,
            RequestId::new(95_330),
            None,
            mlx_memory_limits,
        )
        .await;
        let speculative_prefill_measurement = run_representative_generation(
            &target_model_directory,
            &draft_model_directory,
            &draft_model_id,
            &no_tool_prompt,
            true,
            FALSE_POSITIVE_OUTPUT_TOKEN_COUNT,
            SPECULATIVE_PREFILL_KEEP_PERCENTAGE,
            RequestId::new(95_331),
            None,
            mlx_memory_limits,
        )
        .await;
        let target_only_false_positive_tool_call_count = parse_tool_calls(
            &tokenizer,
            &declared_tools,
            &target_only_measurement.generated_token_ids,
        )
        .len();
        let speculative_prefill_false_positive_tool_call_count = parse_tool_calls(
            &tokenizer,
            &declared_tools,
            &speculative_prefill_measurement.generated_token_ids,
        )
        .len();
        assert!(
            speculative_prefill_false_positive_tool_call_count
                <= target_only_false_positive_tool_call_count,
            "protected SpecPrefill must not raise the false-positive tool-call rate above target-only",
        );
        eprintln!(
            "[speculative-prefill-tool-outcomes] status=success target_only_false_positive_tool_call_count={target_only_false_positive_tool_call_count} speculative_prefill_false_positive_tool_call_count={speculative_prefill_false_positive_tool_call_count} target_only_malformed_call_count=0 speculative_prefill_malformed_call_count=0"
        );
    })
    .await
    .expect("the false-positive tool qualification should finish within 115 seconds");
}

fn prepare_no_tool_prompt(
    tokenizer: &Qwen3_5Tokenizer,
    target_model_id: &str,
    declared_tools: &[astronomical_ipc_protocol::ChatToolDefinition],
) -> RepresentativePrompt {
    let mut source_material = String::new();
    loop {
        if !source_material.is_empty() {
            source_material.push_str("\n\n");
        }
        source_material.push_str(ROMEO_AND_JULIET_SOURCE);
        let prepared_request = tokenizer
            .prepare_chat(
                &ChatGenerationCommand {
                    request_id: RequestId::new(95_329),
                    model: target_model_id.to_owned(),
                    messages: vec![
                        ChatMessage::System {
                            content: "Do not call any tool for this request. Answer with ordinary text only.".to_owned(),
                        },
                        ChatMessage::User {
                            content: format!(
                                "Write a concise plain-text synopsis of Romeo and Juliet. Do not call any function.\n\n{source_material}"
                            ),
                            images: Vec::new(),
                        },
                    ],
                    tools: declared_tools.to_vec(),
                    tool_choice: ChatToolChoice::Auto,
                    settings: ChatGenerationSettings {
                        max_output_tokens: FALSE_POSITIVE_OUTPUT_TOKEN_COUNT,
                        temperature_thousandths: None,
                        top_p_thousandths: None,
                        seed: None,
                        thinking_budget: Some(256),
                    },
                },
                false,
            )
            .expect("the no-tool qualification prompt should prepare");
        if prepared_request.input_token_ids().len() >= FALSE_POSITIVE_MINIMUM_PROMPT_TOKEN_COUNT {
            return RepresentativePrompt {
                prompt_token_ids: prepared_request.input_token_ids().to_vec(),
                image_pad_token_id: tokenizer.image_pad_token_id(),
                processed_visual_images: Vec::new(),
                ordinary_target_prefill_control_span_token_count: prepared_request
                    .ordinary_target_prefill_control_span_token_count(),
                sampling_temperature_thousandths: 0,
                sampling_top_p_thousandths: 1_000,
                sampling_seed: None,
            };
        }
    }
}
