use std::path::Path;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use astronomical_model_serving::{Qwen3_5ArtifactValidator, Qwen3_5Tokenizer};

const REPRESENTATIVE_SOURCE_TEXT: &str = include_str!(
    "../../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

pub(crate) fn prepare_romeo_and_juliet_three_paragraph_summary_prompt(
    model_directory: &Path,
    target_model_id: &str,
    request_id: RequestId,
    required_prompt_token_count: usize,
    maximum_output_token_count: u16,
) -> Vec<u32> {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, u32::from(maximum_output_token_count))
        .expect("the configured summary target artifact should validate");
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .expect("the configured summary tokenizer should load");
    assert!(
        required_prompt_token_count < validated_artifact.config().maximum_position_count() as usize,
        "the summary prompt must remain below the validated model context limit"
    );
    let mut repeated_source_material = String::new();
    let prepared_chat_request = loop {
        if !repeated_source_material.is_empty() {
            repeated_source_material.push_str("\n\n");
        }
        repeated_source_material.push_str(REPRESENTATIVE_SOURCE_TEXT);
        let prepared_chat_request = tokenizer
            .prepare_chat(
                &ChatGenerationCommand {
                    request_id,
                    model: target_model_id.to_owned(),
                    messages: vec![ChatMessage::User {
                        content: format!(
                            "Summarize the supplied Romeo and Juliet source in exactly three concise prose paragraphs. Do not use a heading, bullets, or a numbered list. Preserve the central conflict, the major decisions, and the tragic outcome.\n\nSource material:\n{repeated_source_material}"
                        ),
                        images: Vec::new(),
                    }],
                    tools: Vec::new(),
                    tool_choice: ChatToolChoice::None,
                    settings: ChatGenerationSettings {
                        max_output_tokens: maximum_output_token_count,
                        temperature_thousandths: Some(1_000),
                        top_p_thousandths: Some(1_000),
                        seed: None,
                        thinking_budget: None,
                    },
                    structured_generation: None,
                },
                false,
            )
            .expect("the configured summary prompt should prepare");
        if prepared_chat_request.input_token_ids().len() >= required_prompt_token_count {
            break prepared_chat_request;
        }
    };
    let complete_prompt_token_ids = prepared_chat_request.input_token_ids();
    let assistant_suffix_start_token_index = complete_prompt_token_ids
        .iter()
        .rposition(|token_id| *token_id == tokenizer.im_end_token_id())
        .expect("the summary prompt should contain the assistant suffix marker");
    let assistant_suffix_token_ids =
        &complete_prompt_token_ids[assistant_suffix_start_token_index..];
    let retained_source_prefix_token_count = required_prompt_token_count
        .checked_sub(assistant_suffix_token_ids.len())
        .expect("the assistant suffix must fit the requested summary prompt length");
    assert!(retained_source_prefix_token_count < assistant_suffix_start_token_index);
    let mut prompt_token_ids =
        complete_prompt_token_ids[..retained_source_prefix_token_count].to_vec();
    prompt_token_ids.extend_from_slice(assistant_suffix_token_ids);
    assert_eq!(prompt_token_ids.len(), required_prompt_token_count);
    prompt_token_ids
}
