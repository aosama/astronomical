use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use astronomical_model_serving::Qwen3_5Tokenizer;

pub(super) const LARGE_PREFILL_ACCEPTANCE_OUTPUT_TOKEN_COUNT: usize = 1_024;
const MINIMUM_LARGE_PREFILL_ACCEPTANCE_PROMPT_TOKEN_COUNT: usize = 16_384;
const ROMEO_AND_JULIET_SOURCE: &str = include_str!(
    "../../../../apps/inference-worker/tests/fixtures/model_metrics_5000_romeo_and_juliet_words.txt"
);

pub(super) fn representative_long_generation_prompt_token_ids(
    prompt_tokenizer: &Qwen3_5Tokenizer,
    model_id: &str,
) -> Vec<u32> {
    for source_repetition_count in 3..=8 {
        let prompt_content = format!(
            "Romeo and Juliet source material:\n\n{}\n\nWrite a detailed study guide that preserves the characters, relationships, major events, and tragic ending.",
            ROMEO_AND_JULIET_SOURCE.repeat(source_repetition_count),
        );
        let prepared_request = prompt_tokenizer
            .prepare_chat(
                &ChatGenerationCommand {
                    request_id: RequestId::new(9_000),
                    model: model_id.to_owned(),
                    messages: vec![ChatMessage::User {
                        content: prompt_content,
                        images: Vec::new(),
                    }],
                    tools: Vec::new(),
                    tool_choice: ChatToolChoice::None,
                    settings: ChatGenerationSettings {
                        max_output_tokens: LARGE_PREFILL_ACCEPTANCE_OUTPUT_TOKEN_COUNT as u16,
                        temperature_thousandths: None,
                        top_p_thousandths: None,
                        seed: None,
                        thinking_budget: Some(256),
                    },
                    qwen_thinking_channel_seed: None,
                },
                false,
            )
            .expect("the representative acceptance prompt should prepare");
        if prepared_request.input_token_ids().len()
            >= MINIMUM_LARGE_PREFILL_ACCEPTANCE_PROMPT_TOKEN_COUNT
        {
            return prepared_request.input_token_ids().to_vec();
        }
    }
    panic!("the Romeo and Juliet acceptance prompt did not reach 16,384 input tokens")
}
