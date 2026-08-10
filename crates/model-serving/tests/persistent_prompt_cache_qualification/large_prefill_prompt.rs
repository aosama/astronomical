use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use astronomical_model_serving::Qwen3_5Tokenizer;

pub(super) const LARGE_PREFILL_QUALIFICATION_OUTPUT_TOKEN_COUNT: usize = 1_024;
const MINIMUM_LARGE_PREFILL_QUALIFICATION_PROMPT_TOKEN_COUNT: usize = 16_384;

pub(super) fn representative_long_generation_prompt_token_ids(
    prompt_tokenizer: &Qwen3_5Tokenizer,
    model_id: &str,
) -> Vec<u32> {
    let public_domain_context_sentence = "The observer records the changing sky, compares each measurement, and explains the evidence. ";
    for context_sentence_repetition_count in (1_024..=4_096).step_by(256) {
        let prompt_content = format!(
            "Background notes:\n\n{}\n\nWrite a numbered study guide with at least 2,000 distinct entries. Each entry must contain one complete explanatory sentence. Continue until every entry is present and do not conclude early.",
            public_domain_context_sentence.repeat(context_sentence_repetition_count),
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
                        max_output_tokens: LARGE_PREFILL_QUALIFICATION_OUTPUT_TOKEN_COUNT as u16,
                        temperature_thousandths: None,
                        top_p_thousandths: None,
                        seed: None,
                        thinking_budget: Some(256),
                    },
                },
                false,
            )
            .expect("the representative qualification prompt should prepare");
        if prepared_request.input_token_ids().len()
            >= MINIMUM_LARGE_PREFILL_QUALIFICATION_PROMPT_TOKEN_COUNT
        {
            return prepared_request.input_token_ids().to_vec();
        }
    }
    panic!("the representative qualification prompt did not reach 16,384 input tokens")
}
