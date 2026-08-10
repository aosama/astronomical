use std::path::Path;

use astronomical_ipc_protocol::{
    ChatGenerationCommand, ChatGenerationSettings, ChatMessage, ChatToolChoice, RequestId,
};
use astronomical_model_serving::{
    Qwen3_5ArtifactValidator, Qwen3_5PromptRenderer, Qwen3_5Tokenizer,
};

pub(crate) fn prepare_reproduced_long_prompt_token_ids_for_model(
    model_directory: &Path,
    model_id: &str,
    prompt_token_count: usize,
    maximum_output_token_count: u16,
) -> Result<Vec<u32>, ExactModelPromptError> {
    let validated_artifact = Qwen3_5ArtifactValidator::new()
        .validate(model_directory, 20_480)
        .map_err(|source| ExactModelPromptError::operation("artifact validation failed", source))?;
    let tokenizer = Qwen3_5Tokenizer::from_validated_artifact(&validated_artifact)
        .map_err(|source| ExactModelPromptError::operation("tokenizer loading failed", source))?;
    let mut deterministic_prompt_text = format!(
        "Write a detailed technical report of at least {maximum_output_token_count} tokens that synthesizes the following benchmark context without omitting operational constraints. "
    );
    let deterministic_long_prompt_segment_count = prompt_token_count.div_ceil(16);
    for prompt_segment_index in 0..deterministic_long_prompt_segment_count {
        deterministic_prompt_text.push_str(&format!(
            "Segment {prompt_segment_index:04}: Measure production paged mixture-of-experts prompt processing on Apple silicon with direct layer expert pages, dense-or-compact page selection, exact routed-index copying, memory-budget fallback, and reproducible throughput evidence. "
        ));
    }
    let chat_generation_command = ChatGenerationCommand {
        request_id: RequestId::new(9_048),
        model: model_id.to_owned(),
        messages: vec![ChatMessage::User {
            content: deterministic_prompt_text,
            images: Vec::new(),
        }],
        tools: Vec::new(),
        tool_choice: ChatToolChoice::None,
        settings: ChatGenerationSettings {
            max_output_tokens: maximum_output_token_count,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: None,
            thinking_budget: Some(256),
        },
    };
    let rendered_prompt = Qwen3_5PromptRenderer::render(
        &chat_generation_command.messages,
        &chat_generation_command.tools,
        true,
        &[],
    )
    .map_err(|source| ExactModelPromptError::operation("chat prompt rendering failed", source))?;
    let mut prepared_prompt_token_ids = tokenizer
        .encode_prompt(&rendered_prompt)
        .map_err(|source| ExactModelPromptError::operation("prompt tokenization failed", source))?;
    if prepared_prompt_token_ids.len() < prompt_token_count {
        return Err(ExactModelPromptError::Contract(format!(
            "deterministic prompt prepared {} tokens, fewer than the required {prompt_token_count}",
            prepared_prompt_token_ids.len()
        )));
    }
    let chat_template_suffix_start_index = prepared_prompt_token_ids
        .iter()
        .rposition(|token_id| *token_id == tokenizer.im_end_token_id())
        .ok_or_else(|| {
            ExactModelPromptError::Contract(
                "prepared chat prompt has no final user-message terminator".to_owned(),
            )
        })?;
    let chat_template_suffix_token_ids =
        prepared_prompt_token_ids.split_off(chat_template_suffix_start_index);
    let retained_prompt_prefix_token_count = prompt_token_count
        .checked_sub(chat_template_suffix_token_ids.len())
        .ok_or_else(|| {
            ExactModelPromptError::Contract(
                "requested prompt length cannot contain the chat-template suffix".to_owned(),
            )
        })?;
    prepared_prompt_token_ids.truncate(retained_prompt_prefix_token_count);
    prepared_prompt_token_ids.extend(chat_template_suffix_token_ids);
    if prepared_prompt_token_ids.len() != prompt_token_count {
        return Err(ExactModelPromptError::Contract(format!(
            "prepared prompt contains {} tokens instead of {prompt_token_count}",
            prepared_prompt_token_ids.len()
        )));
    }
    Ok(prepared_prompt_token_ids)
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum ExactModelPromptError {
    #[error("{description}: {source}")]
    Operation {
        description: &'static str,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("{0}")]
    Contract(String),
}

impl ExactModelPromptError {
    fn operation(
        description: &'static str,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Operation {
            description,
            source: Box::new(source),
        }
    }
}
