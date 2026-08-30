#![allow(dead_code)]

use std::path::Path;

use astronomical_ipc_protocol::ChatMessage;
use astronomical_model_serving::Qwen3_5PromptRenderer;
use tokenizers::Tokenizer;

pub(crate) fn build_exact_model_prompt_content(
    model_directory: &Path,
    source_document: &str,
    instruction: &str,
    target_prompt_tokens: usize,
) -> String {
    let tokenizer = Tokenizer::from_file(model_directory.join("tokenizer.json"))
        .expect("the pinned Ornith tokenizer should load");
    let source_words = source_document.split_whitespace().collect::<Vec<_>>();
    let mut minimum_word_count = 0_usize;
    let mut maximum_word_count = source_words.len();
    while minimum_word_count < maximum_word_count {
        let candidate_word_count = (minimum_word_count + maximum_word_count).div_ceil(2);
        let candidate_content = format!(
            "{instruction}\n\n{}",
            source_words[..candidate_word_count].join(" ")
        );
        if rendered_prompt_token_count(&tokenizer, &candidate_content) <= target_prompt_tokens {
            minimum_word_count = candidate_word_count;
        } else {
            maximum_word_count = candidate_word_count - 1;
        }
    }
    let mut exact_content = format!(
        "{instruction}\n\n{}",
        source_words[..minimum_word_count].join(" ")
    );
    let rendered_token_count_before_filler =
        rendered_prompt_token_count(&tokenizer, &exact_content);
    let missing_prompt_tokens = target_prompt_tokens - rendered_token_count_before_filler;
    append_missing_single_token_fillers(&mut exact_content, missing_prompt_tokens);
    let rendered_token_count = rendered_prompt_token_count(&tokenizer, &exact_content);
    assert_eq!(rendered_token_count, target_prompt_tokens);
    exact_content
}

pub(crate) fn append_missing_single_token_fillers(
    prompt_content: &mut String,
    missing_prompt_tokens: usize,
) {
    prompt_content.reserve(missing_prompt_tokens.saturating_mul(4));
    for _missing_prompt_token in 0..missing_prompt_tokens {
        prompt_content.push_str(" the");
    }
}

fn rendered_prompt_token_count(tokenizer: &Tokenizer, content: &str) -> usize {
    let rendered_prompt = Qwen3_5PromptRenderer::render(
        &[ChatMessage::User {
            content: content.to_owned(),
            images: Vec::new(),
        }],
        &[],
        true,
        &[],
        None,
    )
    .expect("the benchmark prompt should render");
    tokenizer
        .encode(rendered_prompt, false)
        .expect("the benchmark prompt should tokenize")
        .len()
}
