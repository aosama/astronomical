//! Shared pack, tokenizer and prompt helpers for the folded ternary journeys.
//!
//! Every journey loads the real installed pack through the explicit test-run
//! configuration (never a hardcoded developer path), feeds Romeo and Juliet
//! source text through the pack's own tokenizer, and fixes the
//! prompt-processing chunk at 2,048 tokens.

use std::path::PathBuf;

/// The fixed prompt-processing chunk every tier uses, so prefill numbers are
/// reproducible across runs and comparable with the serving configuration.
pub(crate) const PREFILL_CHUNK_TOKENS: usize = 2_048;
/// The short prompt the measurement journeys read before a reply: about
/// 500 tokens of the story.
pub(crate) const SHORT_PROMPT_TOKENS: usize = 500;

/// The Romeo and Juliet source every prompt is cut from, word by word.
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_10000_romeo_and_juliet_words.txt");
/// The continuation instruction prepended to the source excerpt.
const CONTINUATION_INSTRUCTION: &str =
    "Continue the supplied Romeo and Juliet story above for approximately 500 tokens.";

/// Resolves the installed pack root, refusing to guess a developer path.
pub(crate) fn pack_directory() -> PathBuf {
    test_harness::test_run::TestRunContext::current().required_model_directory()
}

/// Resolves the upstream reference dump for the parity journey.
pub(crate) fn parity_reference_path() -> PathBuf {
    let reference_path =
        test_harness::test_run::TestRunContext::current().required_reference_artifact_path();
    assert!(
        reference_path.is_file(),
        "the parity reference artifact must be a file"
    );
    reference_path
}

/// The pack's own tokenizer.
pub(crate) fn pack_tokenizer() -> tokenizers::Tokenizer {
    tokenizers::Tokenizer::from_file(pack_directory().join("tokenizer.json"))
        .expect("the pack tokenizer should load")
}

/// Encodes one rendered prompt into model token ids.
pub(crate) fn encode_prompt(tokenizer: &tokenizers::Tokenizer, rendered_prompt: &str) -> Vec<i32> {
    tokenizer
        .encode(rendered_prompt, false)
        .expect("the rendered prompt should encode")
        .get_ids()
        .iter()
        .map(|token_id| *token_id as i32)
        .collect()
}

/// Renders the single-user-turn, thinking-disabled chat template of the pack
/// around a continuation instruction and a Romeo and Juliet excerpt.
fn rendered_continuation_prompt(word_count: usize) -> String {
    let excerpt = ROMEO_AND_JULIET_SOURCE
        .split_whitespace()
        .take(word_count)
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        "<|im_start|>user\n{CONTINUATION_INSTRUCTION}\n\n{excerpt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
}

/// Cuts the Romeo and Juliet source at the fewest words whose rendered
/// prompt encodes to at least `minimum_token_count` tokens, so a tier sends
/// at least the required prompt length without overshooting by more than a
/// word's worth of tokens.
pub(crate) fn romeo_and_juliet_prompt_token_ids(
    tokenizer: &tokenizers::Tokenizer,
    minimum_token_count: usize,
) -> Vec<i32> {
    let total_word_count = ROMEO_AND_JULIET_SOURCE.split_whitespace().count();
    let encoded_length = |word_count: usize| {
        encode_prompt(tokenizer, &rendered_continuation_prompt(word_count)).len()
    };
    assert!(
        encoded_length(total_word_count) >= minimum_token_count,
        "the Romeo and Juliet source must encode to at least {minimum_token_count} tokens"
    );
    let (mut low, mut high) = (1_usize, total_word_count);
    while low < high {
        let middle = low + (high - low) / 2;
        if encoded_length(middle) >= minimum_token_count {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    encode_prompt(tokenizer, &rendered_continuation_prompt(low))
}
