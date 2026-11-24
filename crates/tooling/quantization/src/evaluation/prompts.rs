//! The prompts a quant is evaluated on: the bundled calibration mix, so the
//! evaluation reflects the domain mix the quantizer was tuned for.

use crate::calibration::mix::BUNDLED_OPTIQ_MIX_JSONL;
use crate::calibration::{CalibrationMix, CalibrationSample, CalibrationTokenizer};
use crate::error::QuantizationError;
use crate::sensitivity::CalibrationForwardSample;

#[cfg(test)]
#[path = "prompts_tests.rs"]
mod prompts_tests;

/// OptiQ's `evaluate_kl` defaults.
pub const DEFAULT_EVALUATION_PROMPT_COUNT: usize = 64;
pub const DEFAULT_EVALUATION_SEQUENCE_LENGTH_TOKENS: usize = 256;

/// The first `prompt_count` samples of the bundled mix as flat text, the
/// mix repeated when it is shorter than the request.
pub fn bundled_evaluation_prompts(prompt_count: usize) -> Result<Vec<String>, QuantizationError> {
    let samples = CalibrationMix::parse_jsonl(BUNDLED_OPTIQ_MIX_JSONL)?;
    Ok(evaluation_prompts_from(&samples, prompt_count))
}

/// OptiQ's `_load_default_prompts`: each sample as flat text (a chat as
/// `[role]: content` turns, not through the chat template: the question is
/// whether the candidate drifts on this content, not whether the template
/// renders), the first `prompt_count` of them, cycling when there are fewer.
#[must_use]
pub fn evaluation_prompts_from(samples: &[CalibrationSample], prompt_count: usize) -> Vec<String> {
    let mut rendered_prompts = Vec::new();
    for sample in samples {
        // OptiQ prefers a sample's raw text here, where calibration prefers
        // its messages.
        let rendered_prompt = match &sample.text {
            Some(text) => Some(text.clone()),
            None => sample.render_text(),
        };
        if let Some(rendered_prompt) = rendered_prompt {
            rendered_prompts.push(rendered_prompt);
        }
    }
    let mut evaluation_prompts = Vec::with_capacity(prompt_count);
    if rendered_prompts.is_empty() {
        return evaluation_prompts;
    }
    for prompt_index in 0..prompt_count {
        evaluation_prompts.push(rendered_prompts[prompt_index % rendered_prompts.len()].clone());
    }
    evaluation_prompts
}

/// Encodes each prompt without special tokens and keeps its first
/// `sequence_length_tokens` tokens. A prompt that encodes to nothing is
/// dropped: there is no position to measure.
pub fn tokenized_evaluation_prompts(
    prompts: &[String],
    tokenizer: &dyn CalibrationTokenizer,
    sequence_length_tokens: usize,
) -> Result<Vec<CalibrationForwardSample>, QuantizationError> {
    let mut tokenized_prompts = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        let mut token_ids = tokenizer.encode(prompt)?;
        token_ids.truncate(sequence_length_tokens);
        if !token_ids.is_empty() {
            tokenized_prompts.push(CalibrationForwardSample { token_ids });
        }
    }
    Ok(tokenized_prompts)
}
