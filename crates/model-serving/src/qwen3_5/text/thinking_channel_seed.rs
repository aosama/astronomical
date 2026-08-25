//! Seeds optional reasoning text after a Qwen3.5 `<think>` open.
//!
//! The seed is untrusted user Markdown. It is escaped with the same reserved-marker
//! rules as other template content so a seeded `</think>` cannot close the block.

use super::template_safe_content::append_template_safe_content;
use super::{Qwen3_5PromptRenderer, Qwen3_5Tokenizer, Qwen3_5TokenizerError};

const THINK_END: &str = "</think>";
const THINK_START: &str = "<think>";

/// Escapes a nonempty thinking-channel seed before it enters the model template.
#[must_use]
pub(crate) fn escaped_qwen_thinking_channel_seed(untrusted_seed: Option<&str>) -> Option<String> {
    let trimmed_seed = untrusted_seed?.trim();
    if trimmed_seed.is_empty() {
        return None;
    }
    let mut escaped_seed = String::new();
    append_template_safe_content(&mut escaped_seed, trimmed_seed);
    Some(escaped_seed)
}

/// Writes the Qwen3.5 assistant reasoning prefix and an escaped seed when enabled.
pub(crate) fn append_open_thinking_block(
    rendered_prompt: &mut String,
    enable_thinking: bool,
    untrusted_seed: Option<&str>,
) {
    rendered_prompt.push_str(THINK_START);
    rendered_prompt.push('\n');
    if enable_thinking {
        if let Some(escaped_seed) = escaped_qwen_thinking_channel_seed(untrusted_seed) {
            rendered_prompt.push_str(&escaped_seed);
            rendered_prompt.push('\n');
        }
        return;
    }
    rendered_prompt.push('\n');
    rendered_prompt.push_str(THINK_END);
    rendered_prompt.push_str("\n\n");
}

impl Qwen3_5Tokenizer {
    /// Encodes correction feedback and reopens the same seeded reasoning block.
    pub fn encode_model_visible_correction(
        &self,
        correction_text: &str,
        enable_thinking: bool,
        thinking_channel_seed: Option<&str>,
    ) -> Result<Vec<u32>, Qwen3_5TokenizerError> {
        let rendered_correction = Qwen3_5PromptRenderer::render_model_visible_correction(
            correction_text,
            enable_thinking,
            thinking_channel_seed,
        );
        self.encode_prompt(&rendered_correction)
    }
}
