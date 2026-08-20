//! Exact one-message Qwen3 rendering required by the official FLUX profile.

const USER_PREFIX: &str = "<|im_start|>user\n";
const NON_THINKING_GENERATION_SUFFIX: &str =
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

pub(crate) struct Flux2KleinPromptRenderer;

impl Flux2KleinPromptRenderer {
    /// Matches `apply_chat_template(..., add_generation_prompt=true,
    /// enable_thinking=false)` without introducing a default system message.
    pub(crate) fn render_user_prompt(user_prompt: &str) -> String {
        let mut rendered_prompt = String::with_capacity(
            USER_PREFIX.len() + user_prompt.len() + NON_THINKING_GENERATION_SUFFIX.len(),
        );
        rendered_prompt.push_str(USER_PREFIX);
        rendered_prompt.push_str(user_prompt);
        rendered_prompt.push_str(NON_THINKING_GENERATION_SUFFIX);
        rendered_prompt
    }

    pub(crate) fn render_user_prompts(user_prompts: &[String]) -> Vec<String> {
        user_prompts
            .iter()
            .map(|user_prompt| Self::render_user_prompt(user_prompt))
            .collect()
    }
}
