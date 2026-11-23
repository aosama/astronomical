//! The thinking-control fields Chat Completions and Responses both read, in
//! every spelling clients send, as one group each request flattens in.
//!
//! Each request declares this group before its catch-all map of other
//! fields. Serde hands a flattened struct its fields first and the map only
//! what is left, so these spellings never reach the map: Responses, which
//! refuses any field left there, keeps accepting all of them.

use serde::Deserialize;

use crate::{
    ChatTemplateKwargsRequestObject, ReasoningRequestObject, ThinkingControls,
    ThinkingControlsError, ThinkingControlsInputs,
};

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub(crate) struct ThinkingControlFields {
    /// Maximum tokens the model may spend inside the thinking block before being
    /// forced to close it and start the visible response. When `None`, the model
    /// thinks freely up to the output budget.
    #[serde(default)]
    thinking_budget: Option<u32>,
    /// Coding-agent spelling of the thinking budget: the agent resolves this
    /// field name from its provider compatibility configuration and falls back
    /// to it when the server declares thinking-budget support.
    #[serde(default)]
    thinking_token_budget: Option<u32>,
    /// Documented OpenAI-compatible alias for the thinking budget used by
    /// coding-agent traffic alongside `thinking_token_budget`.
    #[serde(default)]
    thinking_budget_tokens: Option<u32>,
    /// OpenRouter-style reasoning object carrying an effort level, a direct
    /// token limit, an on/off switch, and a stream-exclusion preference.
    #[serde(default)]
    reasoning: Option<ReasoningRequestObject>,
    /// OpenAI `reasoning_effort` level name.
    #[serde(default)]
    reasoning_effort: Option<String>,
    /// Qwen-style flat switch for the thinking channel.
    #[serde(default)]
    enable_thinking: Option<bool>,
    /// vLLM-style template-kwarg block carrying the thinking toggle.
    #[serde(default)]
    chat_template_kwargs: Option<ChatTemplateKwargsRequestObject>,
}

impl ThinkingControlFields {
    /// Resolves every submitted spelling into one hard budget plus the stream
    /// exclusion preference, mirroring the output-token budget rule: several
    /// spellings are fine while they agree, and disagreement is a caller bug
    /// that must fail loudly instead of silently picking one.
    pub(crate) fn resolve(&self) -> Result<ThinkingControls, ThinkingControlsError> {
        ThinkingControlsInputs {
            thinking_budget: self.thinking_budget,
            thinking_token_budget: self.thinking_token_budget,
            thinking_budget_tokens: self.thinking_budget_tokens,
            reasoning: self.reasoning.as_ref(),
            reasoning_effort: self.reasoning_effort.as_deref(),
            enable_thinking: self.enable_thinking,
            chat_template_kwargs: self.chat_template_kwargs.as_ref(),
        }
        .resolve()
    }
}
