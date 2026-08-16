use std::collections::BTreeMap;
use std::sync::Arc;

use tokenizers::Tokenizer;

use super::LagunaSamplerConfig;
use super::template_program::LagunaTemplateProgram;

/// Canonical facts derived by executing the retained artifact template at startup.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LagunaTemplateContract {
    pub(super) bos_token_content: String,
    pub(super) default_system_message: String,
    pub(super) default_thinking_enabled: bool,
    pub(super) preserves_prior_reasoning: bool,
}

/// Canonical owner of every model-side value needed by Laguna text generation.
#[derive(Clone, Debug)]
pub struct LagunaTextArtifactDescriptor {
    // Parsing tokenizer JSON is startup work. Every request and processor shares
    // the already-certified tokenizer instead of retaining and reparsing its bytes.
    tokenizer: Arc<Tokenizer>,
    model_vocabulary_size: u32,
    maximum_context_tokens: u32,
    bos_token_id: u32,
    pad_token_id: u32,
    end_token_ids: Arc<[u32]>,
    token_ids_by_content: Arc<BTreeMap<String, u32>>,
    template_program: Arc<LagunaTemplateProgram>,
    template_contract: LagunaTemplateContract,
    generation_default_thinking_enabled: Option<bool>,
    reasoning_parser_id: String,
    tool_call_parser_id: String,
    sampler_config: LagunaSamplerConfig,
}

impl LagunaTextArtifactDescriptor {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        tokenizer: Tokenizer,
        model_vocabulary_size: u32,
        maximum_context_tokens: u32,
        bos_token_id: u32,
        pad_token_id: u32,
        end_token_ids: Vec<u32>,
        token_ids_by_content: BTreeMap<String, u32>,
        template_program: LagunaTemplateProgram,
        template_contract: LagunaTemplateContract,
        generation_default_thinking_enabled: Option<bool>,
        reasoning_parser_id: String,
        tool_call_parser_id: String,
        sampler_config: LagunaSamplerConfig,
    ) -> Self {
        Self {
            tokenizer: Arc::new(tokenizer),
            model_vocabulary_size,
            maximum_context_tokens,
            bos_token_id,
            pad_token_id,
            end_token_ids: Arc::from(end_token_ids),
            token_ids_by_content: Arc::new(token_ids_by_content),
            template_program: Arc::new(template_program),
            template_contract,
            generation_default_thinking_enabled,
            reasoning_parser_id,
            tool_call_parser_id,
            sampler_config,
        }
    }

    /// Returns the model output-vocabulary bound.
    #[must_use]
    pub const fn model_vocabulary_size(&self) -> u32 {
        self.model_vocabulary_size
    }

    /// Returns the model-native shared prompt and generation context.
    #[must_use]
    pub const fn maximum_context_tokens(&self) -> u32 {
        self.maximum_context_tokens
    }

    /// Returns the configured beginning-of-sequence token ID.
    #[must_use]
    pub const fn bos_token_id(&self) -> u32 {
        self.bos_token_id
    }

    /// Returns the configured padding token ID.
    #[must_use]
    pub const fn pad_token_id(&self) -> u32 {
        self.pad_token_id
    }

    /// Returns the sorted, deduplicated union of every configured end token.
    #[must_use]
    pub fn end_token_ids(&self) -> &[u32] {
        &self.end_token_ids
    }

    /// Tests membership without assuming any Laguna-family token constant.
    #[must_use]
    pub fn is_end_token(&self, token_id: u32) -> bool {
        self.end_token_ids.binary_search(&token_id).is_ok()
    }

    /// Discovers a tokenizer token by content rather than a hardcoded ID.
    #[must_use]
    pub fn token_id_for(&self, token_content: &str) -> Option<u32> {
        self.token_ids_by_content.get(token_content).copied()
    }

    /// Returns the template-local thinking default.
    #[must_use]
    pub const fn default_thinking_enabled(&self) -> bool {
        self.template_contract.default_thinking_enabled
    }

    /// Returns whether disabled current thinking still preserves historical reasoning.
    #[must_use]
    pub const fn preserves_prior_reasoning(&self) -> bool {
        self.template_contract.preserves_prior_reasoning
    }

    /// Returns the fallback system behavior derived from the artifact template.
    #[must_use]
    pub fn default_system_message(&self) -> &str {
        &self.template_contract.default_system_message
    }

    /// Returns the certified reasoning parser identifier.
    #[must_use]
    pub fn reasoning_parser_id(&self) -> &str {
        &self.reasoning_parser_id
    }

    /// Returns the certified tool-call parser identifier.
    #[must_use]
    pub fn tool_call_parser_id(&self) -> &str {
        &self.tool_call_parser_id
    }

    /// Returns artifact sampling defaults without request mutation.
    #[must_use]
    pub const fn sampler_config(&self) -> &LagunaSamplerConfig {
        &self.sampler_config
    }

    pub(super) fn template_contract(&self) -> &LagunaTemplateContract {
        &self.template_contract
    }

    pub(super) fn template_program(&self) -> &LagunaTemplateProgram {
        &self.template_program
    }

    pub(super) fn tokenizer(&self) -> &Arc<Tokenizer> {
        &self.tokenizer
    }

    /// Returns the generation-config thinking override before request precedence is applied.
    #[must_use]
    pub const fn generation_default_thinking_enabled(&self) -> Option<bool> {
        self.generation_default_thinking_enabled
    }
}
