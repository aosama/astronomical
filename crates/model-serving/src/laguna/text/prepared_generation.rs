use astronomical_ipc_protocol::ChatToolDefinition;

use super::{
    LagunaInferenceRequest, LagunaOutputParser, LagunaOutputParserError, LagunaSamplerConfig,
    LagunaTextArtifactDescriptor,
};

/// Public prepared journey plus the future architecture-facing inference request.
#[derive(Debug)]
pub struct LagunaPreparedGeneration {
    inference_request: LagunaInferenceRequest,
    rendered_prompt: String,
    sampler_config: LagunaSamplerConfig,
    thinking_enabled: bool,
    thinking_budget: Option<u16>,
    descriptor: LagunaTextArtifactDescriptor,
    declared_tools: Vec<ChatToolDefinition>,
}

impl LagunaPreparedGeneration {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        inference_request: LagunaInferenceRequest,
        rendered_prompt: String,
        sampler_config: LagunaSamplerConfig,
        thinking_enabled: bool,
        thinking_budget: Option<u16>,
        descriptor: LagunaTextArtifactDescriptor,
        declared_tools: Vec<ChatToolDefinition>,
    ) -> Self {
        Self {
            inference_request,
            rendered_prompt,
            sampler_config,
            thinking_enabled,
            thinking_budget,
            descriptor,
            declared_tools,
        }
    }

    /// Returns prompt token IDs consumed by Laguna inference.
    #[must_use]
    pub fn prompt_token_ids(&self) -> &[u32] {
        self.inference_request.prompt_token_ids()
    }

    /// Returns the model-visible artifact-template prompt.
    #[must_use]
    pub fn rendered_prompt(&self) -> &str {
        &self.rendered_prompt
    }

    /// Returns the effective request thinking mode.
    #[must_use]
    pub const fn thinking_enabled(&self) -> bool {
        self.thinking_enabled
    }

    /// Returns whether generation starts after a prompt-owned `<think>` marker.
    #[must_use]
    pub const fn generation_starts_in_reasoning(&self) -> bool {
        self.thinking_enabled
    }

    /// Returns the positive explicit thinking budget, when supplied.
    #[must_use]
    pub const fn thinking_budget(&self) -> Option<u16> {
        self.thinking_budget
    }

    /// Returns effective artifact defaults plus request sampling overrides.
    #[must_use]
    pub const fn sampler_config(&self) -> &LagunaSamplerConfig {
        &self.sampler_config
    }

    /// Tests one generated ID against the complete normalized end-token union.
    #[must_use]
    pub fn is_end_token(&self, token_id: u32) -> bool {
        self.descriptor.is_end_token(token_id)
    }

    /// Creates independent parser state from this request's exact tool declarations.
    pub fn new_output_parser(&self) -> Result<LagunaOutputParser, LagunaOutputParserError> {
        LagunaOutputParser::new(
            &self.descriptor,
            &self.declared_tools,
            self.generation_starts_in_reasoning(),
        )
    }

    /// Transfers the architecture-facing inference request to the engine.
    #[must_use]
    pub fn into_inference_request(self) -> LagunaInferenceRequest {
        self.inference_request
    }
}
