use astronomical_ipc_protocol::RequestId;

use crate::{PerformanceAttribution, PreparedInferenceRequest};

use super::LagunaSamplerConfig;

/// Architecture-facing sampling strategy retained without another model-family dependency.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LagunaSamplingStrategy {
    HighestLogit,
    Sample(LagunaSamplerConfig),
}

/// Token-level Laguna request prepared for a future Laguna inference engine.
#[derive(Clone, Debug)]
pub struct LagunaInferenceRequest {
    request_id: RequestId,
    prompt_token_ids: Vec<u32>,
    maximum_output_tokens: u16,
    sampling_strategy: LagunaSamplingStrategy,
    generation_starts_in_reasoning: bool,
    thinking_budget: Option<u16>,
    performance_attribution: PerformanceAttribution,
}

impl PreparedInferenceRequest for LagunaInferenceRequest {
    fn prompt_token_count(&self) -> usize {
        self.prompt_token_ids.len()
    }
}

impl LagunaInferenceRequest {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        request_id: RequestId,
        prompt_token_ids: Vec<u32>,
        maximum_output_tokens: u16,
        sampler_config: LagunaSamplerConfig,
        generation_starts_in_reasoning: bool,
        thinking_budget: Option<u16>,
        performance_attribution: PerformanceAttribution,
    ) -> Self {
        let sampling_strategy = if sampler_config.uses_sampling() {
            LagunaSamplingStrategy::Sample(sampler_config)
        } else {
            LagunaSamplingStrategy::HighestLogit
        };
        Self {
            request_id,
            prompt_token_ids,
            maximum_output_tokens,
            sampling_strategy,
            generation_starts_in_reasoning,
            thinking_budget,
            performance_attribution,
        }
    }

    /// Returns the request correlation ID.
    #[must_use]
    pub fn request_id(&self) -> RequestId {
        self.request_id
    }

    /// Returns the artifact-template prompt token IDs.
    #[must_use]
    pub fn prompt_token_ids(&self) -> &[u32] {
        &self.prompt_token_ids
    }

    /// Returns the accepted request output-token budget.
    #[must_use]
    pub const fn maximum_output_tokens(&self) -> u16 {
        self.maximum_output_tokens
    }

    /// Returns the effective architecture-facing sampler strategy.
    #[must_use]
    pub const fn sampling_strategy(&self) -> &LagunaSamplingStrategy {
        &self.sampling_strategy
    }

    /// Returns whether output continues the prompt-owned reasoning block.
    #[must_use]
    pub const fn generation_starts_in_reasoning(&self) -> bool {
        self.generation_starts_in_reasoning
    }

    /// Returns the positive request-local thinking budget when supplied.
    #[must_use]
    pub const fn thinking_budget(&self) -> Option<u16> {
        self.thinking_budget
    }

    /// Transfers request attribution to the future inference implementation.
    #[must_use]
    pub fn into_performance_attribution(self) -> PerformanceAttribution {
        self.performance_attribution
    }
}
