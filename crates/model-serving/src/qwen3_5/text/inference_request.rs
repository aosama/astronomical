use astronomical_ipc_protocol::RequestId;

use crate::{PerformanceAttribution, PreparedInferenceRequest};

use super::Qwen3_5ProcessedImage;

/// Token-level request prepared for Qwen3.5 inference.
#[derive(Clone, Debug)]
pub struct Qwen3_5InferenceRequest {
    input_token_ids: Vec<u32>,
    visual_embeddings: Option<Vec<f32>>,
    visual_embedding_row_count: usize,
    processed_visual_images: Vec<Qwen3_5ProcessedImage>,
    image_pad_token_id: Option<u32>,
    max_output_tokens: u16,
    request_id: RequestId,
    sampling_strategy: Qwen3_5SamplingStrategy,
    thinking_budget: Option<u16>,
    performance_attribution: PerformanceAttribution,
}

impl PreparedInferenceRequest for Qwen3_5InferenceRequest {
    fn prompt_token_count(&self) -> usize {
        self.input_token_ids.len()
    }
}

impl Qwen3_5InferenceRequest {
    /// Creates a token-level request accepted by Qwen3.5 inference.
    pub fn new(request_id: RequestId, input_token_ids: Vec<u32>, max_output_tokens: u16) -> Self {
        Self {
            input_token_ids,
            visual_embeddings: None,
            visual_embedding_row_count: 0,
            processed_visual_images: Vec::new(),
            image_pad_token_id: None,
            max_output_tokens,
            request_id,
            sampling_strategy: Qwen3_5SamplingStrategy::Greedy,
            thinking_budget: None,
            performance_attribution: PerformanceAttribution::disabled(),
        }
    }

    /// Creates a token-level request using Qwen3.5's fixed top-k sampling family.
    pub fn new_sampling(
        request_id: RequestId,
        input_token_ids: Vec<u32>,
        max_output_tokens: u16,
        temperature_thousandths: u16,
        top_p_thousandths: u16,
        seed: Option<u64>,
    ) -> Self {
        let sampling_strategy = if temperature_thousandths == 0 {
            Qwen3_5SamplingStrategy::Greedy
        } else {
            Qwen3_5SamplingStrategy::TopKTopP {
                temperature_thousandths,
                top_k: 20,
                top_p_thousandths,
                seed,
            }
        };
        Self {
            input_token_ids,
            visual_embeddings: None,
            visual_embedding_row_count: 0,
            processed_visual_images: Vec::new(),
            image_pad_token_id: None,
            max_output_tokens,
            request_id,
            sampling_strategy,
            thinking_budget: None,
            performance_attribution: PerformanceAttribution::disabled(),
        }
    }

    /// Attaches pre-computed visual embeddings and clears pending processed images.
    #[must_use]
    pub fn with_visual_embeddings(
        mut self,
        visual_embeddings: Vec<f32>,
        visual_embedding_row_count: usize,
    ) -> Self {
        self.visual_embeddings = Some(visual_embeddings);
        self.visual_embedding_row_count = visual_embedding_row_count;
        self.processed_visual_images.clear();
        self
    }

    /// Attaches CPU-processed images for inference-side vision-tower execution.
    #[must_use]
    pub fn with_processed_visual_images(
        mut self,
        processed_visual_images: Vec<Qwen3_5ProcessedImage>,
    ) -> Self {
        self.visual_embeddings = None;
        self.visual_embedding_row_count = 0;
        self.processed_visual_images = processed_visual_images;
        self
    }

    /// Attaches the tokenizer-validated image-pad token identifier.
    #[must_use]
    pub const fn with_image_pad_token_id(mut self, image_pad_token_id: u32) -> Self {
        self.image_pad_token_id = Some(image_pad_token_id);
        self
    }

    /// Sets the maximum number of tokens the model may spend in its thinking block.
    #[must_use]
    pub const fn with_thinking_budget(mut self, thinking_budget: u16) -> Self {
        self.thinking_budget = Some(thinking_budget);
        self
    }

    /// Attaches the request-local critical-path performance accumulator.
    #[must_use]
    pub fn with_performance_attribution(
        mut self,
        performance_attribution: PerformanceAttribution,
    ) -> Self {
        self.performance_attribution = performance_attribution;
        self
    }

    /// Transfers the request-local performance accumulator to native inference.
    #[cfg(feature = "direct-mlx")]
    pub(crate) fn take_performance_attribution(&mut self) -> PerformanceAttribution {
        std::mem::replace(
            &mut self.performance_attribution,
            PerformanceAttribution::disabled(),
        )
    }

    /// Returns the optional thinking-token budget.
    #[must_use]
    pub const fn thinking_budget(&self) -> Option<u16> {
        self.thinking_budget
    }

    /// Returns whether pre-computed visual embeddings are attached.
    #[must_use]
    pub fn has_visual_embeddings(&self) -> bool {
        self.visual_embeddings.is_some()
    }

    /// Returns the flat pre-computed visual embedding values, if any.
    #[must_use]
    pub fn visual_embeddings(&self) -> Option<&[f32]> {
        self.visual_embeddings.as_deref()
    }

    /// Returns the visual embedding row count after spatial merging.
    #[must_use]
    pub fn visual_embedding_row_count(&self) -> usize {
        self.visual_embedding_row_count
    }

    /// Returns whether CPU-processed images are awaiting vision execution.
    #[must_use]
    pub fn has_processed_visual_images(&self) -> bool {
        !self.processed_visual_images.is_empty()
    }

    /// Returns CPU-processed images in prompt order.
    #[must_use]
    pub fn processed_visual_images(&self) -> &[Qwen3_5ProcessedImage] {
        &self.processed_visual_images
    }

    /// Returns the tokenizer-validated image-pad token identifier, if attached.
    #[must_use]
    pub const fn image_pad_token_id(&self) -> Option<u32> {
        self.image_pad_token_id
    }

    /// Returns the request correlation identifier.
    #[must_use]
    pub fn request_id(&self) -> RequestId {
        self.request_id
    }

    /// Returns the tokenized and formatted prompt.
    #[must_use]
    pub fn input_token_ids(&self) -> &[u32] {
        &self.input_token_ids
    }

    /// Returns the accepted output-token budget.
    #[must_use]
    pub fn max_output_tokens(&self) -> u16 {
        self.max_output_tokens
    }

    /// Returns the selected Qwen3.5 sampling strategy.
    #[must_use]
    pub const fn sampling_strategy(&self) -> Qwen3_5SamplingStrategy {
        self.sampling_strategy
    }
}

/// Sampling behavior selected for Qwen3.5 token generation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen3_5SamplingStrategy {
    /// Selects the maximum finite logit without random state.
    Greedy,
    /// Applies temperature, fixed top-k, top-p, and an optional deterministic seed.
    TopKTopP {
        temperature_thousandths: u16,
        top_k: u16,
        top_p_thousandths: u16,
        seed: Option<u64>,
    },
}
