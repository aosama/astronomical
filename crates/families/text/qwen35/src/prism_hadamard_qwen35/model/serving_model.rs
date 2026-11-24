//! What the shared Qwen 3.5 generation loop reads from the Prism text model
//! (issue #2473, Prism step 5). Every answer comes from what the Prism engine
//! already uses, so a request served through the shared loop reads the same
//! limits and makes the same greedy choice as one served by Prism's own loop.

use mlx_bridge::{MlxArray, MlxCompiledElementwiseGraphs};
use runtime_integration::MlxRuntime;

use super::PrismHadamardTextModel;
use crate::AstroDecoderStateCacheLayout;
use crate::qwen3_5_core::model::Qwen3_5ServingModel;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_core::vision::Qwen3_5VisionModel;

impl Qwen3_5ServingModel for PrismHadamardTextModel {
    fn runtime(&self) -> &MlxRuntime {
        &self.runtime
    }

    fn vocabulary_size(&self) -> u32 {
        self.pack.config().text().vocab_size
    }

    fn maximum_position_count(&self) -> u32 {
        self.pack.config().text().max_position_embeddings
    }

    fn astro_decoder_state_cache_layout(&self) -> &AstroDecoderStateCacheLayout {
        &self.astro_decoder_state_cache_layout
    }

    /// The same last-axis argmax Prism's own greedy sampling submits.
    fn select_highest_logit_token(
        &self,
        final_position_logits: &MlxArray,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        Ok(self.greedy_token_lazily(final_position_logits)?)
    }

    /// A folded pack carries text weights only.
    fn vision_model(&self) -> Option<&Qwen3_5VisionModel> {
        None
    }

    /// The compiled graphs Prism's own forward already runs on.
    fn compiled_elementwise_graphs(&self) -> &MlxCompiledElementwiseGraphs {
        &self.compiled_elementwise_graphs
    }

    /// Nothing to release: a folded pack has no vision tower.
    fn release_vision_tower_after_image_request(&mut self) {}
}

#[cfg(test)]
#[path = "serving_model_tests.rs"]
mod serving_model_tests;
