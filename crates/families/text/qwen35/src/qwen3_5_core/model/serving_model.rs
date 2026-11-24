//! What the shared generation loop reads from a loaded text model (issue
//! #2473): its MLX runtime, its vocabulary and position limits, the layout
//! of its decoder state, and its vision tower when it has one.
//!
//! The loop used to require a `Qwen3_5ModelBase`, which owns the Qwen 3.5
//! weights. Asking this trait instead lets a model with other weights, the
//! folded Prism Hadamard pack, run through the same loop. The model base
//! answers from its own fields, and every model that dereferences to one
//! answers from that base, so the resident and streaming engines read
//! exactly what they read before.

use std::ops::DerefMut;

use mlx_bridge::{MlxArray, MlxCompiledElementwiseGraphs};
use runtime_integration::MlxRuntime;

use super::Qwen3_5ModelBase;
use crate::AstroDecoderStateCacheLayout;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_core::vision::Qwen3_5VisionModel;

pub trait Qwen3_5ServingModel {
    fn runtime(&self) -> &MlxRuntime;

    /// The number of token ids the model's output scores.
    fn vocabulary_size(&self) -> u32;

    /// The model's own position range, the hard limit of a request.
    fn maximum_position_count(&self) -> u32;

    fn astro_decoder_state_cache_layout(&self) -> &AstroDecoderStateCacheLayout;

    /// The greedy choice: the highest-scoring token at the final position.
    fn select_highest_logit_token(
        &self,
        final_position_logits: &MlxArray,
    ) -> Result<MlxArray, Qwen3_5ExecutionError>;

    /// The vision tower, absent for a text-only model.
    fn vision_model(&self) -> Option<&Qwen3_5VisionModel>;

    fn compiled_elementwise_graphs(&self) -> &MlxCompiledElementwiseGraphs;

    /// Frees the vision tower's working memory once an image request has
    /// read its images.
    fn release_vision_tower_after_image_request(&mut self);
}

impl Qwen3_5ServingModel for Qwen3_5ModelBase {
    fn runtime(&self) -> &MlxRuntime {
        Qwen3_5ModelBase::runtime(self)
    }

    fn vocabulary_size(&self) -> u32 {
        self.config.vocabulary_size()
    }

    fn maximum_position_count(&self) -> u32 {
        self.config.maximum_position_count()
    }

    fn astro_decoder_state_cache_layout(&self) -> &AstroDecoderStateCacheLayout {
        Qwen3_5ModelBase::astro_decoder_state_cache_layout(self)
    }

    fn select_highest_logit_token(
        &self,
        final_position_logits: &MlxArray,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        Qwen3_5ModelBase::select_highest_logit_token(self, final_position_logits)
    }

    fn vision_model(&self) -> Option<&Qwen3_5VisionModel> {
        Qwen3_5ModelBase::vision_model(self)
    }

    fn compiled_elementwise_graphs(&self) -> &MlxCompiledElementwiseGraphs {
        &self.compiled_elementwise_graphs
    }

    fn release_vision_tower_after_image_request(&mut self) {
        Qwen3_5ModelBase::release_vision_tower_after_image_request(self);
    }
}

/// A resident or streaming model answers from the model base it wraps.
impl<Model> Qwen3_5ServingModel for Model
where
    Model: DerefMut<Target = Qwen3_5ModelBase>,
{
    fn runtime(&self) -> &MlxRuntime {
        Qwen3_5ServingModel::runtime(&**self)
    }

    fn vocabulary_size(&self) -> u32 {
        Qwen3_5ServingModel::vocabulary_size(&**self)
    }

    fn maximum_position_count(&self) -> u32 {
        Qwen3_5ServingModel::maximum_position_count(&**self)
    }

    fn astro_decoder_state_cache_layout(&self) -> &AstroDecoderStateCacheLayout {
        Qwen3_5ServingModel::astro_decoder_state_cache_layout(&**self)
    }

    fn select_highest_logit_token(
        &self,
        final_position_logits: &MlxArray,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        Qwen3_5ServingModel::select_highest_logit_token(&**self, final_position_logits)
    }

    fn vision_model(&self) -> Option<&Qwen3_5VisionModel> {
        Qwen3_5ServingModel::vision_model(&**self)
    }

    fn compiled_elementwise_graphs(&self) -> &MlxCompiledElementwiseGraphs {
        Qwen3_5ServingModel::compiled_elementwise_graphs(&**self)
    }

    fn release_vision_tower_after_image_request(&mut self) {
        Qwen3_5ServingModel::release_vision_tower_after_image_request(&mut **self);
    }
}
