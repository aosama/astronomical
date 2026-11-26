//! How the Qwen 3.5 models answer the text shell's generation loop (issue
//! #2473): the model base answers from its own fields, and the text model
//! answers from the base it wraps.
//!
//! The shell's trait asks only what any text model can answer. The vision
//! tower itself is a Qwen 3.5 type, so the steps that run it reach it
//! through [`Qwen3_5VisionTowerOwner`], which stays with the family.

use std::ops::Deref;

use mlx_bridge::{MlxArray, MlxCompiledElementwiseGraphs};
use runtime_integration::MlxRuntime;
use text_generation::execution_error::TextExecutionError;
use text_generation::model::TextServingModel;

use super::Qwen3_5ModelBase;
use crate::AstroDecoderStateCacheLayout;
use crate::qwen3_5_core::vision::Qwen3_5VisionModel;

/// A Qwen 3.5 model the request steps can take the vision tower from.
pub trait Qwen3_5VisionTowerOwner: TextServingModel {
    /// The vision tower, absent for a text-only model.
    fn vision_model(&self) -> Option<&Qwen3_5VisionModel>;
}

impl Qwen3_5VisionTowerOwner for Qwen3_5ModelBase {
    fn vision_model(&self) -> Option<&Qwen3_5VisionModel> {
        Qwen3_5ModelBase::vision_model(self)
    }
}

/// A text model hands out the tower of the base it wraps.
impl<Model> Qwen3_5VisionTowerOwner for Model
where
    Model: Deref<Target = Qwen3_5ModelBase> + TextServingModel,
{
    fn vision_model(&self) -> Option<&Qwen3_5VisionModel> {
        Qwen3_5ModelBase::vision_model(self)
    }
}

impl TextServingModel for Qwen3_5ModelBase {
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
    ) -> Result<MlxArray, TextExecutionError> {
        Qwen3_5ModelBase::select_highest_logit_token(self, final_position_logits)
    }

    fn vision_tower_resident_payload_bytes(&self) -> u64 {
        Qwen3_5ModelBase::vision_model(self).map_or(0, Qwen3_5VisionModel::resident_payload_bytes)
    }

    fn compiled_elementwise_graphs(&self) -> &MlxCompiledElementwiseGraphs {
        &self.compiled_elementwise_graphs
    }

    fn release_vision_tower_after_image_request(&mut self) {
        Qwen3_5ModelBase::release_vision_tower_after_image_request(self);
    }
}

/// Answers the shell's [`TextServingModel`] for a model that wraps a
/// [`Qwen3_5ModelBase`], from that base. A macro, because the trait belongs
/// to the shell: Rust allows a crate to implement another crate's trait only
/// for its own named types, never for every type that dereferences to one.
macro_rules! serve_text_from_wrapped_model_base {
    ($model:ty) => {
        impl ::text_generation::model::TextServingModel for $model {
            fn runtime(&self) -> &::runtime_integration::MlxRuntime {
                ::text_generation::model::TextServingModel::runtime(&**self)
            }

            fn vocabulary_size(&self) -> u32 {
                ::text_generation::model::TextServingModel::vocabulary_size(&**self)
            }

            fn maximum_position_count(&self) -> u32 {
                ::text_generation::model::TextServingModel::maximum_position_count(&**self)
            }

            fn astro_decoder_state_cache_layout(&self) -> &$crate::AstroDecoderStateCacheLayout {
                ::text_generation::model::TextServingModel::astro_decoder_state_cache_layout(
                    &**self,
                )
            }

            fn select_highest_logit_token(
                &self,
                final_position_logits: &::mlx_bridge::MlxArray,
            ) -> Result<
                ::mlx_bridge::MlxArray,
                ::text_generation::execution_error::TextExecutionError,
            > {
                ::text_generation::model::TextServingModel::select_highest_logit_token(
                    &**self,
                    final_position_logits,
                )
            }

            fn vision_tower_resident_payload_bytes(&self) -> u64 {
                ::text_generation::model::TextServingModel::vision_tower_resident_payload_bytes(
                    &**self,
                )
            }

            fn compiled_elementwise_graphs(&self) -> &::mlx_bridge::MlxCompiledElementwiseGraphs {
                ::text_generation::model::TextServingModel::compiled_elementwise_graphs(&**self)
            }

            fn release_vision_tower_after_image_request(&mut self) {
                ::text_generation::model::TextServingModel::release_vision_tower_after_image_request(
                    &mut **self,
                );
            }
        }
    };
}

pub(crate) use serve_text_from_wrapped_model_base;
