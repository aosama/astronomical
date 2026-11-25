//! What the generation loop reads from a loaded text model: its MLX
//! runtime, its vocabulary and position limits, the layout of its decoder
//! state, and what its vision tower holds when it has one.
//!
//! The loop asks this trait and never a family's own model type, so a model
//! with other weights runs through the same loop. Each family answers from
//! its own model.

use mlx_bridge::{MlxArray, MlxCompiledElementwiseGraphs};
use runtime_integration::MlxRuntime;

use crate::AstroDecoderStateCacheLayout;
use crate::execution_error::TextExecutionError;

pub trait TextServingModel {
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
    ) -> Result<MlxArray, TextExecutionError>;

    /// The bytes the vision tower holds in MLX memory; zero for a text-only
    /// model and once the tower's working memory has been released.
    fn vision_tower_resident_payload_bytes(&self) -> u64;

    fn compiled_elementwise_graphs(&self) -> &MlxCompiledElementwiseGraphs;

    /// Frees the vision tower's working memory once an image request has
    /// read its images.
    fn release_vision_tower_after_image_request(&mut self);
}
