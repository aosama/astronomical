//! One request's decoder state, as the `astro_decoder_state_cache` layout of
//! the loaded model shapes it, and its bridge to the blocks the
//! `astro_ssd_prompt_cache` stores.
//!
//! The stack is built from the model-owned layout alone, so any text family
//! whose layers are described by that layout uses it unchanged.

#[cfg(feature = "direct-mlx")]
mod astro_ssd_prompt_cache_boundary_checkpoint;
#[cfg(feature = "direct-mlx")]
pub mod family_decoder_state;
#[cfg(feature = "direct-mlx")]
mod persistent_state_bridge;
#[cfg(feature = "direct-mlx")]
mod persistent_state_kv_restore;
#[cfg(feature = "direct-mlx")]
pub mod reply_token_mlx_buffer_cache_reclaim;
#[cfg(all(test, feature = "direct-mlx"))]
mod reply_token_mlx_buffer_cache_reclaim_tests;
#[cfg(feature = "direct-mlx")]
pub mod request_state;
// The role names carry no MLX type, and a family's layout builder needs them
// in every build.
pub mod tensor_roles;

#[cfg(feature = "direct-mlx")]
pub use astro_ssd_prompt_cache_boundary_checkpoint::{
    AstroSsdPromptCacheBoundaryCheckpoint, AstroSsdPromptCacheBoundaryCheckpointCollector,
};
#[cfg(feature = "direct-mlx")]
pub use persistent_state_bridge::AstroSsdPromptCacheStateBridgeError;
#[cfg(feature = "direct-mlx")]
pub use request_state::{
    RequestDecoderStateStack, RequestDecoderStateStackAllocationCheckpoint,
    RequestDecoderStateStackCheckpoint,
};
