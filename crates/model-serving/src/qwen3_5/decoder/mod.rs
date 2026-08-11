pub(crate) mod cache_layout;
#[cfg(feature = "direct-mlx")]
mod persistent_prompt_cache_boundary_checkpoint;
#[cfg(feature = "direct-mlx")]
mod persistent_state_bridge;
#[cfg(feature = "direct-mlx")]
#[path = "request_state/mod.rs"]
pub(crate) mod request_decoder_state;
#[cfg(feature = "direct-mlx")]
mod speculative_prefill_target_state_bridge;

pub use cache_layout::{Qwen3_5DecoderLayerCacheDtypes, qwen3_5_decoder_cache_layout};
#[cfg(feature = "direct-mlx")]
pub use persistent_prompt_cache_boundary_checkpoint::{
    Qwen3_5PersistentPromptCacheBoundaryCheckpoint,
    Qwen3_5PersistentPromptCacheBoundaryCheckpointCollector,
};
#[cfg(feature = "direct-mlx")]
pub use persistent_state_bridge::PersistentPromptCacheStateBridgeError;
#[cfg(feature = "direct-mlx")]
pub use request_decoder_state::{
    Qwen3_5MtpRequestState, Qwen3_5MtpRequestStateAllocationCheckpoint, RequestDecoderStateStack,
    RequestDecoderStateStackAllocationCheckpoint, RequestDecoderStateStackCheckpoint,
};

pub(crate) use super::configuration::Qwen3_5Config;
