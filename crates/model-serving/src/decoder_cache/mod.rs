//! Closed decoder-cache layouts shared by model implementations and persistent storage.

#[cfg(feature = "direct-mlx")]
mod append_only_attention_state;
#[cfg(feature = "direct-mlx")]
mod convolution_state;
#[cfg(feature = "direct-mlx")]
mod gated_delta_recurrent_state;
mod layout;
#[cfg(feature = "direct-mlx")]
mod live_state;

#[cfg(feature = "direct-mlx")]
pub use append_only_attention_state::{
    DEFAULT_FULL_ATTENTION_KV_STATE_GROWTH_TOKENS, FullAttentionKeyValueState,
    FullAttentionKeyValueStateAllocationCheckpoint,
};
#[cfg(feature = "direct-mlx")]
pub use convolution_state::{ConvolutionState, ConvolutionStateCheckpoint};
#[cfg(feature = "direct-mlx")]
pub use gated_delta_recurrent_state::{
    GatedDeltaRecurrentState, GatedDeltaRecurrentStateCheckpoint,
};
pub use layout::{
    DEFAULT_APPEND_ONLY_ATTENTION_CAPACITY_GROWTH_TOKENS, DecoderCacheLayerLayout,
    DecoderCacheLayout, DecoderCacheLayoutError, DecoderCachePersistedTensorLayout,
    DecoderCacheTensorDtype, DecoderCacheTensorLayout,
};
#[cfg(feature = "direct-mlx")]
pub use live_state::{
    DecoderCacheState, DecoderCacheStateAllocationCheckpoint, DecoderCacheStateCheckpoint,
};
