//! Closed astro-decoder-state-cache layouts shared by model implementations and persistent storage.

#[cfg(feature = "direct-mlx")]
mod append_only_attention_state;
#[cfg(feature = "direct-mlx")]
mod append_only_attention_state_operations;
#[cfg(feature = "direct-mlx")]
mod convolution_state;
#[cfg(feature = "direct-mlx")]
mod gated_delta_recurrent_state;
#[cfg(feature = "direct-mlx")]
mod incremental_block_restore;
mod layout;
mod layout_error;
#[cfg(feature = "direct-mlx")]
mod live_state;
mod persistence_layouts;
#[cfg(feature = "direct-mlx")]
mod rotating_attention_state;
mod rotating_layout;
mod storage_geometry;

#[cfg(test)]
mod astro_decoder_state_cache_tests;

#[cfg(feature = "direct-mlx")]
pub use append_only_attention_state::{
    FullAttentionKeyValueState, FullAttentionKeyValueStateAllocationCheckpoint,
};
#[cfg(feature = "direct-mlx")]
pub use convolution_state::{
    ConvolutionState, ConvolutionStateBoundaryCheckpointUpdate, ConvolutionStateCheckpoint,
};
#[cfg(feature = "direct-mlx")]
pub use gated_delta_recurrent_state::{
    GatedDeltaRecurrentState, GatedDeltaRecurrentStateCheckpoint,
};
pub use layout::{
    AstroDecoderStateCacheLayerLayout, AstroDecoderStateCacheLayout,
    AstroDecoderStateCachePersistedTensorLayout, AstroDecoderStateCacheTensorDtype,
    AstroDecoderStateCacheTensorLayout,
};
pub use layout_error::AstroDecoderStateCacheLayoutError;
#[cfg(feature = "direct-mlx")]
pub use live_state::{
    AstroDecoderStateCacheState, AstroDecoderStateCacheStateAllocationCheckpoint,
    AstroDecoderStateCacheStateCheckpoint,
};
#[cfg(feature = "direct-mlx")]
pub use rotating_attention_state::{
    RotatingKeyValueState, RotatingKeyValueStateAllocationCheckpoint,
};
