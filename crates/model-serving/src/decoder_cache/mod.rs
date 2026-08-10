//! Closed decoder-cache layouts shared by model implementations and persistent storage.

#[cfg(feature = "direct-mlx")]
mod append_only_attention_state;
#[cfg(feature = "direct-mlx")]
mod append_only_attention_state_operations;
#[cfg(feature = "direct-mlx")]
mod convolution_state;
#[cfg(feature = "direct-mlx")]
mod gated_delta_recurrent_state;
mod layout;
mod layout_error;
#[cfg(feature = "direct-mlx")]
mod live_state;
mod storage_geometry;

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
    DecoderCacheLayerLayout, DecoderCacheLayout, DecoderCachePersistedTensorLayout,
    DecoderCacheTensorDtype, DecoderCacheTensorLayout,
};
pub use layout_error::DecoderCacheLayoutError;
#[cfg(feature = "direct-mlx")]
pub use live_state::{
    DecoderCacheState, DecoderCacheStateAllocationCheckpoint, DecoderCacheStateCheckpoint,
};
