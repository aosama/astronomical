//! The shared Qwen3.5 model base type and its paging-free execution pieces.

#[cfg(feature = "direct-mlx")]
pub(crate) mod base;
#[cfg(feature = "direct-mlx")]
pub(crate) mod decoder_layer_attention;
#[cfg(feature = "direct-mlx")]
pub(crate) mod full_attention;
#[cfg(feature = "direct-mlx")]
pub(crate) mod gated_delta;
#[cfg(feature = "direct-mlx")]
pub(crate) mod model_chunking_configuration;

#[cfg(feature = "direct-mlx")]
pub(crate) use base::Qwen3_5ModelBase;
#[cfg(feature = "direct-mlx")]
pub(crate) use decoder_layer_attention::Qwen3_5DecoderLayerAttentionOutput;
