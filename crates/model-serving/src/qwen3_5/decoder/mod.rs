pub(crate) mod cache_layout;
#[cfg(feature = "direct-mlx")]
mod persistent_state_bridge;
#[cfg(feature = "direct-mlx")]
#[path = "request_state/mod.rs"]
pub(crate) mod request_decoder_state;

pub use cache_layout::qwen3_5_decoder_cache_layout;
#[cfg(feature = "direct-mlx")]
pub use request_decoder_state::{
    Qwen3_5MtpRequestState, Qwen3_5MtpRequestStateAllocationCheckpoint,
    Qwen3_5MtpUnavailableReason, RequestDecoderStateStack,
    RequestDecoderStateStackAllocationCheckpoint, RequestDecoderStateStackCheckpoint,
};

pub(crate) use super::configuration::Qwen3_5Config;
