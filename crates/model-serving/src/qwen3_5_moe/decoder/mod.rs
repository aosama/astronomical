pub(crate) mod cache_layout;
#[cfg(feature = "direct-mlx")]
mod persistent_state_bridge;
#[cfg(feature = "direct-mlx")]
#[path = "request_state/mod.rs"]
pub(crate) mod request_decoder_state;

pub use cache_layout::qwen3_5_moe_decoder_cache_layout;
#[cfg(feature = "direct-mlx")]
pub use request_decoder_state::{
    Qwen3_5MoEMtpRequestState, Qwen3_5MoEMtpRequestStateAllocationCheckpoint,
    Qwen3_5MoEMtpUnavailableReason, RequestDecoderStateStack,
    RequestDecoderStateStackAllocationCheckpoint, RequestDecoderStateStackCheckpoint,
};

pub(crate) use super::configuration::Qwen3_5MoEConfig;
