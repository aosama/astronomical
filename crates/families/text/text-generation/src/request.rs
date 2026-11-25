//! The token-level request a text family prepares for the generation loop:
//! its prompt tokens, sampling strategy, thinking allowance and processed
//! images, and the sampler and thinking-budget state the loop drives.

mod inference_request;
mod processed_image;
#[cfg(feature = "direct-mlx")]
pub mod sampler;
pub mod thinking_budget;

pub use inference_request::{TextInferenceRequest, TextSamplingStrategy};
pub use processed_image::{ImageGrid, ProcessedImage};
#[cfg(feature = "direct-mlx")]
pub use thinking_budget::minimum_bounded_output_token_count;
