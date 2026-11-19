//! Native Qwen3-4B prompt preparation and request-scoped FLUX text conditioning.

#[cfg(test)]
mod tests;

mod batch;
mod error;
mod prompt;
mod tokenizer;

#[cfg(feature = "direct-mlx")]
mod conditioner;
#[cfg(feature = "direct-mlx")]
mod layer;
#[cfg(feature = "direct-mlx")]
mod state;
#[cfg(feature = "direct-mlx")]
mod weights;

pub use batch::FLUX2_KLEIN_CONDITIONING_SEQUENCE_LENGTH;
pub use error::Flux2KleinTextConditioningError;
pub use tokenizer::Flux2KleinTokenizer;

#[cfg(feature = "direct-mlx")]
pub use conditioner::Flux2KleinTextConditioner;
#[cfg(feature = "direct-mlx")]
pub use state::{
    Flux2KleinTextConditioning, Flux2KleinTextConditioningAdvance, Flux2KleinTextConditioningState,
};
