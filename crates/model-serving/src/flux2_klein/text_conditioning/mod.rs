//! Native Qwen3-4B prompt preparation and request-scoped FLUX text conditioning.

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

pub(crate) use batch::FLUX2_KLEIN_CONDITIONING_SEQUENCE_LENGTH;
pub(crate) use error::Flux2KleinTextConditioningError;
pub(crate) use tokenizer::Flux2KleinTokenizer;

#[cfg(feature = "direct-mlx")]
pub(crate) use conditioner::Flux2KleinTextConditioner;
#[cfg(feature = "direct-mlx")]
pub(crate) use state::{
    Flux2KleinTextConditioning, Flux2KleinTextConditioningAdvance, Flux2KleinTextConditioningState,
};
