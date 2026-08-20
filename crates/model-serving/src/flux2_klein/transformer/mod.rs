//! Native MLX execution for the official FLUX.2 Klein 4B denoising transformer.
//!
//! Loading, arithmetic, block equations, and bounded orchestration remain
//! separate so the image engine can cancel and attribute work between GPU roots.

mod blocks;
mod error;
mod execution;
mod geometry;
mod math;
mod preparation;
mod weights;

pub use error::{Flux2KleinTransformerError, Flux2KleinTransformerGeometryError};
pub use execution::{
    Flux2KleinBlockGroupEvent, Flux2KleinBlockKind, Flux2KleinForwardAdvance,
    Flux2KleinForwardState, Flux2KleinTransformer, Flux2KleinTransformerOutput,
};
pub use geometry::Flux2KleinTransformerGeometry;
#[doc(hidden)]
pub use math::apply_rope_for_component_oracle;
pub use preparation::{Flux2KleinComponentOracle, Flux2KleinTransformerInputs};
pub use weights::Flux2KleinTransformerWeights;
