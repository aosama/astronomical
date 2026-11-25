//! What the generation loop asks of a loaded text model: its runtime and
//! limits ([`TextServingModel`]), what it holds in MLX memory
//! ([`memory_breakdown::TextModelMemoryOwnership`]), and the forwards the
//! loop drives ([`text_model::TextModel`]).

pub mod memory_breakdown;
pub mod model_chunking_configuration;
pub mod serving_model;
pub mod text_model;

pub use serving_model::TextServingModel;
