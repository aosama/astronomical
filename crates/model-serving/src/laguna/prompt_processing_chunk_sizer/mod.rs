//! Laguna-owned deterministic prompt-processing chunk selection.

mod configuration;
mod sizer;

pub use configuration::LagunaPromptProcessingChunkSizerError;
pub use sizer::LagunaPromptProcessingChunkSizer;
