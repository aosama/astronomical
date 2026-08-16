//! Laguna-owned prompt-processing chunk selection.
//!
//! The architecture-neutral optimizer owns learning. This adapter builds opaque
//! contexts from canonical Laguna descriptors and records a measurement only
//! after the selected chunk completes.

mod configuration;
mod execution_profile;
mod measurement_context;
mod optimization_outcome;
mod persisted_state;
mod sizer;

pub use configuration::LagunaPromptProcessingChunkSizerError;
pub use execution_profile::LagunaPromptProcessingExecutionProfile;
pub use sizer::LagunaPromptProcessingChunkSizer;
