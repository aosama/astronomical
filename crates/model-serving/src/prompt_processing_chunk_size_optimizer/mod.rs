//! Model-independent online selection of prompt-processing chunk capacities.
//!
//! The optimizer owns measurement retention, context matching, exploration,
//! stale refresh, cumulative-latency planning, and persisted evidence. Model
//! adapters construct opaque contexts and execute selections; engine, IPC, and
//! presentation layers consume outcomes without adding optimizer policy here.

mod context;
mod context_statistics;
mod episode_latency_planner;
mod error;
mod measurement;
mod measurement_summary;
mod optimizer;
pub(crate) mod persistence;

pub use context::PromptProcessingMeasurementContext;
pub use error::PromptProcessingChunkSizeOptimizerError;
pub use measurement::PromptProcessingChunkMeasurement;
pub use measurement_summary::{
    CandidateChunkMeasurementSummary, CandidateMeasurementSource, CandidateMeasurementSummaries,
};
pub use optimizer::{
    PromptProcessingChunkSizeOptimizer, PromptProcessingChunkSizeSelection,
    PromptProcessingChunkSizeSelectionReason,
};
