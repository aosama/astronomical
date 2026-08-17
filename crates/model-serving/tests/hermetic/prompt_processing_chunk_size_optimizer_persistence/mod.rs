//! Hermetic tests for prompt-processing chunk-size optimizer state persistence.
//!
//! Covers measurements, deterministic selections, and recoverable invalid files.

use astronomical_model_serving::{
    PromptProcessingChunkMeasurement, PromptProcessingChunkSizeOptimizer,
    PromptProcessingMeasurementContext,
};

mod invalid_state;
mod round_trip;
mod state_file;
mod support;
