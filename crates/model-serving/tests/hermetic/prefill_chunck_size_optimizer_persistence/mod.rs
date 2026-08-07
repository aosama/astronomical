//! Hermetic tests for prefill-chunck-size optimizer state persistence.
//!
//! Covers transition state, deterministic decisions, and recoverable invalid files.

use astronomical_model_serving::{
    PrefillChunckSizeOptimizer, PrefillChunckSizeOptimizerContext,
    PrefillChunckSizeOptimizerObservation,
};

mod invalid_state;
mod re_exploration;
mod round_trip;
mod state_file;
mod support;
