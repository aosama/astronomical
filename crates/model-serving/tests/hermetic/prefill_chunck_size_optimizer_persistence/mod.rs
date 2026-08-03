//! Hermetic tests for prefill-chunck-size optimizer state persistence.
//!
//! Test portfolio generated through verbalized sampling: each test scenario is
//! scored by (likelihood × impact) and the highest-priority plus most-diverse
//! cases are selected. Rare-but-dangerous scenarios (corrupt files, model
//! mismatches, re-exploration state) are deliberately included because they
//! cause silent cold-start regressions if missed.

use astronomical_model_serving::{
    PrefillChunckSizeOptimizer, PrefillChunckSizeOptimizerContext,
    PrefillChunckSizeOptimizerDecisionReason, PrefillChunckSizeOptimizerObservation,
};

mod invalid_state;
mod re_exploration;
mod round_trip;
mod state_file;
mod support;
