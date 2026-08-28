//! Laguna Poolside output-parser hermetic coverage.
//!
//! Streaming, well-formed layouts, and marker-defect permutations share one
//! literary parser fixture so happy-path and fail-open contracts stay aligned.

mod happy_path_permutations;
mod marker_permutations;
mod streaming;
mod support;
