//! Runtime-level entry points for the custom safetensors input/output
//! engines.
//!
//! These thin methods exist so callers see one coherent surface on
//! [`MlxRuntime`] while the engines themselves live under
//! `crate::safetensors`. The two loading calls differ in one essential way:
//!
//! - [`MlxRuntime::load_safetensors`] presents one whole file as-is through a
//!   retained descriptor — the ordinary path for dense weights and resident
//!   expert payloads.
//! - [`MlxRuntime::load_safetensors_from_bounded_ranges`] presents a
//!   *virtual* safetensors payload assembled from selected byte ranges of a
//!   source file — the core input/output primitive for expert paging, which
//!   must read only the router-selected expert rows and never the full shard.
//!
//! Both return lazy arrays: the file descriptors stay alive inside the
//! results precisely because MLX may still trigger reads when the arrays are
//! first evaluated (see `crate::safetensors::lazy_map`).

use std::fs::File;

use crate::{
    BoundedReadInterval, MlxRuntime, MlxRuntimeError, MlxSafetensors, PositionalFileReadMetrics,
    SafetensorsLoadResult,
};
use mlx_c_rust::MlxStream;

impl MlxRuntime {
    /// Loads a safetensors map from the caller's retained read-only file
    /// descriptor without reopening any mutable path identity.
    ///
    /// Opening by descriptor (rather than by path) means the bytes served to
    /// MLX cannot change under the process when a model directory is updated
    /// on disk mid-session: the descriptor pins the original file identity.
    pub fn load_safetensors(
        &self,
        weights_file: File,
        positional_file_read_metrics: Option<std::sync::Arc<PositionalFileReadMetrics>>,
    ) -> Result<MlxSafetensors, MlxRuntimeError> {
        MlxSafetensors::load(weights_file, positional_file_read_metrics)
    }

    /// Loads safetensors tensors from bounded multi-range reads on a source file.
    ///
    /// Maps a virtual safetensors payload to only the selected source-file ranges.
    /// This is the core input/output primitive for expert paging; see
    /// `crate::safetensors::bounded_range_reader` for the interval contract
    /// (virtual contiguity, non-overlapping source ranges, exact payload
    /// coverage) that the caller's manifest must satisfy.
    pub fn load_safetensors_from_bounded_ranges(
        &self,
        source_file: File,
        synthetic_header_bytes: Vec<u8>,
        intervals: Vec<BoundedReadInterval>,
        total_payload_bytes: u64,
        expert_file_read_metrics: Option<std::sync::Arc<PositionalFileReadMetrics>>,
    ) -> Result<SafetensorsLoadResult, MlxRuntimeError> {
        // A dedicated CPU stream keeps header materialization off the
        // graphics-processor stream; no kernel work is needed to interpret
        // safetensors bytes.
        let stream = MlxStream::default_cpu()?;
        crate::safetensors::lazy_map::load_safetensors_from_bounded_ranges(
            source_file,
            synthetic_header_bytes,
            intervals,
            total_payload_bytes,
            &stream,
            expert_file_read_metrics,
        )
    }
}
