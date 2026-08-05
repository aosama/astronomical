use std::fs::File;

use crate::{
    BoundedReadInterval, MlxRuntime, MlxRuntimeError, MlxSafetensors, PositionalFileReadMetrics,
    SafetensorsLoadResult, mlx_stream::MlxStream,
};

impl MlxRuntime {
    /// Loads a safetensors map from the caller's retained read-only file
    /// descriptor without reopening any mutable path identity.
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
    /// This is the core input/output primitive for expert paging.
    pub fn load_safetensors_from_bounded_ranges(
        &self,
        source_file: File,
        synthetic_header_bytes: Vec<u8>,
        intervals: Vec<BoundedReadInterval>,
        total_payload_bytes: u64,
        expert_file_read_metrics: Option<std::sync::Arc<PositionalFileReadMetrics>>,
    ) -> Result<SafetensorsLoadResult, MlxRuntimeError> {
        let stream = MlxStream::default_cpu()?;
        crate::mlx_safetensors::load_safetensors_from_bounded_ranges(
            source_file,
            synthetic_header_bytes,
            intervals,
            total_payload_bytes,
            &stream,
            expert_file_read_metrics,
        )
    }

    pub(crate) const fn gpu_stream(&self) -> &MlxStream {
        &self.gpu_stream
    }
}
