use crate::{MlxRuntimeError, raw};

/// Owned MLX stream handle used to preserve runtime thread affinity.
#[derive(Debug)]
pub(crate) struct MlxStream(raw::mlx_stream);

impl MlxStream {
    pub(crate) fn default_cpu() -> Result<Self, MlxRuntimeError> {
        // SAFETY: The runtime error handler is installed and the returned
        // stream is placed immediately under RAII ownership.
        let raw_stream = unsafe { raw::mlx_default_cpu_stream_new() };
        Self::from_raw(raw_stream, "acquire the default MLX CPU stream")
    }

    pub(crate) fn default_gpu() -> Result<Self, MlxRuntimeError> {
        // SAFETY: The runtime error handler is installed and the returned
        // stream is placed immediately under RAII ownership.
        let raw_stream = unsafe { raw::mlx_default_gpu_stream_new() };
        Self::from_raw(raw_stream, "acquire the default MLX GPU stream")
    }

    pub(crate) const fn raw(&self) -> raw::mlx_stream {
        self.0
    }

    fn from_raw(
        raw_stream: raw::mlx_stream,
        operation: &'static str,
    ) -> Result<Self, MlxRuntimeError> {
        if raw_stream.ctx.is_null() {
            return Err(MlxRuntimeError::RuntimeOperation {
                operation,
                description: "MLX returned an empty stream handle".to_owned(),
            });
        }
        Ok(Self(raw_stream))
    }
}

impl Drop for MlxStream {
    fn drop(&mut self) {
        // SAFETY: This owner releases its live stream handle exactly once.
        unsafe {
            raw::mlx_stream_free(self.0);
        }
    }
}
