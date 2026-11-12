//! The MLX memory a speech engine reports: an idle observation on request,
//! and one taken after each request has ended and returned its memory.
//!
//! A speech request is forward passes whose arrays are dropped when it ends
//! (a Phonon voice's cached state is the only thing kept, and it is a few
//! megabytes), so it shares the decision engines' memory shape and report.

use astronomical_runtime_integration::MlxRuntime;

use crate::memory::forward_pass_telemetry::forward_pass_memory_telemetry;
use crate::memory::telemetry::MlxMemoryTelemetry;

pub(crate) struct SpeechMemoryAccount {
    effective_mlx_memory_ceiling_bytes: u64,
    /// The size of the weight file, which stands in for the resident
    /// weights as it does for the decision engines.
    resident_weight_bytes: u64,
    post_cleanup_memory_telemetry: Option<MlxMemoryTelemetry>,
}

impl SpeechMemoryAccount {
    pub(crate) fn new(effective_mlx_memory_ceiling_bytes: u64, resident_weight_bytes: u64) -> Self {
        Self {
            effective_mlx_memory_ceiling_bytes,
            resident_weight_bytes,
            post_cleanup_memory_telemetry: None,
        }
    }

    pub(crate) fn observe(&self, runtime: &MlxRuntime) -> Option<MlxMemoryTelemetry> {
        let memory_snapshot = runtime.memory_snapshot().ok()?;
        Some(forward_pass_memory_telemetry(
            &memory_snapshot,
            self.effective_mlx_memory_ceiling_bytes,
            self.resident_weight_bytes,
        ))
    }

    /// Returns a finished request's memory to the system, then records what
    /// the next request starts from.
    pub(crate) fn clean_up_after_request(&mut self, runtime: &MlxRuntime) {
        // A failed clean-up leaves the memory cached, not lost; the next
        // request reuses it, so it is not worth failing the request over.
        let _cleanup_result = runtime.synchronize_gpu_stream_and_clear_allocator_cache();
        self.post_cleanup_memory_telemetry = self.observe(runtime);
    }

    pub(crate) fn take_post_cleanup_memory_telemetry(&mut self) -> Option<MlxMemoryTelemetry> {
        self.post_cleanup_memory_telemetry.take()
    }
}
