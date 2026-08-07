use std::hash::{DefaultHasher, Hash, Hasher};

use astronomical_model_serving::MlxMemoryTelemetry;

#[derive(Default)]
pub(super) struct BenchmarkMeasurement {
    pub(super) generated_token_ids: Vec<u32>,
    pub(super) completed_prefill_chunck_tokens: Vec<usize>,
    pub(super) prefill_elapsed_millis: u64,
    pub(super) time_to_first_token_seconds: f64,
    pub(super) generation_elapsed_seconds: f64,
    pub(super) total_request_elapsed_seconds: f64,
    pub(super) maximum_active_mlx_memory_bytes: u64,
    pub(super) maximum_peak_mlx_memory_bytes: u64,
}

impl BenchmarkMeasurement {
    pub(super) fn generated_token_id_fingerprint(&self) -> u64 {
        let mut generated_token_id_hasher = DefaultHasher::new();
        self.generated_token_ids
            .hash(&mut generated_token_id_hasher);
        generated_token_id_hasher.finish()
    }

    pub(super) fn tokens_per_second(&self) -> f64 {
        self.generated_token_ids.len().saturating_sub(1) as f64
            / self.generation_elapsed_seconds.max(f64::EPSILON)
    }

    pub(super) fn record_mlx_memory_telemetry(
        &mut self,
        mlx_memory_telemetry: Option<MlxMemoryTelemetry>,
    ) {
        if let Some(mlx_memory_telemetry) = mlx_memory_telemetry {
            self.maximum_active_mlx_memory_bytes = self
                .maximum_active_mlx_memory_bytes
                .max(mlx_memory_telemetry.active_memory_bytes);
            self.maximum_peak_mlx_memory_bytes = self
                .maximum_peak_mlx_memory_bytes
                .max(mlx_memory_telemetry.peak_memory_bytes);
        }
    }
}
