use std::{collections::BTreeMap, time::Duration};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::time::Instant;

use super::model_process_metrics::WorkerPhysicalFootprint;

#[derive(Clone, Debug)]
pub(super) struct PrefillChunckMeasurement {
    pub(super) sequence_number: usize,
    pub(super) start_token: u32,
    pub(super) end_token: u32,
    pub(super) actual_prefill_chunck_tokens: u32,
    pub(super) selected_prefill_chunck_tokens: u32,
    pub(super) elapsed_millis: u64,
    pub(super) forward_prefill_chunck_elapsed_millis: u64,
    pub(super) mlx_active_memory_bytes: u64,
    pub(super) mlx_allocator_cache_memory_bytes: u64,
    pub(super) mlx_peak_memory_bytes: u64,
}

/// Stable digest of visible typed output events for one worker continuation.
pub(super) struct TypedOutputEventDigest {
    typed_output_event_sha256: Sha256,
}

impl TypedOutputEventDigest {
    pub(super) fn new() -> Self {
        Self {
            typed_output_event_sha256: Sha256::new(),
        }
    }

    pub(super) fn record_reasoning_fragment(&mut self, reasoning_fragment: &str) {
        self.record_textual_output_event(b"reasoning", reasoning_fragment);
    }

    pub(super) fn record_text_fragment(&mut self, text_fragment: &str) {
        self.record_textual_output_event(b"text", text_fragment);
    }

    pub(super) fn record_tool_call(
        &mut self,
        tool_call_index: u16,
        function_name: &str,
        arguments_json: &str,
    ) {
        self.record_event_component(b"tool_call");
        self.record_event_component(&tool_call_index.to_le_bytes());
        self.record_event_component(function_name.as_bytes());
        self.record_event_component(arguments_json.as_bytes());
    }

    pub(super) fn finish(self) -> String {
        let typed_output_event_digest_bytes = self.typed_output_event_sha256.finalize();
        let mut typed_output_event_digest_hex = String::with_capacity(64);
        for output_digest_byte in typed_output_event_digest_bytes {
            typed_output_event_digest_hex.push_str(&format!("{output_digest_byte:02x}"));
        }
        typed_output_event_digest_hex
    }

    fn record_textual_output_event(&mut self, output_event_type: &[u8], output_text: &str) {
        self.record_event_component(output_event_type);
        self.record_event_component(output_text.as_bytes());
    }

    fn record_event_component(&mut self, output_event_component: &[u8]) {
        self.typed_output_event_sha256.update(
            u64::try_from(output_event_component.len())
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        self.typed_output_event_sha256
            .update(output_event_component);
    }
}

impl PrefillChunckMeasurement {
    fn tokens_per_second(&self) -> f64 {
        if self.elapsed_millis == 0 {
            return 0.0;
        }
        f64::from(self.actual_prefill_chunck_tokens) / (self.elapsed_millis as f64 / 1_000.0)
    }

    fn context_bucket(&self) -> u32 {
        self.start_token / 32_768
    }
}

pub(super) struct PrefillMeasurementAccumulator {
    previous_cumulative_elapsed_millis: u64,
    previous_cumulative_processed_tokens: u32,
    chuncks: Vec<PrefillChunckMeasurement>,
}

impl PrefillMeasurementAccumulator {
    pub(super) const fn new() -> Self {
        Self {
            previous_cumulative_elapsed_millis: 0,
            previous_cumulative_processed_tokens: 0,
            chuncks: Vec::new(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record(
        &mut self,
        cumulative_processed_tokens: u32,
        cumulative_elapsed_millis: u64,
        forward_prefill_chunck_elapsed_millis: Option<u64>,
        completed_prefill_chunck_tokens: Option<u32>,
        mlx_active_memory_bytes: Option<u64>,
        mlx_allocator_cache_memory_bytes: Option<u64>,
        mlx_peak_memory_bytes: Option<u64>,
    ) {
        let (
            Some(selected_prefill_chunck_tokens),
            Some(forward_prefill_chunck_elapsed_millis),
            Some(mlx_active_memory_bytes),
            Some(mlx_allocator_cache_memory_bytes),
            Some(mlx_peak_memory_bytes),
        ) = (
            completed_prefill_chunck_tokens,
            forward_prefill_chunck_elapsed_millis,
            mlx_active_memory_bytes,
            mlx_allocator_cache_memory_bytes,
            mlx_peak_memory_bytes,
        )
        else {
            return;
        };
        if cumulative_processed_tokens <= self.previous_cumulative_processed_tokens {
            return;
        }
        let actual_prefill_chunck_tokens =
            cumulative_processed_tokens.saturating_sub(self.previous_cumulative_processed_tokens);
        let elapsed_millis =
            cumulative_elapsed_millis.saturating_sub(self.previous_cumulative_elapsed_millis);
        self.chuncks.push(PrefillChunckMeasurement {
            sequence_number: self.chuncks.len() + 1,
            start_token: self.previous_cumulative_processed_tokens,
            end_token: cumulative_processed_tokens,
            actual_prefill_chunck_tokens,
            selected_prefill_chunck_tokens,
            elapsed_millis,
            forward_prefill_chunck_elapsed_millis,
            mlx_active_memory_bytes,
            mlx_allocator_cache_memory_bytes,
            mlx_peak_memory_bytes,
        });
        self.previous_cumulative_processed_tokens = cumulative_processed_tokens;
        self.previous_cumulative_elapsed_millis = cumulative_elapsed_millis;
    }

    pub(super) fn chuncks(&self) -> &[PrefillChunckMeasurement] {
        &self.chuncks
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_prefill_benchmark_report(
    benchmark_mode_label: &str,
    fixed_prefill_chunck_tokens: Option<u32>,
    target_prompt_tokens: usize,
    maximum_output_tokens: u16,
    worker_startup_seconds: f64,
    request_started_at: Instant,
    first_output_at: Instant,
    response_completed_at: Instant,
    generated_token_count: u16,
    completion_reason: String,
    typed_output_event_digest: String,
    final_expert_memory_mode: &'static str,
    optimizer_state_loaded_before_run: bool,
    maximum_gpu_wired_memory_bytes: usize,
    idle_worker_footprint: WorkerPhysicalFootprint,
    first_output_footprint: WorkerPhysicalFootprint,
    completed_footprint: WorkerPhysicalFootprint,
    prefill_measurements: &PrefillMeasurementAccumulator,
) -> Value {
    let prompt_processing_seconds = first_output_at
        .duration_since(request_started_at)
        .as_secs_f64();
    let generation_seconds = response_completed_at
        .duration_since(first_output_at)
        .as_secs_f64();
    let total_request_seconds = response_completed_at
        .duration_since(request_started_at)
        .as_secs_f64();
    let native_prefill_millis = prefill_measurements.previous_cumulative_elapsed_millis;
    let native_prefill_seconds = native_prefill_millis as f64 / 1_000.0;
    let native_prefill_tokens = prefill_measurements.previous_cumulative_processed_tokens;
    let chunck_rates = prefill_measurements
        .chuncks()
        .iter()
        .map(PrefillChunckMeasurement::tokens_per_second)
        .collect::<Vec<_>>();
    let mut selected_chunck_histogram = BTreeMap::<u32, usize>::new();
    for chunck_measurement in prefill_measurements.chuncks() {
        *selected_chunck_histogram
            .entry(chunck_measurement.selected_prefill_chunck_tokens)
            .or_default() += 1;
    }
    let chunck_reports = prefill_measurements
        .chuncks()
        .iter()
        .map(|chunck_measurement| {
            json!({
                "actual_prefill_chunck_tokens": chunck_measurement.actual_prefill_chunck_tokens,
                "context_bucket": chunck_measurement.context_bucket(),
                "elapsed_millis": chunck_measurement.elapsed_millis,
                "end_token": chunck_measurement.end_token,
                "forward_prefill_chunck_elapsed_millis": chunck_measurement.forward_prefill_chunck_elapsed_millis,
                "mlx_active_memory_bytes": chunck_measurement.mlx_active_memory_bytes,
                "mlx_allocator_cache_memory_bytes": chunck_measurement.mlx_allocator_cache_memory_bytes,
                "mlx_peak_memory_bytes": chunck_measurement.mlx_peak_memory_bytes,
                "selected_prefill_chunck_tokens": chunck_measurement.selected_prefill_chunck_tokens,
                "sequence_number": chunck_measurement.sequence_number,
                "start_token": chunck_measurement.start_token,
                "tokens_per_second": chunck_measurement.tokens_per_second(),
            })
        })
        .collect::<Vec<_>>();

    json!({
        "cached_tokens": 0,
        "chunck_count": prefill_measurements.chuncks().len(),
        "chunck_rates": {
            "maximum": chunck_rates.iter().copied().reduce(f64::max),
            "median": nearest_rank_percentile(&chunck_rates, 0.5),
            "minimum": chunck_rates.iter().copied().reduce(f64::min),
            "p95": nearest_rank_percentile(&chunck_rates, 0.95),
        },
        "chuncks": chunck_reports,
        "completion_reason": completion_reason,
        "completion_tokens": generated_token_count,
        "final_expert_memory_mode": final_expert_memory_mode,
        "fixed_prefill_chunck_tokens": fixed_prefill_chunck_tokens,
        "generation_seconds": generation_seconds,
        "generation_tokens_per_second": (generated_token_count > 1).then(|| f64::from(generated_token_count - 1) / generation_seconds),
        "gpu_wired_limit_bytes": maximum_gpu_wired_memory_bytes,
        "maximum_output_tokens": maximum_output_tokens,
        "mode": benchmark_mode_label,
        "native_prefill_seconds": native_prefill_seconds,
        "native_prefill_tokens": native_prefill_tokens,
        "native_prefill_tokens_per_second": f64::from(native_prefill_tokens) / native_prefill_seconds,
        "optimizer_state_loaded_before_run": optimizer_state_loaded_before_run,
        "physical_footprint": {
            "completed_current_bytes": completed_footprint.current_bytes,
            "completed_peak_bytes": completed_footprint.peak_bytes,
            "first_output_current_bytes": first_output_footprint.current_bytes,
            "first_output_peak_bytes": first_output_footprint.peak_bytes,
            "idle_worker_current_bytes": idle_worker_footprint.current_bytes,
            "idle_worker_peak_bytes": idle_worker_footprint.peak_bytes,
        },
        "prompt_processing_seconds": prompt_processing_seconds,
        "prompt_processing_tokens_per_second": target_prompt_tokens as f64 / prompt_processing_seconds,
        "prompt_tokens": target_prompt_tokens,
        "selected_chunck_histogram": selected_chunck_histogram,
        "total_request_seconds": total_request_seconds,
        "typed_output_event_sha256": typed_output_event_digest,
        "worker_startup_seconds": worker_startup_seconds,
    })
}

fn nearest_rank_percentile(measurements: &[f64], percentile: f64) -> f64 {
    if measurements.is_empty() {
        return 0.0;
    }
    let mut sorted_measurements = measurements.to_vec();
    sorted_measurements.sort_by(f64::total_cmp);
    let nearest_rank = (percentile * sorted_measurements.len() as f64).ceil() as usize;
    sorted_measurements[nearest_rank
        .saturating_sub(1)
        .min(sorted_measurements.len() - 1)]
}

#[test]
fn should_preserve_event_type_and_order_in_typed_output_event_digest() {
    let mut text_then_reasoning_digest = TypedOutputEventDigest::new();
    text_then_reasoning_digest.record_text_fragment("same payload");
    text_then_reasoning_digest.record_reasoning_fragment("same payload");

    let mut reasoning_then_text_digest = TypedOutputEventDigest::new();
    reasoning_then_text_digest.record_reasoning_fragment("same payload");
    reasoning_then_text_digest.record_text_fragment("same payload");

    assert_ne!(
        text_then_reasoning_digest.finish(),
        reasoning_then_text_digest.finish(),
        "a continuation digest must distinguish event type and order"
    );
}

#[test]
fn should_include_typed_output_digest_and_final_expert_memory_mode_in_benchmark_report() {
    let mut prefill_measurements = PrefillMeasurementAccumulator::new();
    prefill_measurements.record(
        1_023,
        1_000,
        Some(900),
        Some(4_096),
        Some(9_000_000_000),
        Some(0),
        Some(9_500_000_000),
    );
    let request_started_at = Instant::now();
    let first_output_at = request_started_at + Duration::from_secs(1);
    let response_completed_at = first_output_at + Duration::from_secs(2);
    let worker_physical_footprint = WorkerPhysicalFootprint {
        current_bytes: 1,
        peak_bytes: 2,
    };

    let benchmark_report = build_prefill_benchmark_report(
        "fixed_candidate",
        Some(4_096),
        1_024,
        512,
        1.0,
        request_started_at,
        first_output_at,
        response_completed_at,
        512,
        "MaximumOutputTokens".to_owned(),
        "abc123".to_owned(),
        "paged",
        false,
        10_000_000_000,
        worker_physical_footprint,
        worker_physical_footprint,
        worker_physical_footprint,
        &prefill_measurements,
    );

    assert_eq!(benchmark_report["typed_output_event_sha256"], "abc123");
    assert_eq!(benchmark_report["final_expert_memory_mode"], "paged");
}

#[test]
fn should_derive_each_prefill_chunck_from_cumulative_worker_progress() {
    let mut prefill_measurements = PrefillMeasurementAccumulator::new();

    prefill_measurements.record(
        2_048,
        1_500,
        Some(1_400),
        Some(2_048),
        Some(11_000),
        Some(12_000),
        Some(13_000),
    );
    prefill_measurements.record(
        4_096,
        2_900,
        Some(1_300),
        Some(2_048),
        Some(14_000),
        Some(15_000),
        Some(16_000),
    );

    assert_eq!(prefill_measurements.chuncks().len(), 2);
    assert_eq!(
        prefill_measurements.chuncks()[0].actual_prefill_chunck_tokens,
        2_048
    );
    assert_eq!(prefill_measurements.chuncks()[0].elapsed_millis, 1_500);
    assert_eq!(
        prefill_measurements.chuncks()[1].actual_prefill_chunck_tokens,
        2_048
    );
    assert_eq!(prefill_measurements.chuncks()[1].elapsed_millis, 1_400);
    assert_eq!(
        prefill_measurements.chuncks()[1].forward_prefill_chunck_elapsed_millis,
        1_300
    );
    assert_eq!(
        prefill_measurements.chuncks()[1].mlx_active_memory_bytes,
        14_000
    );
}

#[test]
fn should_select_nearest_rank_percentiles_for_chunck_measurements() {
    let measurements = [10.0, 20.0, 30.0, 40.0, 50.0];

    assert_eq!(nearest_rank_percentile(&measurements, 0.5), 30.0);
    assert_eq!(nearest_rank_percentile(&measurements, 0.95), 50.0);
}
