//! The record one throughput journey prints for the measurement it took.
//!
//! A journey prints one JSON object naming the model, the rates, the worker
//! build and the machine, so a run's output is enough to read the measurement
//! by. Nothing is kept on disk: release speed is recorded by
//! `cargo xtask measure-speed` in the ledger under `registry/release_speed`
//! (issue #2582), which replaced the history file these records once appended to.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::Value;

use ipc_protocol::WorkerDecodeAstroExpertCachePolicy;
use test_harness::test_run::TestRunContext;

use crate::performance_throughput::machine_specs::MachineSpecs;

/// The journey family one throughput record came from, so text, vision and
/// memory-cell measurements name themselves in a run's output.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ThroughputJourneyKind {
    Text,
    Vision,
    MemoryCeilingSweep,
    DecodeFullFit,
    /// The directed 5,000-input/300-output resident-model summarization under
    /// the automatic engine choice at every assigned memory ceiling, whether
    /// it selects the resident or the streaming engine; kept distinct from
    /// Text, whose shape is 10,000 tokens in and 500 out.
    ResidentMemoryCeilingProbe,
    /// One row of the streaming scoreboard (issue #2221). Those rows are
    /// taken at the product's default prompt chunk and are written to their
    /// own file.
    StreamingScoreboard,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryCellBudgetExhaustedStage {
    LaunchOrIdle,
    Warmup,
    Measured,
}

/// The memory-constraint cell one ceiling-sweep record ran under, plus the
/// residency and byte-traffic evidence that distinguishes deliberate expert
/// paging from kernel-paged swap collapse. The core pair is
/// `positional_read_byte_count` (bytes our bounded pager streamed) beside
/// `process_physical_disk_read_bytes` (bytes the operating system moved for
/// the worker process): a resident-mode cell with near-zero positional reads
/// and multi-gigabyte physical reads is thrashing through the kernel pager,
/// not serving.
#[derive(Clone, Debug, Serialize)]
pub struct MemoryCellRecord {
    pub configured_maximum_mlx_memory_bytes: u64,
    pub machine_recommended_working_set_bytes: u64,
    pub expert_residency_mode: String,
    pub expert_residency_total_layer_count: u32,
    pub expert_residency_resident_expert_count: u32,
    pub resident_expert_payload_bytes: u64,
    pub peak_mlx_memory_bytes: u64,
    pub final_active_mlx_memory_bytes: u64,
    pub positional_read_byte_count: u64,
    pub process_physical_disk_read_bytes: Option<u64>,
    pub process_physical_disk_written_bytes: Option<u64>,
    pub prefill_chunk_count: u64,
    pub maximum_decode_advance_seconds: f64,
    pub sweep_cell_order: u32,
    /// Physical disk bytes the worker process read while LOADING the model
    /// (from the model-loading attribution report). Beside
    /// `process_physical_disk_read_bytes` (generation-phase) it separates the
    /// one-time artifact load from serving-time page-ins: a resident-mode cell
    /// whose generation-phase reads climb into gigabytes is being paged by the
    /// kernel, not by the expert pager.
    pub model_loading_process_disk_read_bytes: Option<u64>,
    /// True when the cell exceeded the 45-second serving budget and was
    /// cancelled; the rates and token counts above are partial evidence, while
    /// the byte counters cover everything observed before the boundary.
    pub budget_exhausted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_exhausted_stage: Option<MemoryCellBudgetExhaustedStage>,
}

/// One throughput measurement captured from the measured completion of a
/// journey, recorded against the host and the wall-clock time it was taken on.
///
/// The first columns are ordered for human reading: the time, the model, the
/// journey family, the decode `astro_expert_cache` policy arm, then the prefill and decode
/// tokens-per-second as whole numbers. The remaining fields carry the host,
/// the git commit, and the input, output, and `astro_ssd_prompt_cache` token counts plus prefill
/// and decode latencies from the measured completion.
#[derive(Clone, Debug, Serialize)]
pub struct ThroughputRecord {
    pub timestamp: String,
    pub model_id: String,
    pub journey: ThroughputJourneyKind,
    pub decode_astro_expert_cache_policy: WorkerDecodeAstroExpertCachePolicy,
    pub prefill_tokens_per_second: u32,
    pub decode_tokens_per_second: u32,
    pub git_commit: Option<String>,
    pub worker_build_identity: WorkerBuildIdentity,
    pub machine: MachineSpecs,
    pub total_input_tokens: u32,
    pub total_output_tokens: u16,
    pub cached_tokens: u32,
    pub prefill_time_seconds: f64,
    pub decode_time_seconds: f64,
    /// The MLX memory ceiling the journey assigned to the worker, in bytes.
    /// Several ceilings share one journey kind (the engine-versus-ceiling
    /// cells all record as `resident_memory_ceiling_probe`), so without it
    /// their records could not be told apart. Omitted when the journey
    /// ran under the default serving ceiling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub configured_maximum_mlx_memory_bytes: Option<u64>,
    /// Which completion of the worker's life this measured run was: the
    /// in-worker warmup is ordinal one, the first measured completion ordinal
    /// two, later measured completions count on. Early ordinals are
    /// systematically slower (first-use graph compilation, cold warm tables):
    /// a cold worker's 607 tokens/s once sat beside a warmed worker's 2,445
    /// tokens/s from the same minute with nothing to tell them apart.
    /// First-touch rates and warmed rates must never be compared as if they
    /// were the same measurement configuration.
    pub completion_ordinal_in_worker: u32,
    /// Present only on memory-ceiling-sweep records.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_cell: Option<MemoryCellRecord>,
}

#[derive(Clone, Debug, Serialize)]
pub struct WorkerBuildIdentity {
    pub source_commit: String,
    pub source_is_dirty: bool,
    pub executable_sha256: String,
}

/// Formats a Unix-epoch millisecond value as a human-readable UTC timestamp such
/// as `2026-10-03 15:47:12 UTC`. The days-to-civil conversion follows Howard
/// Hinnant's published algorithm, so it stays correct across leap years without
/// pulling in a date library.
pub fn format_utc_timestamp(millis_since_unix_epoch: u64) -> String {
    let total_seconds = (millis_since_unix_epoch / 1_000) as i64;
    let days = total_seconds / 86_400;
    let seconds_of_day = total_seconds % 86_400;
    let hours = seconds_of_day / 3_600;
    let minutes = (seconds_of_day % 3_600) / 60;
    let seconds = seconds_of_day % 60;

    let z = days + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let month_from_march = (5 * doy + 2) / 153;
    let day = doy - (153 * month_from_march + 2) / 5 + 1;
    let month = if month_from_march < 10 {
        month_from_march + 3
    } else {
        month_from_march - 9
    };
    let actual_year = if month <= 2 { year + 1 } else { year };

    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02} UTC",
        actual_year, month, day, hours, minutes, seconds
    )
}

/// The JSON document a journey prints for one measurement.
pub fn throughput_record_json(record: &ThroughputRecord) -> Value {
    serde_json::to_value(record).unwrap_or(Value::Null)
}

/// Records the source identity captured by the runner for the built executable.
pub fn recorded_git_commit() -> Option<String> {
    TestRunContext::current().source_commit
}

pub fn recorded_worker_build_identity() -> WorkerBuildIdentity {
    worker_build_identity(&TestRunContext::current()).unwrap_or_else(|identity_error| {
        panic!("throughput requires cargo xtask test-run build identity: {identity_error}")
    })
}

fn worker_build_identity(run_context: &TestRunContext) -> Result<WorkerBuildIdentity, String> {
    let source_commit = run_context
        .source_commit
        .clone()
        .ok_or_else(|| "the runner did not record the source commit".to_owned())?;
    let executable_sha256 = run_context
        .workspace_binary_sha256
        .get("astronomical-inference-worker")
        .cloned()
        .ok_or_else(|| "the runner did not identify its built inference worker".to_owned())?;
    Ok(WorkerBuildIdentity {
        source_commit,
        source_is_dirty: run_context.source_is_dirty,
        executable_sha256,
    })
}

/// Current time as milliseconds since the Unix epoch.
pub fn current_unix_epoch_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
#[path = "throughput_record_tests.rs"]
mod throughput_record_tests;
