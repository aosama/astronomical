//! Durable, append-only historical record of throughput measurements.
//!
//! Each measurement appends a single JSON object as one line of a JSONL file so
//! that a machine's throughput can be compared across commits and dates. The log
//! lives beside the surface that produces it and is source-controlled, so the
//! trend travels with the code. The path is resolved from the
//! `ASTRONOMICAL_PERF_HISTORY` environment variable, falling back to
//! `tests/performance_throughput/throughput-history.jsonl`.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::Value;

use crate::performance_throughput::machine_specs::MachineSpecs;

const DEFAULT_HISTORY_RELATIVE_PATH: &str = "tests/performance_throughput/throughput-history.jsonl";
const HISTORY_ENV_VAR: &str = "ASTRONOMICAL_PERF_HISTORY";
const GIT_COMMIT_ENV_VAR: &str = "ASTRONOMICAL_PERF_COMMIT";

/// The journey family one throughput record came from. Every line of the
/// shared history log carries this field so text, vision, and memory-cell
/// measurements delineate themselves without separate files; lines written
/// before the field existed are text-journey measurements.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ThroughputJourneyKind {
    Text,
    Vision,
    MemoryCeilingSweep,
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
}

/// One throughput measurement captured from the measured completion of a
/// journey, recorded against the host and the wall-clock time it was taken on.
///
/// The first columns are ordered for human reading: the time, the model, the
/// journey family, then the prefill and decode tokens-per-second as whole
/// numbers. The remaining fields carry the host, the git commit, and the
/// input, output, and cached token counts plus prefill and decode latencies
/// from the measured completion.
#[derive(Clone, Debug, Serialize)]
pub struct ThroughputRecord {
    pub timestamp: String,
    pub model_id: String,
    pub journey: ThroughputJourneyKind,
    pub prefill_tokens_per_second: u32,
    pub decode_tokens_per_second: u32,
    pub git_commit: Option<String>,
    pub machine: MachineSpecs,
    pub total_input_tokens: u32,
    pub total_output_tokens: u16,
    pub cached_tokens: u32,
    pub prefill_time_seconds: f64,
    pub decode_time_seconds: f64,
    /// Present only on memory-ceiling-sweep records; omitted on every other
    /// journey so historical lines stay additive-compatible.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_cell: Option<MemoryCellRecord>,
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

/// Builds a JSON document for one measurement without persisting it, so callers
/// can print the record to stdout regardless of whether the history log is
/// writable.
pub fn throughput_record_json(record: &ThroughputRecord) -> Value {
    serde_json::to_value(record).unwrap_or(Value::Null)
}

/// Resolves the history log path from the environment or the default target
/// location.
pub fn history_log_path() -> PathBuf {
    std::env::var_os(HISTORY_ENV_VAR)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_HISTORY_RELATIVE_PATH))
}

/// Appends one measurement to the durable history log.
///
/// Persistence is best-effort: if the history directory cannot be created or the
/// log cannot be opened, the failure is returned to the caller, which reports it
/// but does not treat it as a measurement failure.
pub fn append_throughput_history(
    record: &ThroughputRecord,
    history_path: &std::path::Path,
) -> std::io::Result<()> {
    if let Some(parent) = history_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let document = serde_json::to_string(record).map_err(std::io::Error::other)?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(history_path)
        .map_err(std::io::Error::other)?;
    writeln!(file, "{document}")?;
    Ok(())
}

/// Resolves the recorded git commit, if the environment provides one.
pub fn recorded_git_commit() -> Option<String> {
    std::env::var(GIT_COMMIT_ENV_VAR)
        .ok()
        .filter(|value| !value.is_empty())
}

/// Current time as milliseconds since the Unix epoch.
pub fn current_unix_epoch_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as u64)
        .unwrap_or(0)
}
