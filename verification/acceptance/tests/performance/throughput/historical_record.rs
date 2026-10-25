//! Durable, append-only historical record of throughput measurements.
//!
//! Each measurement appends a single JSON object as one line of a JSONL file so
//! that a machine's throughput can be compared across commits and dates. The log
//! lives beside the surface that produces it and is source-controlled, so the
//! trend travels with the code. The runner configuration's
//! `throughput_history_path` selects a destination, falling back to the
//! source-controlled ledger under `verification/acceptance/tests/performance/throughput/`.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::Value;

use astronomical_ipc_protocol::WorkerDecodeExpertCachePolicy;
use astronomical_test_harness::test_run::TestRunContext;

use crate::performance_throughput::machine_specs::MachineSpecs;

/// Repo-root-relative path of the source-controlled throughput history.
///
/// The default is anchored to the compile-time manifest directory of the
/// acceptance package rather than to the test process's working directory: the
/// standard acceptance launcher starts test binaries with the package
/// directory as their working directory, so a launch-directory-relative
/// default resolved to the package's own path repeated inside itself and
/// measurements forked into an unintended doubled nested copy instead of this
/// ledger. `CARGO_MANIFEST_DIR` is baked in at compile time, so every launch
/// directory resolves to the same canonical file. It must stay in sync with
/// the file's actual location under
/// `verification/acceptance/tests/performance/throughput/`; the Oct-13 test-tree
/// reorganization moved that file and left this constant stale, forking new
/// measurements into an untracked sibling until the drift was noticed.
const DEFAULT_HISTORY_RELATIVE_PATH: &str =
    "verification/acceptance/tests/performance/throughput/throughput-history.jsonl";

/// The journey family one throughput record came from. Every line of the
/// shared history log carries this field so text, vision, and memory-cell
/// measurements delineate themselves without separate files; lines written
/// before this field existed are text-journey measurements. Historical rows
/// without an explicit decode-cache policy were produced by the baseline path.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ThroughputJourneyKind {
    Text,
    Vision,
    MemoryCeilingSweep,
    DecodeFullFit,
    /// The directed 5,000-input/300-output resident-model comparison probe at
    /// two assigned memory ceilings; kept distinct from Text so the canonical
    /// >=10,000-input/500-output series stays clean.
    ResidentMemoryCeilingProbe,
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
/// journey family, the decode-cache policy arm, then the prefill and decode
/// tokens-per-second as whole numbers. The remaining fields carry the host,
/// the git commit, and the input, output, and cached token counts plus prefill
/// and decode latencies from the measured completion.
#[derive(Clone, Debug, Serialize)]
pub struct ThroughputRecord {
    pub timestamp: String,
    pub model_id: String,
    pub journey: ThroughputJourneyKind,
    pub decode_expert_cache_policy: WorkerDecodeExpertCachePolicy,
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
    /// Present only on memory-ceiling-sweep records; omitted on every other
    /// journey so historical lines stay additive-compatible.
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

/// Builds a JSON document for one measurement without persisting it, so callers
/// can print the record to stdout regardless of whether the history log is
/// writable.
pub fn throughput_record_json(record: &ThroughputRecord) -> Value {
    serde_json::to_value(record).unwrap_or(Value::Null)
}

/// Resolves the history log path from explicit runner inputs or the workspace default.
pub fn history_log_path() -> PathBuf {
    resolve_history_log_path(
        TestRunContext::current()
            .configuration
            .throughput_history_path
            .as_deref(),
    )
}

fn resolve_history_log_path(configured_history_path: Option<&Path>) -> PathBuf {
    configured_history_path
        .map(Path::to_path_buf)
        .unwrap_or_else(default_history_log_path)
}

/// Anchors the default ledger location to the acceptance package's manifest
/// directory (two levels above it is the workspace root), so the resolution
/// cannot depend on the launch directory.
fn default_history_log_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(DEFAULT_HISTORY_RELATIVE_PATH)
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
#[path = "historical_record_tests.rs"]
mod historical_record_tests;
