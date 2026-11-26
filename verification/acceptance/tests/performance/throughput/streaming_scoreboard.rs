//! Streaming scoreboard: the measurements the streaming memory redesign is
//! judged by, taken the same way before and after each of its phases
//! (issue #2221).
//!
//! One cell is one model under an MLX limit small enough that the model has
//! to stream from storage. Each cell runs the standard serving journey — a
//! short warmup, then about 10,000 prompt tokens and 500 reply tokens of the
//! Romeo and Juliet fixture — with the `astro_ssd_prompt_cache` off,
//! performance attribution off and the product's default prompt chunk, so a
//! row describes what a person running the app gets.
//!
//! A row reports speed and, beside it, the memory behaviour the redesign
//! targets: how much of the expert entitlement was filled, how many times a
//! prompt chunk hit the limit, how many bytes the measured request read from
//! storage, and how much of the limit no owner could explain. The memory
//! figures are read once, when the measured request has finished.
//!
//! Rows are printed and appended to `streaming-scoreboard.jsonl` under the
//! acceptance evidence root. They do not compare with the throughput
//! journeys' records, which are taken at a fixed 2,048-token chunk.
//!
//! Each cell is its own process because MLX limits are process-global. The
//! `measure-streaming-scoreboard` alias in `.cargo/config.toml` runs them.

use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Duration;

use astronomical_daemon::WorkerHealthSnapshot;
use ipc_protocol::{ChatImageInput, WorkerDecodeAstroExpertCachePolicy};
use serde::Serialize;
use serial_test::serial;

use crate::performance_throughput::support::{
    self as throughput_support, JourneyRunOutcome, PromptChunking, ThroughputJourney,
};
use crate::performance_throughput::throughput_record::{
    ThroughputJourneyKind, WorkerBuildIdentity, current_unix_epoch_millis, format_utc_timestamp,
    recorded_worker_build_identity,
};
use crate::support;

const WARMUP_INPUT_INSTRUCTION: &str = "Continue the supplied Romeo and Juliet story above.";
const WARMUP_ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_warmup_romeo_and_juliet.txt");
const WARMUP_MAXIMUM_OUTPUT_TOKENS: u16 = 50;
const MEASURED_INPUT_INSTRUCTION: &str =
    "Continue the supplied Romeo and Juliet story above for approximately 500 tokens.";
const MEASURED_ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_10000_tokens_romeo_and_juliet.txt");
const MEASURED_MAXIMUM_OUTPUT_TOKENS: u16 = 500;
const TEMPERATURE_THOUSANDTHS: u16 = 1_000;
/// A model streaming most of its layers reads tens of gigabytes per request,
/// so a cell gets minutes, not the one minute the ceiling sweep allows.
const SCOREBOARD_SERVING_BUDGET: Duration = Duration::from_secs(20 * 60);
/// Leaves a minute beyond the serving budget for worker shutdown and the row.
const SCOREBOARD_TEST_TIMEOUT: Duration = Duration::from_secs(21 * 60);
const BYTES_PER_GIBIBYTE: u64 = 1 << 30;
const SCOREBOARD_EVIDENCE_DIRECTORY_NAME: &str = "streaming-scoreboard";
const SCOREBOARD_FILE_NAME: &str = "streaming-scoreboard.jsonl";
/// The worker's warning when a prompt chunk hits the MLX limit.
const LIMIT_OVERFLOW_LOG_MARKER: &str = "exceeded the MLX active-memory limit";

macro_rules! streaming_scoreboard_cell {
    ($test_function_name:ident, $model_id_function:path, $limit_gibibytes:literal) => {
        #[tokio::test(flavor = "multi_thread")]
        #[ignore = concat!(
                                    "loads a real model under a ",
                                    stringify!($limit_gibibytes),
                                    " GiB MLX limit and records one streaming scoreboard row"
                                )]
        #[serial]
        async fn $test_function_name() {
            tokio::time::timeout(
                SCOREBOARD_TEST_TIMEOUT,
                run_scoreboard_cell($model_id_function(), $limit_gibibytes),
            )
            .await
            .expect("the scoreboard cell must finish inside its test timeout");
        }
    };
}

streaming_scoreboard_cell!(
    should_record_the_small_sparse_model_scoreboard_row_at_an_11_gib_limit,
    support::small_sparse_moe_streaming_model_id,
    11
);
streaming_scoreboard_cell!(
    should_record_the_large_sparse_model_scoreboard_row_at_a_19_gib_limit,
    support::large_sparse_moe_streaming_model_id,
    19
);
streaming_scoreboard_cell!(
    should_record_the_largest_sparse_model_scoreboard_row_at_a_32_gib_limit,
    support::largest_sparse_moe_streaming_model_id,
    32
);
streaming_scoreboard_cell!(
    should_record_the_dense_model_scoreboard_row_at_a_3_gib_limit,
    support::dense_vision_model_id,
    3
);

/// One model under one limit, as one line of the scoreboard.
#[derive(Clone, Debug, PartialEq, Serialize)]
struct StreamingScoreboardRow {
    timestamp: String,
    source_commit: String,
    /// Whether the worker was built from a tree with uncommitted changes, in
    /// which case the commit alone does not identify what was measured.
    source_is_dirty: bool,
    model_id: String,
    limit_bytes: u64,
    prompt_token_count: u32,
    reply_token_count: u16,
    prompt_tokens_per_second: f64,
    reply_tokens_per_second: f64,
    peak_memory_bytes: u64,
    final_active_memory_bytes: u64,
    kept_expert_payload_bytes: u64,
    unfilled_expert_entitlement_bytes: u64,
    /// Kept expert bytes over kept plus unfilled; `None` for a model with no
    /// expert entitlement at all, such as a dense model.
    expert_entitlement_share_filled: Option<f64>,
    dense_kept_layer_count: Option<u32>,
    dense_layer_count: Option<u32>,
    /// `None` when the worker left no log to count in.
    limit_overflow_count: Option<u64>,
    /// Expert bytes the measured request read while reading its prompt and
    /// while writing its reply; `None` when the worker published no expert
    /// streaming statistics.
    measured_prompt_storage_read_bytes: Option<u64>,
    measured_reply_storage_read_bytes: Option<u64>,
    unexplained_headroom_bytes: u64,
    owner_overrun_bytes: u64,
    /// What the memory ledger saw over the measured request; `None` from a
    /// worker whose engine the ledger does not watch.
    memory_ledger: Option<LedgerScoreboardReading>,
}

/// The memory ledger's worst readings over the measured request, and the
/// accounts it ended with.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct LedgerScoreboardReading {
    observed_step_count: u64,
    largest_unaccounted_bytes: u64,
    largest_overcounted_bytes: u64,
    largest_kept_over_capacity_bytes: u64,
    largest_reading_step_peak_beyond_held_bytes: u64,
    largest_writing_step_peak_beyond_held_bytes: u64,
    probed_mlx_working_buffer_bytes: Option<u64>,
    lowest_kept_share_while_reading_permille: Option<u32>,
    mlx_working_buffer_bytes: u64,
    kept_weight_capacity_bytes: u64,
}

async fn run_scoreboard_cell(model_id: &str, limit_gibibytes: u64) {
    let worker_build_identity = recorded_worker_build_identity();
    let limit_bytes = limit_gibibytes * BYTES_PER_GIBIBYTE;
    let run_outcome =
        throughput_support::measure_throughput(model_id, &scoreboard_journey(limit_bytes)).await;
    let scoreboard_row = compose_scoreboard_row(&run_outcome, limit_bytes, worker_build_identity);
    publish_scoreboard_row(&scoreboard_row);
    // The row is published first so a cell that ran out of time or produced
    // an off-shape request still leaves its evidence behind.
    assert!(
        !run_outcome.measured_budget_exhausted,
        "the {limit_gibibytes} GiB scoreboard cell for {model_id} ran out of its serving budget"
    );
    throughput_support::assert_measurement_shape(&run_outcome.measurement);
}

fn scoreboard_journey(limit_bytes: u64) -> ThroughputJourney {
    ThroughputJourney {
        journey_kind: ThroughputJourneyKind::StreamingScoreboard,
        decode_astro_expert_cache_policy: WorkerDecodeAstroExpertCachePolicy::Baseline,
        warmup_input_prompt: throughput_support::romeo_and_juliet_warmup_prompt(
            WARMUP_INPUT_INSTRUCTION,
            WARMUP_ROMEO_AND_JULIET_SOURCE,
        ),
        warmup_images: Vec::<ChatImageInput>::new(),
        warmup_output_tokens: WARMUP_MAXIMUM_OUTPUT_TOKENS,
        measured_input_prompt: format!(
            "{MEASURED_INPUT_INSTRUCTION}\n\n{MEASURED_ROMEO_AND_JULIET_SOURCE}"
        ),
        measured_images: Vec::<ChatImageInput>::new(),
        measured_output_tokens: MEASURED_MAXIMUM_OUTPUT_TOKENS,
        temperature_thousandths: TEMPERATURE_THOUSANDTHS,
        maximum_mlx_memory_bytes: Some(limit_bytes),
        attribution_enabled: false,
        prompt_chunking: PromptChunking::ProductDefault,
        journey_timeout: SCOREBOARD_SERVING_BUDGET,
        model_directory: None,
    }
}

fn compose_scoreboard_row(
    run_outcome: &JourneyRunOutcome,
    limit_bytes: u64,
    worker_build_identity: WorkerBuildIdentity,
) -> StreamingScoreboardRow {
    let measurement = &run_outcome.measurement;
    let final_health = &run_outcome.final_health_snapshot;
    let memory_reading = memory_reading_at_request_end(final_health);
    let expert_residency = final_health.expert_residency.as_ref();
    let kept_expert_payload_bytes =
        expert_residency.map_or(0, |residency| residency.resident_expert_payload_bytes);
    let dense_layers = expert_residency.and_then(|residency| residency.dense_layers.as_ref());
    let storage_reads = measured_request_storage_reads(
        run_outcome.health_snapshot_after_warmup.as_ref(),
        final_health,
    );
    StreamingScoreboardRow {
        timestamp: format_utc_timestamp(current_unix_epoch_millis()),
        source_commit: worker_build_identity.source_commit,
        source_is_dirty: worker_build_identity.source_is_dirty,
        model_id: measurement.model_id.clone(),
        limit_bytes,
        prompt_token_count: measurement.total_input_tokens,
        reply_token_count: measurement.total_output_tokens,
        prompt_tokens_per_second: measurement.prefill_tokens_per_second,
        reply_tokens_per_second: measurement.decode_tokens_per_second,
        peak_memory_bytes: memory_reading.peak_memory_bytes,
        final_active_memory_bytes: memory_reading.active_memory_bytes,
        kept_expert_payload_bytes,
        unfilled_expert_entitlement_bytes: memory_reading.unfilled_expert_entitlement_bytes,
        // A dense model keeps layers, not experts; its kept and total layer
        // counts below say how full it is.
        expert_entitlement_share_filled: match dense_layers {
            Some(_) => None,
            None => entitlement_share_filled(
                kept_expert_payload_bytes,
                memory_reading.unfilled_expert_entitlement_bytes,
            ),
        },
        dense_kept_layer_count: dense_layers.map(|dense_layers| dense_layers.kept_layer_count),
        dense_layer_count: dense_layers.map(|dense_layers| dense_layers.decoder_layer_count),
        limit_overflow_count: limit_overflow_count_in_worker_logs(&run_outcome.logging_directory),
        measured_prompt_storage_read_bytes: storage_reads.map(|reads| reads.prompt_read_bytes),
        measured_reply_storage_read_bytes: storage_reads.map(|reads| reads.reply_read_bytes),
        unexplained_headroom_bytes: memory_reading.unexplained_headroom_bytes,
        owner_overrun_bytes: memory_reading.owner_overrun_bytes,
        memory_ledger: ledger_scoreboard_reading(final_health),
    }
}

fn ledger_scoreboard_reading(
    final_health: &WorkerHealthSnapshot,
) -> Option<LedgerScoreboardReading> {
    let ledger_snapshot = final_health
        .latest_mlx_memory_snapshot
        .as_ref()?
        .memory_ledger
        .as_ref()?;
    Some(LedgerScoreboardReading {
        observed_step_count: ledger_snapshot.observed_step_count,
        largest_unaccounted_bytes: ledger_snapshot.largest_unaccounted_bytes,
        largest_overcounted_bytes: ledger_snapshot.largest_overcounted_bytes,
        largest_kept_over_capacity_bytes: ledger_snapshot.largest_kept_over_capacity_bytes,
        largest_reading_step_peak_beyond_held_bytes: ledger_snapshot
            .largest_reading_step_peak_beyond_held_bytes,
        largest_writing_step_peak_beyond_held_bytes: ledger_snapshot
            .largest_writing_step_peak_beyond_held_bytes,
        probed_mlx_working_buffer_bytes: ledger_snapshot.probed_mlx_working_buffer_bytes,
        lowest_kept_share_while_reading_permille: ledger_snapshot
            .lowest_kept_share_while_reading_permille,
        mlx_working_buffer_bytes: ledger_snapshot.mlx_working_buffer_bytes,
        kept_weight_capacity_bytes: ledger_snapshot.kept_weight_capacity_bytes,
    })
}

/// The memory figures of the worker's last published MLX snapshot; all zero
/// when the worker published none.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct MemoryReading {
    peak_memory_bytes: u64,
    active_memory_bytes: u64,
    unfilled_expert_entitlement_bytes: u64,
    unexplained_headroom_bytes: u64,
    owner_overrun_bytes: u64,
}

fn memory_reading_at_request_end(final_health: &WorkerHealthSnapshot) -> MemoryReading {
    let Some(memory_snapshot) = final_health.latest_mlx_memory_snapshot.as_ref() else {
        return MemoryReading::default();
    };
    let mut memory_reading = MemoryReading {
        peak_memory_bytes: memory_snapshot.peak_memory_bytes,
        active_memory_bytes: memory_snapshot.active_memory_bytes,
        ..MemoryReading::default()
    };
    if let Some(utilization) = memory_snapshot.memory_ceiling_utilization.as_ref() {
        memory_reading.unfilled_expert_entitlement_bytes =
            utilization.unseated_expert_entitlement_bytes;
        memory_reading.unexplained_headroom_bytes = utilization.unexplained_headroom_bytes;
        memory_reading.owner_overrun_bytes = utilization.owner_overrun_bytes;
    }
    memory_reading
}

/// Expert bytes read from storage, split by the request stage that read
/// them: a session's running totals, or what one request added to them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ExpertStorageReads {
    prompt_read_bytes: u64,
    reply_read_bytes: u64,
}

/// What the measured request alone read: the session counters at the end
/// less the same counters after the warmup. `None` unless the worker
/// published expert streaming statistics at both instants: a dense model
/// publishes none, and without the earlier reading the warmup's bytes could
/// not be told apart from the measured request's.
fn measured_request_storage_reads(
    health_after_warmup: Option<&WorkerHealthSnapshot>,
    final_health: &WorkerHealthSnapshot,
) -> Option<ExpertStorageReads> {
    let session_reads_at_end = session_storage_reads(final_health)?;
    let session_reads_after_warmup = session_storage_reads(health_after_warmup?)?;
    Some(reads_since(
        session_reads_after_warmup,
        session_reads_at_end,
    ))
}

fn session_storage_reads(health_snapshot: &WorkerHealthSnapshot) -> Option<ExpertStorageReads> {
    let session_stats = health_snapshot
        .expert_streaming_observatory
        .latest_stats
        .as_ref()?;
    Some(ExpertStorageReads {
        prompt_read_bytes: session_stats.mandatory_prefill_source_bytes,
        reply_read_bytes: session_stats.mandatory_decode_source_bytes,
    })
}

/// Later session counters less earlier ones. Session counters only grow, so
/// a smaller later value means the two snapshots are from different engine
/// loads; saturating reports zero for that instead of a wrapped number.
const fn reads_since(
    earlier_session_reads: ExpertStorageReads,
    later_session_reads: ExpertStorageReads,
) -> ExpertStorageReads {
    ExpertStorageReads {
        prompt_read_bytes: later_session_reads
            .prompt_read_bytes
            .saturating_sub(earlier_session_reads.prompt_read_bytes),
        reply_read_bytes: later_session_reads
            .reply_read_bytes
            .saturating_sub(earlier_session_reads.reply_read_bytes),
    }
}

/// Kept expert bytes as a share of what the engine was entitled to keep.
fn entitlement_share_filled(
    kept_expert_payload_bytes: u64,
    unfilled_expert_entitlement_bytes: u64,
) -> Option<f64> {
    let entitlement_bytes =
        kept_expert_payload_bytes.saturating_add(unfilled_expert_entitlement_bytes);
    if entitlement_bytes == 0 {
        return None;
    }
    Some(kept_expert_payload_bytes as f64 / entitlement_bytes as f64)
}

/// Counts limit overflows across every worker log in the directory.
fn limit_overflow_count_in_worker_logs(logging_directory: &Path) -> Option<u64> {
    let directory_entries = fs::read_dir(logging_directory).ok()?;
    let mut worker_log_count = 0_u64;
    let mut limit_overflow_count = 0_u64;
    for directory_entry in directory_entries.flatten() {
        let file_name = directory_entry.file_name();
        let file_name = file_name.to_string_lossy();
        if !file_name.starts_with("worker") || !file_name.contains(".log") {
            continue;
        }
        let Ok(worker_log_text) = fs::read_to_string(directory_entry.path()) else {
            continue;
        };
        worker_log_count += 1;
        limit_overflow_count += limit_overflow_count_in(&worker_log_text);
    }
    (worker_log_count > 0).then_some(limit_overflow_count)
}

fn limit_overflow_count_in(worker_log_text: &str) -> u64 {
    let mut limit_overflow_count = 0_u64;
    for log_line in worker_log_text.lines() {
        if log_line.contains(LIMIT_OVERFLOW_LOG_MARKER) {
            limit_overflow_count += 1;
        }
    }
    limit_overflow_count
}

fn publish_scoreboard_row(scoreboard_row: &StreamingScoreboardRow) {
    let row_json = serde_json::to_string(scoreboard_row)
        .expect("a scoreboard row holds only numbers and strings, which always serialize");
    eprintln!("[streaming-scoreboard] {row_json}");
    let evidence_directory =
        support::acceptance_evidence::acceptance_evidence_root(SCOREBOARD_EVIDENCE_DIRECTORY_NAME);
    fs::create_dir_all(&evidence_directory)
        .expect("the scoreboard evidence directory should be created");
    let scoreboard_path = evidence_directory.join(SCOREBOARD_FILE_NAME);
    let mut scoreboard_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&scoreboard_path)
        .expect("the scoreboard file should open for appending");
    writeln!(scoreboard_file, "{row_json}").expect("the scoreboard row should be appended");
    eprintln!(
        "[streaming-scoreboard] appended to {}",
        scoreboard_path.display()
    );
}

#[cfg(test)]
#[path = "streaming_scoreboard_tests.rs"]
mod tests;
