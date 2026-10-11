//! Memory-ceiling sweep: the evidence lane for the dynamic serving-envelope
//! hypothesis (issue #1120).
//!
//! Each ceiling cell runs ONE production-shaped journey under an explicitly
//! configured MLX ceiling — the same >=10,000-token Romeo and Juliet input and
//! 500-token output as the text lane, persistent prompt cache disabled —
//! with performance attribution ON, and appends one durable history record
//! whose `memory_cell` carries the residency and byte-traffic evidence.
//! Every cell uses a 2,048-token prefill chunk and remains at or below 35 GB.
//! Attribution stays on for every cell equally, so the ceiling CURVE is
//! internally consistent even though absolute rates carry attribution
//! overhead.
//!
//! Every cell has one hard 60-second test timeout. The serving phase is stopped
//! at 45 seconds so the worker can be cancelled, shut down, and its partial
//! evidence persisted before the test timeout. A slow cell fails with its
//! attribution report instead of continuing into a long external runner.
//!
//! The cells run one process each (MLX limits are process-global); the cargo
//! commands live in <repo-root>/cargo-commands-for-testing.md. The recommended
//! protocol brackets the sweep with a 32 GB reference cell and interleaves
//! ceilings so page-cache warmth cannot masquerade as a ceiling effect.

use std::fs;
use std::time::Duration;

use astronomical_ipc_protocol::ChatImageInput;
use serial_test::serial;

use crate::performance_throughput::historical_record::{
    MemoryCellRecord, ThroughputJourneyKind, ThroughputRecord, append_throughput_history,
    current_unix_epoch_millis, format_utc_timestamp, history_log_path, recorded_git_commit,
    throughput_record_json,
};
use crate::performance_throughput::machine_specs::MachineSpecs;
use crate::performance_throughput::support::{
    self as throughput_support, JourneyRunOutcome, ThroughputJourney,
};
use crate::support;

const WARMUP_INPUT_INSTRUCTION: &str = "Continue the supplied Romeo and Juliet story above.";
const WARMUP_ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../fixtures/model_metrics_warmup_romeo_and_juliet.txt");
const WARMUP_MAXIMUM_OUTPUT_TOKENS: u16 = 50;
const MEASURED_INPUT_INSTRUCTION: &str =
    "Continue the supplied Romeo and Juliet story above for approximately 500 tokens.";
const MEASURED_ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../fixtures/model_metrics_10000_tokens_romeo_and_juliet.txt");
const MEASURED_MAXIMUM_OUTPUT_TOKENS: u16 = 500;
const TEMPERATURE_THOUSANDTHS: u16 = 1_000;
/// Leaves 15 seconds inside the 60-second test timeout for cancellation,
/// worker shutdown, and durable partial-evidence persistence.
const MEMORY_CELL_SERVING_BUDGET: Duration = Duration::from_secs(45);
const MEMORY_CELL_TEST_TIMEOUT: Duration = Duration::from_secs(60);
const SWEEP_CELL_ORDER_ENVIRONMENT_VARIABLE: &str = "MEMORY_SWEEP_CELL_ORDER";

/// The serving budget must leave the time the evidence pass needs: it stops the
/// worker, reaps it, reads its attribution log, and appends the durable record.
/// A lane that spends that window inside the journey loses the measurement it
/// exists to keep, so the relationship is asserted rather than assumed.
#[test]
fn should_leave_fifteen_seconds_for_shutdown_and_evidence_within_the_sixty_second_deadline() {
    assert_eq!(
        MEMORY_CELL_TEST_TIMEOUT.saturating_sub(MEMORY_CELL_SERVING_BUDGET),
        Duration::from_secs(15)
    );
}

macro_rules! memory_ceiling_cell {
    ($test_function_name:ident, $ceiling_gb:literal) => {
        #[tokio::test(flavor = "multi_thread")]
        #[ignore = concat!(
            "loads the large sparse MoE and measures serving throughput under a ",
            stringify!($ceiling_gb),
            " GB MLX ceiling, appending a memory-cell history record within the 60-second test timeout"
        )]
        #[serial]
        async fn $test_function_name() {
            tokio::time::timeout(
                MEMORY_CELL_TEST_TIMEOUT,
                run_memory_ceiling_cell($ceiling_gb),
            )
            .await
            .expect("the memory-ceiling SSD journey must finish within 60 seconds");
        }
    };
}

memory_ceiling_cell!(
    should_serve_the_large_sparse_moe_under_a_23gb_ceiling_and_record_the_memory_cell,
    23
);
memory_ceiling_cell!(
    should_serve_the_large_sparse_moe_under_a_28gb_ceiling_and_record_the_memory_cell,
    28
);
memory_ceiling_cell!(
    should_serve_the_large_sparse_moe_under_a_32gb_ceiling_and_record_the_memory_cell,
    32
);
memory_ceiling_cell!(
    should_serve_the_large_sparse_moe_under_a_35gb_ceiling_and_record_the_memory_cell,
    35
);
async fn run_memory_ceiling_cell(ceiling_gb: u64) {
    let configured_maximum_mlx_memory_bytes = ceiling_gb * 1_000_000_000;
    append_memory_cell_record(
        throughput_support::measure_throughput(
            support::large_sparse_moe_model_id(),
            &cell_journey(configured_maximum_mlx_memory_bytes),
        )
        .await,
        configured_maximum_mlx_memory_bytes,
        ceiling_gb,
    )
    .await;
}

async fn append_memory_cell_record(
    run_outcome: JourneyRunOutcome,
    configured_maximum_mlx_memory_bytes: u64,
    ceiling_gb: u64,
) {
    let budget_exhausted = run_outcome.measured_budget_exhausted;
    let mut measurement = run_outcome.measurement.clone();
    if !budget_exhausted {
        throughput_support::assert_measurement_shape(&measurement);
    }
    // A completed cell MUST have written a generation attribution report, so a
    // missing one is a broken evidence lane rather than a slow cell. A
    // budget-exhausted cell is the opposite case: its cancellation may have
    // raced the report write, so absence degrades to an empty report and the
    // record keeps whatever counters were observed before the boundary.
    let attribution_report = if budget_exhausted {
        read_generation_attribution_report(
            &run_outcome.logging_directory,
            throughput_support::MEASURED_REQUEST_ID,
        )
        .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()))
    } else {
        generation_attribution_report(&run_outcome.logging_directory)
    };
    if budget_exhausted {
        let preserved_progress = run_outcome
            .final_health_snapshot
            .active_request_progress
            .clone()
            .or_else(|| run_outcome.last_observed_active_request_progress.clone());
        apply_partial_progress_rates(preserved_progress.as_ref(), &mut measurement);
        // A cell cancelled mid-decode completed its prefill; the cancelled
        // attribution report carries the authoritative prefill span and token
        // count, so the record reports the true prefill rate either way.
        if measurement.prefill_tokens_per_second == 0.0 {
            let prefill_token_count =
                attribution_counter(&attribution_report, "prompt_token_count");
            let prefill_elapsed_seconds = attribution_operation_total_seconds(
                &attribution_report,
                "prompt_prefill_advance_span",
            );
            if prefill_token_count > 0 && prefill_elapsed_seconds > 0.0 {
                measurement.total_input_tokens =
                    u32::try_from(prefill_token_count).unwrap_or(u32::MAX);
                measurement.prefill_time_seconds = prefill_elapsed_seconds;
                measurement.prefill_tokens_per_second =
                    prefill_token_count as f64 / prefill_elapsed_seconds;
            }
        }
    }
    let memory_cell = compose_memory_cell(
        &run_outcome,
        configured_maximum_mlx_memory_bytes,
        budget_exhausted,
        &attribution_report,
    );
    let machine_specs = MachineSpecs::capture().await;
    let record = ThroughputRecord {
        timestamp: format_utc_timestamp(current_unix_epoch_millis()),
        model_id: measurement.model_id.clone(),
        journey: ThroughputJourneyKind::MemoryCeilingSweep,
        prefill_tokens_per_second: (measurement.prefill_tokens_per_second.round()) as u32,
        decode_tokens_per_second: (measurement.decode_tokens_per_second.round()) as u32,
        git_commit: recorded_git_commit(),
        machine: machine_specs,
        total_input_tokens: measurement.total_input_tokens,
        total_output_tokens: measurement.total_output_tokens,
        cached_tokens: measurement.cached_tokens,
        prefill_time_seconds: measurement.prefill_time_seconds,
        decode_time_seconds: measurement.decode_time_seconds,
        memory_cell: Some(memory_cell),
    };
    eprintln!(
        "[memory-ceiling-sweep] cell={ceiling_gb}gb budget_exhausted={budget_exhausted} record={}",
        serde_json::to_string(&throughput_record_json(&record))
            .unwrap_or_else(|_| "record serialization failed".to_owned())
    );
    let history_path = history_log_path();
    match append_throughput_history(&record, &history_path) {
        Ok(()) => eprintln!(
            "[memory-ceiling-sweep] appended durable record to {}",
            history_path.display()
        ),
        Err(error) => panic!(
            "the {ceiling_gb} GB memory-cell record must persist to the durable history: {error}"
        ),
    }
    if budget_exhausted {
        // The durable append above is the deliverable of a budget-exhausted
        // cell: its partial evidence is the measurement the collapse produced.
        // Panicking afterwards keeps the cell red in the harness without
        // discarding that evidence, so the sweep can never be reported as a
        // pass merely because it recorded a collapse.
        panic!(
            "the {ceiling_gb} GB ceiling cell exhausted the {MEMORY_CELL_SERVING_BUDGET:?} serving budget; \
             partial evidence is recorded above and in the durable history — this cell is not servable \
             within the standard test budget"
        );
    }
}

/// Derives partial prefill and decode evidence from the preserved
/// active-request progress, so a budget-exhausted cell still reports the rate
/// it was dying at. A cell cancelled mid-prefill contributes the prefill
/// partial; one cancelled mid-generation contributes the decode partial.
fn apply_partial_progress_rates(
    preserved_progress: Option<&astronomical_supervisor::ActiveRequestProgress>,
    measurement: &mut throughput_support::ThroughputMeasurement,
) {
    match preserved_progress {
        Some(astronomical_supervisor::ActiveRequestProgress::Prefill {
            processed_tokens,
            elapsed_millis,
            ..
        }) => {
            measurement.prefill_time_seconds = *elapsed_millis as f64 / 1_000.0;
            measurement.prefill_tokens_per_second = if *elapsed_millis == 0 {
                0.0
            } else {
                *processed_tokens as f64 * 1_000.0 / *elapsed_millis as f64
            };
        }
        Some(astronomical_supervisor::ActiveRequestProgress::Generation {
            generated_token_count,
            elapsed_millis,
            ..
        }) => {
            measurement.total_output_tokens =
                u16::try_from(*generated_token_count).unwrap_or(u16::MAX);
            measurement.decode_time_seconds = *elapsed_millis as f64 / 1_000.0;
            measurement.decode_tokens_per_second = if *elapsed_millis == 0 {
                0.0
            } else {
                *generated_token_count as f64 * 1_000.0 / *elapsed_millis as f64
            };
        }
        _ => {}
    }
}

fn cell_journey(configured_maximum_mlx_memory_bytes: u64) -> ThroughputJourney {
    ThroughputJourney {
        journey_kind: ThroughputJourneyKind::MemoryCeilingSweep,
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
        maximum_mlx_memory_bytes: Some(configured_maximum_mlx_memory_bytes),
        attribution_enabled: true,
        journey_timeout: MEMORY_CELL_SERVING_BUDGET,
    }
}

fn compose_memory_cell(
    run_outcome: &JourneyRunOutcome,
    configured_maximum_mlx_memory_bytes: u64,
    budget_exhausted: bool,
    attribution_report: &serde_json::Value,
) -> MemoryCellRecord {
    let health_snapshot = &run_outcome.final_health_snapshot;
    let expert_residency = health_snapshot.expert_residency.as_ref();
    let mlx_memory_snapshot = health_snapshot.latest_mlx_memory_snapshot.as_ref();
    MemoryCellRecord {
        configured_maximum_mlx_memory_bytes,
        machine_recommended_working_set_bytes: health_snapshot.machine_mlx_memory_ceiling_bytes,
        expert_residency_mode: health_snapshot
            .expert_memory_mode
            // The published mode is a Debug-formatted enum lowered to the
            // history log's vocabulary. An unpublished mode is recorded as
            // such rather than as "resident": a cell that never published
            // residency is missing evidence, and defaulting it to a real mode
            // would make an unmeasured cell look measured.
            .map(|expert_memory_mode| format!("{expert_memory_mode:?}").to_lowercase())
            .unwrap_or_else(|| "unpublished".to_owned()),
        expert_residency_total_layer_count: expert_residency
            .map(|residency| residency.total_layer_count)
            .unwrap_or(0),
        expert_residency_resident_expert_count: expert_residency
            .map(|residency| residency.resident_expert_count)
            .unwrap_or(0),
        resident_expert_payload_bytes: expert_residency
            .map(|residency| residency.resident_expert_payload_bytes)
            .unwrap_or(0),
        peak_mlx_memory_bytes: mlx_memory_snapshot
            .map(|snapshot| snapshot.peak_memory_bytes)
            .unwrap_or(0),
        final_active_mlx_memory_bytes: mlx_memory_snapshot
            .map(|snapshot| snapshot.active_memory_bytes)
            .unwrap_or(0),
        positional_read_byte_count: attribution_counter(
            &attribution_report,
            "positional_file_read_byte_count",
        ),
        process_physical_disk_read_bytes: attribution_report
            .get("process_physical_disk_read_bytes")
            .and_then(serde_json::Value::as_u64),
        process_physical_disk_written_bytes: attribution_report
            .get("process_physical_disk_written_bytes")
            .and_then(serde_json::Value::as_u64),
        prefill_chunk_count: attribution_counter(&attribution_report, "prefill_chunk_count"),
        maximum_decode_advance_seconds: attribution_operation(
            &attribution_report,
            "decode_advance_span",
        )
        .map(|operation_maximum_elapsed_nanoseconds| {
            operation_maximum_elapsed_nanoseconds as f64 / 1_000_000_000.0
        })
        .unwrap_or(0.0),
        sweep_cell_order: std::env::var(SWEEP_CELL_ORDER_ENVIRONMENT_VARIABLE)
            .ok()
            .and_then(|cell_order| cell_order.parse().ok())
            .unwrap_or(0),
        model_loading_process_disk_read_bytes: model_loading_attribution_report(
            &run_outcome.logging_directory,
        )
        .and_then(|loading_report| {
            loading_report
                .get("process_physical_disk_read_bytes")
                .and_then(serde_json::Value::as_u64)
        }),
        budget_exhausted,
    }
}

fn model_loading_attribution_report(
    logging_directory: &std::path::Path,
) -> Option<serde_json::Value> {
    let attribution_log =
        fs::read_to_string(logging_directory.join("performance-attribution.jsonl")).ok()?;
    attribution_log
        .lines()
        .filter_map(|json_line| serde_json::from_str::<serde_json::Value>(json_line).ok())
        .find(|attribution_report| attribution_report["report_kind"] == "model_loading")
}

fn read_generation_attribution_report(
    logging_directory: &std::path::Path,
    request_id: u64,
) -> Option<serde_json::Value> {
    let attribution_log_path = logging_directory.join("performance-attribution.jsonl");
    let attribution_log = fs::read_to_string(&attribution_log_path).ok()?;
    attribution_log
        .lines()
        .filter_map(|json_line| serde_json::from_str::<serde_json::Value>(json_line).ok())
        .find(|attribution_report| {
            attribution_report["report_kind"] == "generation"
                && attribution_report["request_id"].as_u64() == Some(request_id)
        })
}

fn generation_attribution_report(logging_directory: &std::path::Path) -> serde_json::Value {
    read_generation_attribution_report(
        logging_directory,
        throughput_support::MEASURED_REQUEST_ID,
    )
    .unwrap_or_else(|| {
        panic!(
            "the evidence pass attribution log at {} should contain a generation report for measured request {}",
            logging_directory
                .join("performance-attribution.jsonl")
                .display(),
            throughput_support::MEASURED_REQUEST_ID
        )
    })
}

fn attribution_counter(attribution_report: &serde_json::Value, counter_name: &str) -> u64 {
    attribution_report["counters"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|counter_report| counter_report["counter"] == counter_name)
        .filter_map(|counter_report| counter_report["amount"].as_u64())
        .sum()
}

fn attribution_operation(
    attribution_report: &serde_json::Value,
    operation_name: &str,
) -> Option<u64> {
    attribution_report["operations"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|operation_report| operation_report["operation"] == operation_name)
        .filter_map(|operation_report| operation_report["maximum_elapsed_nanoseconds"].as_u64())
        .max()
}

fn attribution_operation_total_seconds(
    attribution_report: &serde_json::Value,
    operation_name: &str,
) -> f64 {
    attribution_report["operations"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|operation_report| operation_report["operation"] == operation_name)
        .filter_map(|operation_report| operation_report["total_elapsed_nanoseconds"].as_u64())
        .sum::<u64>() as f64
        / 1_000_000_000.0
}

#[cfg(test)]
#[path = "memory_ceiling_sweep_tests.rs"]
mod tests;
