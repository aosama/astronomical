//! One streaming worker driven through the same long prompt twice, so a journey
//! can compare what the first turn read from storage with what the second did.

use std::fs;
use std::path::Path;

use astronomical_daemon::{ChatGenerationExecutor, GenerationPerformanceLog, WorkerHandle};
use ipc_protocol::{RequestId, WorkerDecodeAstroExpertCachePolicy};
use tokio::time::timeout;

use super::{MEASURED_MAXIMUM_OUTPUT_TOKENS, TEMPERATURE_THOUSANDTHS, multi_chunk_measured_prompt};
use crate::performance_throughput::support as throughput_support;
use crate::support;

/// The evidence both turns of a two-turn journey contribute.
pub(super) struct TwoTurnEvidence {
    pub(super) first: throughput_support::ThroughputMeasurement,
    pub(super) second: throughput_support::ThroughputMeasurement,
    pub(super) first_positional_read_bytes: Option<u64>,
    pub(super) second_positional_read_bytes: Option<u64>,
}

/// Drives one worker through the same long prompt twice. The first turn
/// streams and (per the residency plan) seats complete layers; the second
/// turn must reuse that seating instead of re-paying the SSD streams.
pub(super) async fn run_two_turn_streaming_journey(
    attribution_enabled: bool,
    astro_ssd_prompt_cache_enabled: bool,
) -> TwoTurnEvidence {
    let model_id = support::large_sparse_moe_streaming_model_id();
    let model_directory = crate::support::configured_installed_model_directory_by_id(model_id);
    let long_prompt = multi_chunk_measured_prompt();
    let (
        _isolated_home,
        worker_executable_path,
        logging_directory,
        model_policy_catalog,
        mut worker_startup_configuration,
    ) = throughput_support::perf_worker_environment(
        model_id,
        &model_directory,
        attribution_enabled,
        throughput_support::PromptChunking::MeasurementLane,
    );
    worker_startup_configuration.configured_maximum_mlx_memory_bytes =
        Some(throughput_support::MAXIMUM_MODEL_MEMORY_BYTES);
    worker_startup_configuration.astro_ssd_prompt_cache_enabled = astro_ssd_prompt_cache_enabled;
    let performance_log_path = logging_directory.join("performance.jsonl");
    fs::create_dir_all(&logging_directory).expect("the worker logging directory should be created");

    let worker_handle = WorkerHandle::launch_with_startup_configuration(
        worker_executable_path,
        throughput_support::MODEL_LOAD_TIMEOUT,
        GenerationPerformanceLog::open(&logging_directory)
            .expect("the two-turn performance log should open"),
        model_policy_catalog,
        worker_startup_configuration,
    )
    .await
    .expect("the two-turn worker should launch");
    throughput_support::wait_until_idle(&worker_handle, throughput_support::READY_ATTEMPT_LIMIT)
        .await;

    drive_both_long_turns(&worker_handle, model_id, &long_prompt).await;

    let final_health_snapshot = worker_handle.worker_health_snapshot();
    let resident_expert_payload_bytes = final_health_snapshot
        .expert_residency
        .as_ref()
        .map(|residency| u64::from(residency.resident_expert_payload_bytes))
        .unwrap_or(0);
    eprintln!(
        "[multi-chunk-utilization] resident_expert_payload_bytes_after_second_turn={resident_expert_payload_bytes}"
    );
    worker_handle
        .shutdown()
        .await
        .expect("the two-turn worker should shut down");

    read_two_turn_evidence(&performance_log_path, &logging_directory, model_id)
}

async fn drive_both_long_turns(worker_handle: &WorkerHandle, model_id: &str, long_prompt: &str) {
    let _first_completion_text = timeout(
        throughput_support::JOURNEY_TIMEOUT,
        throughput_support::drive_completion(
            worker_handle,
            RequestId::new(1),
            model_id,
            long_prompt,
            Vec::new(),
            MEASURED_MAXIMUM_OUTPUT_TOKENS,
            TEMPERATURE_THOUSANDTHS,
        ),
    )
    .await
    .expect("the first long turn should finish inside the journey budget")
    .expect("the first long turn should complete");
    let _second_completion_text = timeout(
        throughput_support::JOURNEY_TIMEOUT,
        throughput_support::drive_completion(
            worker_handle,
            RequestId::new(throughput_support::MEASURED_REQUEST_ID),
            model_id,
            long_prompt,
            Vec::new(),
            MEASURED_MAXIMUM_OUTPUT_TOKENS,
            TEMPERATURE_THOUSANDTHS,
        ),
    )
    .await
    .expect("the second long turn should finish inside the journey budget")
    .expect("the second long turn should complete");
}

fn read_two_turn_evidence(
    performance_log_path: &Path,
    logging_directory: &Path,
    model_id: &str,
) -> TwoTurnEvidence {
    let first = throughput_support::read_completion_by_index(
        performance_log_path,
        model_id,
        WorkerDecodeAstroExpertCachePolicy::Baseline,
        throughput_support::CompletionIndex::First,
        1,
    )
    .expect("the first turn should write a performance sample");
    let second = throughput_support::read_completion_by_index(
        performance_log_path,
        model_id,
        WorkerDecodeAstroExpertCachePolicy::Baseline,
        throughput_support::CompletionIndex::Last,
        2,
    )
    .expect("the second turn should write a performance sample");

    let first_positional_read_bytes =
        crate::performance_throughput::completion::read_decode_astro_expert_cache_diagnostic_evidence(
            logging_directory,
            1,
        )
        .map(|evidence| evidence.positional_file_read_byte_count);
    let second_positional_read_bytes =
        crate::performance_throughput::completion::read_decode_astro_expert_cache_diagnostic_evidence(
            logging_directory,
            throughput_support::MEASURED_REQUEST_ID,
        )
        .map(|evidence| evidence.positional_file_read_byte_count);

    TwoTurnEvidence {
        first,
        second,
        first_positional_read_bytes,
        second_positional_read_bytes,
    }
}
