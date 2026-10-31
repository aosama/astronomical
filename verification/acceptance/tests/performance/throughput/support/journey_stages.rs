//! The stages one throughput journey is composed of, extracted from the
//! journey driver so each stays a small named unit: worker launch and
//! readiness, the background progress sampler, one budgeted completion, and
//! the budget-exhaustion bookkeeping. The driver (in `support`) owns the
//! protocol — the order the stages run in and the outcome it reports.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use astronomical_daemon::{
    ChatGenerationExecutor, GenerationPerformanceLog, RuntimeModelPolicy, WorkerHandle,
};
use astronomical_ipc_protocol::{ChatImageInput, RequestId, WorkerStartupConfiguration};
use tokio::time::sleep;

use super::{
    JourneyRunOutcome, MAXIMUM_MODEL_MEMORY_BYTES, MEASURED_REQUEST_ID, MODEL_LOAD_TIMEOUT,
    MemoryCellBudgetExhaustedStage, READY_ATTEMPT_LIMIT, ThroughputJourney, ThroughputJourneyKind,
    ThroughputMeasurement, WORKER_LOG_DUMP_LINE_LIMIT, drive_completion,
    dump_worker_logging_directory, wait_until_idle,
};

/// The whole-journey deadline expressed as the remaining budget, as a plain
/// function so every stage budgets against the same instant.
pub(super) fn remaining_journey_budget(cell_deadline: tokio::time::Instant) -> Duration {
    cell_deadline.saturating_duration_since(tokio::time::Instant::now())
}

/// Applies the journey's worker-level overrides to the launch configuration:
/// the per-cell MLX ceiling, the decode-cache experiment arm, and the two
/// diagnostic Qwen3.5 engine knobs.
pub(super) fn apply_journey_overrides(
    worker_startup_configuration: &mut WorkerStartupConfiguration,
    journey: &ThroughputJourney,
) {
    worker_startup_configuration.configured_maximum_mlx_memory_bytes =
        journey.maximum_mlx_memory_bytes;
    worker_startup_configuration.decode_expert_cache_policy = journey.decode_expert_cache_policy;
    worker_startup_configuration.diagnostic_qwen3_5_engine_override =
        journey.diagnostic_qwen3_5_engine_override;
    worker_startup_configuration.diagnostic_qwen3_5_resident_prefill_activation_floor_bytes =
        journey.diagnostic_qwen3_5_resident_prefill_activation_floor_bytes;
}

/// A budget-exhausted completion wrote no performance-log sample, so its
/// rates are zero here. Callers that own an evidence lane (the memory-ceiling
/// sweep) recover partial rates from the preserved request progress and the
/// cancelled attribution report; this driver only reports the shortfall.
pub(super) fn budget_exhausted_measurement(
    model_id: &str,
    journey: &ThroughputJourney,
) -> ThroughputMeasurement {
    ThroughputMeasurement {
        model_id: model_id.to_owned(),
        decode_expert_cache_policy: journey.decode_expert_cache_policy,
        total_input_tokens: 0,
        total_output_tokens: 0,
        cached_tokens: 0,
        prefill_time_seconds: 0.0,
        decode_time_seconds: 0.0,
        prefill_tokens_per_second: 0.0,
        decode_tokens_per_second: 0.0,
        mlx_peak_memory_bytes: None,
        measured_output_text: None,
        decode_cache_diagnostic_evidence: None,
    }
}

/// The outcome a journey reports when its budget ran out before the worker
/// became servable.
pub(super) fn budget_exhausted_outcome(
    model_id: &str,
    journey: &ThroughputJourney,
    logging_directory: PathBuf,
    exhausted_stage: MemoryCellBudgetExhaustedStage,
) -> JourneyRunOutcome {
    JourneyRunOutcome {
        measurement: budget_exhausted_measurement(model_id, journey),
        measured_budget_exhausted: true,
        budget_exhausted_stage: Some(exhausted_stage),
        final_health_snapshot: WorkerHandle::unavailable().worker_health_snapshot(),
        last_observed_active_request_progress: None,
        logging_directory,
    }
}

/// Launches the measured worker, waits for idle readiness, and proves the
/// effective memory ceiling the journey demanded before the model loads: the
/// full-fit diagnostic must run at its dedicated ceiling, and every real-model
/// journey must be capped at or below the effective limit.
pub(super) async fn launch_servable_worker(
    worker_executable_path: PathBuf,
    logging_directory: &Path,
    model_policy_catalog: Arc<HashMap<String, RuntimeModelPolicy>>,
    worker_startup_configuration: WorkerStartupConfiguration,
    journey: &ThroughputJourney,
) -> WorkerHandle {
    let worker_handle = WorkerHandle::launch_with_startup_configuration(
        worker_executable_path,
        MODEL_LOAD_TIMEOUT,
        GenerationPerformanceLog::open(logging_directory)
            .expect("the throughput performance log should open"),
        model_policy_catalog,
        worker_startup_configuration,
    )
    .await
    .expect("the daemon should launch the measured worker");
    wait_until_idle(&worker_handle, READY_ATTEMPT_LIMIT).await;
    if let Some(configured_maximum_mlx_memory_bytes) = journey.maximum_mlx_memory_bytes {
        let effective_mlx_memory_ceiling_bytes = worker_handle
            .worker_health_snapshot()
            .mlx_memory_ceiling_bytes;
        if matches!(journey.journey_kind, ThroughputJourneyKind::DecodeFullFit) {
            assert_eq!(
                effective_mlx_memory_ceiling_bytes,
                astronomical_test_harness::sparse_decode_full_fit::MEMORY_CEILING_BYTES,
                "the full-fit diagnostic must exercise 37 GB, not a clamped streaming cell",
            );
        } else {
            assert!(
                effective_mlx_memory_ceiling_bytes <= MAXIMUM_MODEL_MEMORY_BYTES,
                "the configured {configured_maximum_mlx_memory_bytes}-byte real-model journey must be capped at or below the {MAXIMUM_MODEL_MEMORY_BYTES}-byte effective limit before model loading, got {effective_mlx_memory_ceiling_bytes}",
            );
        }
        if matches!(
            journey.journey_kind,
            ThroughputJourneyKind::MemoryCeilingSweep
        ) {
            eprintln!(
                "[memory-ceiling-sweep] preflight configured_bytes={configured_maximum_mlx_memory_bytes} effective_bytes={effective_mlx_memory_ceiling_bytes} maximum_effective_bytes={MAXIMUM_MODEL_MEMORY_BYTES} status=passed"
            );
        }
    }
    worker_handle
}

/// Spawns the background progress sampler. The sampler preserves the LAST
/// observed per-phase progress even after cancellation clears the health
/// snapshot, so a budget-exhausted cell still reports the prefill or decode
/// rate it was dying at. Returns the shared last-observed slot, the stop
/// signal, and the sampler task.
pub(super) fn spawn_progress_sampler(
    worker_handle: &WorkerHandle,
) -> (
    Arc<Mutex<Option<astronomical_daemon::ActiveRequestProgress>>>,
    Arc<AtomicBool>,
    tokio::task::JoinHandle<()>,
) {
    let last_observed_active_request_progress: Arc<
        Mutex<Option<astronomical_daemon::ActiveRequestProgress>>,
    > = Arc::new(Mutex::new(None));
    let sampler_stop_signal = Arc::new(AtomicBool::new(false));
    let sampler_worker_handle = worker_handle.clone();
    let sampler_progress_slot = Arc::clone(&last_observed_active_request_progress);
    let sampler_stop = Arc::clone(&sampler_stop_signal);
    let progress_sampler = tokio::spawn(async move {
        while !sampler_stop.load(Ordering::Relaxed) {
            sleep(Duration::from_secs(2)).await;
            if sampler_stop.load(Ordering::Relaxed) {
                break;
            }
            let sampled_active_request_progress = sampler_worker_handle
                .worker_health_snapshot()
                .active_request_progress;
            if let Some(active_request_progress) = sampled_active_request_progress {
                let mut preserved_progress = match sampler_progress_slot.lock() {
                    Ok(guarded_slot) => guarded_slot,
                    Err(poisoned_lock) => poisoned_lock.into_inner(),
                };
                *preserved_progress = Some(active_request_progress);
            }
        }
    });
    (
        last_observed_active_request_progress,
        sampler_stop_signal,
        progress_sampler,
    )
}

/// One completion (warmup or measured) described for the budgeted driver:
/// which request it is, what to feed it, and how the budget-exhaustion report
/// names the stage.
pub(super) struct BudgetedCompletionRequest<'a> {
    pub(super) request_id: RequestId,
    pub(super) input_prompt: &'a str,
    pub(super) input_images: Vec<ChatImageInput>,
    pub(super) output_tokens: u16,
    pub(super) stage_label: &'a str,
    pub(super) exhaustion_note: &'a str,
}

/// The result of one budgeted completion: finished (carrying its generated
/// text) or abandoned because the journey budget ran out.
pub(super) enum BudgetedCompletion {
    Completed(Option<String>),
    BudgetExhausted,
}

/// Drives one completion inside the remaining journey budget. A completion
/// error is a hard failure (the worker logs are dumped first). An exhausted
/// budget reports the collapse signal: dropping the pending drive future
/// drops its event receiver, which is the client-disconnect cancellation
/// signal, so the worker releases its generation permit and finalizes a
/// cancelled attribution report.
pub(super) async fn drive_completion_within_budget(
    worker_handle: &WorkerHandle,
    model_id: &str,
    journey: &ThroughputJourney,
    cell_deadline: tokio::time::Instant,
    completion_request: BudgetedCompletionRequest<'_>,
    logging_directory: &Path,
) -> BudgetedCompletion {
    match tokio::time::timeout(
        remaining_journey_budget(cell_deadline),
        drive_completion(
            worker_handle,
            completion_request.request_id,
            model_id,
            completion_request.input_prompt,
            completion_request.input_images,
            completion_request.output_tokens,
            journey.temperature_thousandths,
        ),
    )
    .await
    {
        Ok(completion_result) => match completion_result {
            Ok(generated_text) => BudgetedCompletion::Completed(Some(generated_text)),
            Err(drive_error) => {
                dump_worker_logging_directory(logging_directory, WORKER_LOG_DUMP_LINE_LIMIT);
                panic!("the throughput completion for {model_id} failed: {drive_error}");
            }
        },
        Err(_) => {
            eprintln!(
                "[performance-throughput] budget_exhausted=true model={model_id} stage={} budget_seconds={} — {}",
                completion_request.stage_label,
                journey.journey_timeout.as_secs(),
                completion_request.exhaustion_note,
            );
            BudgetedCompletion::BudgetExhausted
        }
    }
}

/// Which stage the budget ran out in, from the two completion flags.
pub(super) fn resolve_budget_exhausted_stage(
    warmup_completed: bool,
    measured_completed: bool,
) -> Option<MemoryCellBudgetExhaustedStage> {
    if !warmup_completed {
        Some(MemoryCellBudgetExhaustedStage::Warmup)
    } else if !measured_completed {
        Some(MemoryCellBudgetExhaustedStage::Measured)
    } else {
        None
    }
}

/// Takes the last progress the sampler preserved. The sampler holds one final
/// poll's worth of lag; the caller prefers the live snapshot's progress and
/// falls back to this.
pub(super) fn take_preserved_progress(
    preserved_progress_slot: &Mutex<Option<astronomical_daemon::ActiveRequestProgress>>,
) -> Option<astronomical_daemon::ActiveRequestProgress> {
    match preserved_progress_slot.lock() {
        Ok(mut guarded_slot) => guarded_slot.take(),
        Err(poisoned_lock) => poisoned_lock.into_inner().take(),
    }
}

/// The two completions' outcomes the driver's outcome assembly needs.
pub(super) struct JourneyCompletions {
    pub(super) warmup_completed: bool,
    pub(super) measured_completed: bool,
    pub(super) measured_output_text: Option<String>,
}

/// Runs the warmup (unless the journey is a single-request lane) and then the
/// measured completion, each inside the remaining journey budget.
pub(super) async fn drive_journey_completions(
    worker_handle: &WorkerHandle,
    model_id: &str,
    journey: &ThroughputJourney,
    cell_deadline: tokio::time::Instant,
    logging_directory: &Path,
) -> JourneyCompletions {
    // A zero-output warmup marks a single-request lane: each journey launches
    // a fresh worker and measures exactly one completion. The streaming hybrid
    // engine cannot serve two sequential requests under one ceiling once the
    // first request leaves its warm expert tables resident — the second
    // request's prefill streams the remaining layers beside them — so the
    // in-worker warmup is skipped and first-use JIT cost lands on both arms
    // identically through the alternating paired order.
    let warmup_completed = if journey.warmup_output_tokens == 0 {
        eprintln!(
            "[performance-throughput] single-request lane: skipping the in-worker warmup completion model={model_id}"
        );
        true
    } else {
        eprintln!("[performance-throughput] warmup model={model_id}");
        let warmup_request = BudgetedCompletionRequest {
            request_id: RequestId::new(1),
            input_prompt: &journey.warmup_input_prompt,
            input_images: journey.warmup_images.clone(),
            output_tokens: journey.warmup_output_tokens,
            stage_label: "warmup",
            exhaustion_note: "a warmup that cannot finish inside the budget is itself the collapse signal",
        };
        match drive_completion_within_budget(
            &worker_handle,
            model_id,
            journey,
            cell_deadline,
            warmup_request,
            &logging_directory,
        )
        .await
        {
            BudgetedCompletion::Completed(_) => true,
            BudgetedCompletion::BudgetExhausted => false,
        }
    };
    let mut measured_output_text = None;
    let mut measured_completed = false;
    if warmup_completed {
        eprintln!("[performance-throughput] measured model={model_id}");
        let measured_request = BudgetedCompletionRequest {
            request_id: RequestId::new(MEASURED_REQUEST_ID),
            input_prompt: &journey.measured_input_prompt,
            input_images: journey.measured_images.clone(),
            output_tokens: journey.measured_output_tokens,
            stage_label: "measured",
            exhaustion_note: "dropping the stream receiver to cancel via client disconnect",
        };
        match drive_completion_within_budget(
            &worker_handle,
            model_id,
            journey,
            cell_deadline,
            measured_request,
            &logging_directory,
        )
        .await
        {
            BudgetedCompletion::Completed(generated_text) => {
                measured_output_text = generated_text;
                measured_completed = true;
            }
            BudgetedCompletion::BudgetExhausted => {}
        }
    }

    JourneyCompletions {
        warmup_completed,
        measured_completed,
        measured_output_text,
    }
}
