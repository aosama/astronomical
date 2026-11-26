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
    WorkerHealthSnapshot,
};
use ipc_protocol::{ChatImageInput, RequestId, WorkerStartupConfiguration};
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
/// the per-cell MLX ceiling and the decode `astro_expert_cache` experiment arm.
pub(super) fn apply_journey_overrides(
    worker_startup_configuration: &mut WorkerStartupConfiguration,
    journey: &ThroughputJourney,
) {
    worker_startup_configuration.configured_maximum_mlx_memory_bytes =
        journey.maximum_mlx_memory_bytes;
    worker_startup_configuration.decode_astro_expert_cache_policy =
        journey.decode_astro_expert_cache_policy;
}

/// A budget-exhausted completion wrote no performance-log sample, so its
/// rates are zero here. Callers that own an evidence lane (the memory-ceiling
/// sweep) recover partial rates from the preserved request progress and the
/// cancelled attribution report; this driver only reports the shortfall.
/// Pure: which completion of the worker's life the measured run is for this
/// journey shape — ordinal two in a standard journey (the in-worker warmup
/// runs first), ordinal one in a single-request lane with no in-worker
/// warmup. Extracted so both the live measurement and the budget-exhausted
/// fallback report the ordinal from one definition.
fn measured_completion_ordinal_for_journey(journey: &ThroughputJourney) -> u32 {
    if journey.warmup_output_tokens == 0 {
        1
    } else {
        2
    }
}

pub(super) fn budget_exhausted_measurement(
    model_id: &str,
    journey: &ThroughputJourney,
) -> ThroughputMeasurement {
    // The rate is zero here, but the ordinal still says which worker state
    // the shortfall happened in.
    ThroughputMeasurement {
        model_id: model_id.to_owned(),
        decode_astro_expert_cache_policy: journey.decode_astro_expert_cache_policy,
        total_input_tokens: 0,
        total_output_tokens: 0,
        cached_tokens: 0,
        prefill_time_seconds: 0.0,
        decode_time_seconds: 0.0,
        prefill_tokens_per_second: 0.0,
        decode_tokens_per_second: 0.0,
        mlx_peak_memory_bytes: None,
        measured_output_text: None,
        decode_astro_expert_cache_diagnostic_evidence: None,
        completion_ordinal_in_worker: measured_completion_ordinal_for_journey(journey),
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
        health_snapshot_after_warmup: None,
        last_observed_active_request_progress: None,
        logging_directory,
    }
}

/// The journey's measured model directory: the one the journey names, or the
/// one the Development configuration's model directories resolve for the id.
pub(super) fn resolve_journey_model_directory(
    model_id: &str,
    journey: &ThroughputJourney,
) -> PathBuf {
    match journey.model_directory.clone() {
        Some(journey_model_directory) => journey_model_directory,
        None => crate::support::configured_installed_model_directory_by_id(model_id),
    }
}

/// Launches the measured worker inside what is left of the journey budget;
/// `None` when the budget ran out before the worker became servable.
pub(super) async fn launch_servable_worker_within_budget(
    cell_deadline: tokio::time::Instant,
    worker_executable_path: PathBuf,
    logging_directory: &Path,
    model_policy_catalog: Arc<HashMap<String, RuntimeModelPolicy>>,
    worker_startup_configuration: WorkerStartupConfiguration,
    journey: &ThroughputJourney,
) -> Option<WorkerHandle> {
    tokio::time::timeout(
        remaining_journey_budget(cell_deadline),
        launch_servable_worker(
            worker_executable_path,
            logging_directory,
            model_policy_catalog,
            worker_startup_configuration,
            journey,
        ),
    )
    .await
    .ok()
}

/// The outcome a journey reports when its budget ran out before the worker
/// launched and went idle.
pub(super) fn budget_exhausted_at_launch(
    model_id: &str,
    journey: &ThroughputJourney,
    logging_directory: PathBuf,
) -> JourneyRunOutcome {
    eprintln!(
        "[performance-throughput] budget_exhausted=true model={model_id} stage=launch-or-idle budget_seconds={} — the cell never became servable inside the budget",
        journey.journey_timeout.as_secs(),
    );
    budget_exhausted_outcome(
        model_id,
        journey,
        logging_directory,
        MemoryCellBudgetExhaustedStage::LaunchOrIdle,
    )
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
                test_harness::sparse_decode_full_fit::MEMORY_CEILING_BYTES,
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

/// The three pieces of a running progress sampler that the driver needs again
/// when the journey ends: the shared last-observed slot, the stop signal, and
/// the sampler task.
pub(super) struct ProgressSampler {
    preserved_progress_slot: Arc<Mutex<Option<astronomical_daemon::ActiveRequestProgress>>>,
    sampler_stop_signal: Arc<AtomicBool>,
    sampler_task: tokio::task::JoinHandle<()>,
}

/// Spawns the background progress sampler. The sampler preserves the LAST
/// observed per-phase progress even after cancellation clears the health
/// snapshot, so a budget-exhausted cell still reports the prefill or decode
/// rate it was dying at.
pub(super) fn spawn_progress_sampler(worker_handle: &WorkerHandle) -> ProgressSampler {
    let last_observed_active_request_progress: Arc<
        Mutex<Option<astronomical_daemon::ActiveRequestProgress>>,
    > = Arc::new(Mutex::new(None));
    let sampler_stop_signal = Arc::new(AtomicBool::new(false));
    let sampler_worker_handle = worker_handle.clone();
    let sampler_progress_slot = Arc::clone(&last_observed_active_request_progress);
    let sampler_stop = Arc::clone(&sampler_stop_signal);
    let sampler_task = tokio::spawn(async move {
        while !sampler_stop.load(Ordering::Relaxed) {
            sleep(Duration::from_secs(2)).await;
            if sampler_stop.load(Ordering::Relaxed) {
                break;
            }
            let sampled_active_request_progress = sampler_worker_handle
                .worker_health_snapshot()
                .active_request_progress;
            if let Some(active_request_progress) = sampled_active_request_progress {
                super::progress_report::print_progress_line(&active_request_progress);
                let mut preserved_progress = match sampler_progress_slot.lock() {
                    Ok(guarded_slot) => guarded_slot,
                    Err(poisoned_lock) => poisoned_lock.into_inner(),
                };
                *preserved_progress = Some(active_request_progress);
            }
        }
    });
    ProgressSampler {
        preserved_progress_slot: last_observed_active_request_progress,
        sampler_stop_signal,
        sampler_task,
    }
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

/// The worker's health once a warmup has finished, for lanes that report a
/// session counter for the measured completion alone. A single-request lane
/// runs no warmup, and a warmup that ran out of budget is not a boundary to
/// measure from, so both yield `None`.
fn health_snapshot_after_completed_warmup(
    worker_handle: &WorkerHandle,
    journey: &ThroughputJourney,
    warmup_completed: bool,
) -> Option<WorkerHealthSnapshot> {
    if !warmup_completed || journey.warmup_output_tokens == 0 {
        return None;
    }
    Some(worker_handle.worker_health_snapshot())
}

/// The two completions' outcomes the driver's outcome assembly needs.
pub(super) struct JourneyCompletions {
    pub(super) warmup_completed: bool,
    pub(super) measured_completed: bool,
    pub(super) measured_output_text: Option<String>,
    /// Which completion of the worker's life the measured run was: ordinal
    /// two when the warmup ran first, ordinal one in a single-request lane
    /// with no in-worker warmup. The record carries it so cold and
    /// warmed worker rates are never compared as one series.
    pub(super) measured_completion_ordinal_in_worker: u32,
    /// The worker's health once the warmup finished and before the measured
    /// completion starts; `None` without a completed warmup.
    pub(super) health_snapshot_after_warmup: Option<WorkerHealthSnapshot>,
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
    //
    // The warmup's presence is also the measured completion's ordinal in the
    // worker's life: a standard journey measures ordinal two, a
    // single-request lane measures ordinal one. The record carries it so
    // first-touch and warmed rates stay distinguishable forever.
    let (warmup_completed, measured_completion_ordinal_in_worker) = if journey.warmup_output_tokens
        == 0
    {
        eprintln!(
            "[performance-throughput] single-request lane: skipping the in-worker warmup completion model={model_id}"
        );
        (true, measured_completion_ordinal_for_journey(journey))
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
            BudgetedCompletion::Completed(_) => (true, 2),
            BudgetedCompletion::BudgetExhausted => (false, 2),
        }
    };
    let mut measured_output_text = None;
    let mut measured_completed = false;
    let health_snapshot_after_warmup =
        health_snapshot_after_completed_warmup(worker_handle, journey, warmup_completed);
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
        measured_completion_ordinal_in_worker,
        health_snapshot_after_warmup,
    }
}

/// Stops the background progress sampler, shuts the measured worker down, and
/// returns the last active-request progress observed: the live snapshot's when
/// it has one, else what the sampler preserved. `progress_sampler` is what
/// `spawn_progress_sampler` returned.
pub(super) async fn stop_progress_sampler_and_shut_down_worker(
    worker_handle: WorkerHandle,
    final_health_snapshot: &WorkerHealthSnapshot,
    progress_sampler: ProgressSampler,
) -> Option<astronomical_daemon::ActiveRequestProgress> {
    let ProgressSampler {
        preserved_progress_slot,
        sampler_stop_signal,
        sampler_task,
    } = progress_sampler;
    sampler_stop_signal.store(true, Ordering::Relaxed);
    sampler_task.abort();
    let last_observed_active_request_progress =
        match final_health_snapshot.active_request_progress.clone() {
            Some(active_request_progress) => Some(active_request_progress),
            None => take_preserved_progress(&preserved_progress_slot),
        };
    worker_handle
        .shutdown()
        .await
        .expect("the measured worker should terminate and be reaped");
    last_observed_active_request_progress
}
