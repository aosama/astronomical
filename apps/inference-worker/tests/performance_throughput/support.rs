//! Journey driver: launch, warm up, measure, and record one throughput journey.
//!
//! The driver launches the production inference-worker process over the
//! supervisor IPC boundary, runs one short warmup completion to spin up first-use
//! JIT kernels, then one measured completion over the real prompt, and reads the
//! worker's authoritative per-request throughput (`prefill_tok_per_second` and
//! `generation_tok_per_second`) from the generation performance log. The warmup
//! is discarded so JIT compilation never inflates the measured prefill; the
//! measured completion's values feed the reported record. The test case (warmup
//! prompt and output cap, measured prompt and output cap, temperature) is defined
//! by the test that calls this driver, so the test case is readable where the
//! test is.
//!
//! This module owns the journey and its budget. Where the worker is launched
//! lives in `worker_environment`; how one completion is driven and read back
//! lives in `completion`. A journey talks to this driver alone; the measurement
//! type is re-exported here so an evidence lane composes it without reaching
//! past the driver.

use std::fs;
use std::path::PathBuf;
use std::sync::mpsc::RecvTimeoutError;
use std::time::Duration;

use astronomical_ipc_protocol::{ChatImageInput, RequestId};
use astronomical_supervisor::{
    ChatGenerationExecutor, GenerationPerformanceLog, WorkerHandle, WorkerHealthSnapshot,
};
use tokio::time::sleep;

use crate::performance_throughput::completion::{
    drive_completion, dump_worker_logging_directory, read_measured_completion, wait_until_idle,
};

// The evidence lane composes the same measurement type, so it stays reachable
// through the journey driver rather than through its own module path.
pub(crate) use crate::performance_throughput::completion::ThroughputMeasurement;
use crate::performance_throughput::historical_record::{
    MemoryCellBudgetExhaustedStage, ThroughputJourneyKind, ThroughputRecord,
    append_throughput_history, current_unix_epoch_millis, format_utc_timestamp, history_log_path,
    recorded_git_commit, throughput_record_json,
};
use crate::performance_throughput::machine_specs::MachineSpecs;
use crate::performance_throughput::worker_environment::{
    diagnostics_enabled, perf_worker_environment,
};

pub(crate) const MAXIMUM_MODEL_MEMORY_BYTES: u64 = 36_000_000_000;
const MINIMUM_ACCEPTABLE_INPUT_TOKENS: u32 = 9_000;
const MAXIMUM_ACCEPTABLE_INPUT_TOKENS: u32 = 11_000;
const MINIMUM_ACCEPTABLE_OUTPUT_TOKENS: u16 = 450;
const MAXIMUM_ACCEPTABLE_OUTPUT_TOKENS: u16 = 550;
const WARMUP_SOURCE_WORD_COUNT: usize = 350;
pub(crate) const MEASURED_REQUEST_ID: u64 = 2;

pub(crate) fn romeo_and_juliet_warmup_prompt(
    warmup_instruction: &str,
    romeo_and_juliet_source: &str,
) -> String {
    let warmup_source = romeo_and_juliet_source
        .split_whitespace()
        .take(WARMUP_SOURCE_WORD_COUNT)
        .collect::<Vec<_>>()
        .join(" ");
    format!("{warmup_instruction}\n\n{warmup_source}")
}

pub(crate) fn assert_measurement_shape(measured_throughput: &ThroughputMeasurement) {
    assert!(
        (MINIMUM_ACCEPTABLE_INPUT_TOKENS..=MAXIMUM_ACCEPTABLE_INPUT_TOKENS)
            .contains(&measured_throughput.total_input_tokens),
        "the measured Romeo and Juliet prompt should contain 9,000 to 11,000 input tokens, got {}",
        measured_throughput.total_input_tokens,
    );
    assert!(
        (MINIMUM_ACCEPTABLE_OUTPUT_TOKENS..=MAXIMUM_ACCEPTABLE_OUTPUT_TOKENS)
            .contains(&measured_throughput.total_output_tokens),
        "the measured completion should generate 450 to 550 output tokens, got {}",
        measured_throughput.total_output_tokens,
    );
    assert_eq!(
        measured_throughput.cached_tokens, 0,
        "the measured completion must not reuse the persistent prompt cache",
    );
}

pub(crate) const JOURNEY_TIMEOUT: Duration = Duration::from_secs(115);
const MODEL_LOAD_TIMEOUT: Duration = Duration::from_secs(60);
const READY_ATTEMPT_LIMIT: u8 = 70;
const WORKER_LOG_DUMP_LINE_LIMIT: usize = 60;

/// The test case for one throughput journey: the journey family recorded in
/// the durable history line, the short warmup completion (its prompt, images,
/// and output cap), the measured completion (its prompt, images, and output
/// cap), the sampling temperature, the per-cell MLX ceiling override, whether
/// the worker runs with performance attribution, and the built-in journey
/// deadline. The test that defines these values lives beside this driver so
/// the test case is readable where the test is.
#[derive(Clone, Debug)]
pub(crate) struct ThroughputJourney {
    pub(crate) journey_kind: ThroughputJourneyKind,
    pub(crate) warmup_input_prompt: String,
    pub(crate) warmup_images: Vec<ChatImageInput>,
    pub(crate) warmup_output_tokens: u16,
    pub(crate) measured_input_prompt: String,
    pub(crate) measured_images: Vec<ChatImageInput>,
    pub(crate) measured_output_tokens: u16,
    pub(crate) temperature_thousandths: u16,
    pub(crate) maximum_mlx_memory_bytes: Option<u64>,
    pub(crate) attribution_enabled: bool,
    pub(crate) journey_timeout: Duration,
}

impl ThroughputJourney {
    /// The standard production-faithful journey shape: machine-default ceiling,
    /// attribution disabled, and the surface's conventional deadline.
    pub(crate) fn production_default(
        journey_kind: ThroughputJourneyKind,
        warmup_input_prompt: String,
        warmup_images: Vec<ChatImageInput>,
        warmup_output_tokens: u16,
        measured_input_prompt: String,
        measured_images: Vec<ChatImageInput>,
        measured_output_tokens: u16,
        temperature_thousandths: u16,
    ) -> Self {
        Self {
            journey_kind,
            warmup_input_prompt,
            warmup_images,
            warmup_output_tokens,
            measured_input_prompt,
            measured_images,
            measured_output_tokens,
            temperature_thousandths,
            maximum_mlx_memory_bytes: None,
            attribution_enabled: false,
            journey_timeout: JOURNEY_TIMEOUT,
        }
    }
}

/// One journey run's full outcome: the measured completion's rates plus the
/// worker health snapshot and logging directory the memory-cell evidence pass
/// reads residency, memory, and attribution data from. A budget-exhausted
/// completion produced no performance-log sample; its partial evidence lives
/// in the health snapshot's active-request progress and in the worker's
/// cancelled attribution report.
pub(crate) struct JourneyRunOutcome {
    pub(crate) measurement: ThroughputMeasurement,
    pub(crate) measured_budget_exhausted: bool,
    pub(crate) budget_exhausted_stage: Option<MemoryCellBudgetExhaustedStage>,
    pub(crate) final_health_snapshot: WorkerHealthSnapshot,
    /// The last active-request progress observed by the background sampler,
    /// which survives cancellation: a cancelled generation clears the health
    /// snapshot's progress, so a cell that died mid-prefill still reports its
    /// partial prefill rate from here.
    pub(crate) last_observed_active_request_progress:
        Option<astronomical_supervisor::ActiveRequestProgress>,
    pub(crate) logging_directory: PathBuf,
}

/// Runs one journey on a dedicated multi-thread runtime and enforces the
/// journey's built-in timeout so a wedged worker can never hang the test
/// process.
pub(crate) fn run_journey_with_timeout(
    model_id: &'static str,
    journey: ThroughputJourney,
) -> ThroughputMeasurement {
    let journey_timeout = journey.journey_timeout;
    let (sender, receiver) = std::sync::mpsc::channel::<ThroughputMeasurement>();
    let worker = std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("the throughput worker runtime should build");
        let measured_throughput = runtime.block_on(run_throughput_journey(model_id, &journey));
        sender.send(measured_throughput).ok();
    });
    let measured_throughput = match receiver.recv_timeout(journey_timeout) {
        Ok(measured_throughput) => measured_throughput,
        Err(RecvTimeoutError::Timeout) => panic!(
            "the throughput journey for {model_id} exceeded the {journey_timeout:?} deadline"
        ),
        Err(RecvTimeoutError::Disconnected) => {
            panic!("the throughput worker thread terminated before completing")
        }
    };
    worker
        .join()
        .expect("the throughput worker thread should join cleanly");
    measured_throughput
}

/// Runs one full throughput journey and persists a durable historical record.
pub(crate) async fn run_throughput_journey(
    model_id: &str,
    journey: &ThroughputJourney,
) -> ThroughputMeasurement {
    let machine_specs = MachineSpecs::capture().await;
    let outcome = measure_throughput(model_id, journey).await;
    let measurement = outcome.measurement;
    let record = ThroughputRecord {
        timestamp: format_utc_timestamp(current_unix_epoch_millis()),
        model_id: measurement.model_id.clone(),
        journey: journey.journey_kind,
        prefill_tokens_per_second: (measurement.prefill_tokens_per_second.round()) as u32,
        decode_tokens_per_second: (measurement.decode_tokens_per_second.round()) as u32,
        git_commit: recorded_git_commit(),
        machine: machine_specs,
        total_input_tokens: measurement.total_input_tokens,
        total_output_tokens: measurement.total_output_tokens,
        cached_tokens: measurement.cached_tokens,
        prefill_time_seconds: measurement.prefill_time_seconds,
        decode_time_seconds: measurement.decode_time_seconds,
        memory_cell: None,
    };
    eprintln!(
        "[performance-throughput] {}",
        serde_json::to_string_pretty(&throughput_record_json(&record))
            .unwrap_or_else(|_| "{ \"error\": \"record serialization failed\" }".to_owned())
    );
    if diagnostics_enabled() {
        eprintln!(
            "[performance-throughput] diagnostics run: durable history append skipped because attribution distorts throughput"
        );
        return measurement;
    }
    let history_path = history_log_path();
    match append_throughput_history(&record, &history_path) {
        Ok(()) => eprintln!(
            "[performance-throughput] appended durable record to {}",
            history_path.display()
        ),
        Err(error) => eprintln!(
            "[performance-throughput] durable history append failed (measurement still reported): {error}"
        ),
    }
    measurement
}

pub(crate) async fn measure_throughput(
    model_id: &str,
    journey: &ThroughputJourney,
) -> JourneyRunOutcome {
    // The journey budget bounds the WHOLE journey, not just the measured
    // completion: worker launch, idle readiness, warmup, and the measured
    // segment. `run_journey_with_timeout` enforces that same budget from
    // outside, because the worker thread it owns cannot be cancelled from
    // inside — a panic on the driver thread leaves the runtime running.
    let cell_deadline = tokio::time::Instant::now() + journey.journey_timeout;
    let remaining_budget = || cell_deadline.saturating_duration_since(tokio::time::Instant::now());
    let model_directory = crate::support::configured_installed_model_directory_by_id(model_id);
    let (
        _isolated_home,
        worker_executable_path,
        logging_directory,
        model_policy_catalog,
        mut worker_startup_configuration,
    ) = perf_worker_environment(model_id, &model_directory, journey.attribution_enabled);
    worker_startup_configuration.configured_maximum_mlx_memory_bytes =
        journey.maximum_mlx_memory_bytes;
    let performance_log_path = logging_directory.join("performance.jsonl");
    fs::create_dir_all(&logging_directory)
        .expect("the throughput performance log directory should be created");

    let worker_handle = match tokio::time::timeout(remaining_budget(), async {
        let worker_handle = WorkerHandle::launch_with_startup_configuration(
            worker_executable_path,
            MODEL_LOAD_TIMEOUT,
            GenerationPerformanceLog::open(&logging_directory)
                .expect("the throughput performance log should open"),
            model_policy_catalog,
            worker_startup_configuration,
        )
        .await
        .expect("the supervisor should launch the measured worker");
        wait_until_idle(&worker_handle, READY_ATTEMPT_LIMIT).await;
        worker_handle
    })
    .await
    {
        Ok(worker_handle) => worker_handle,
        Err(_) => {
            eprintln!(
                "[performance-throughput] budget_exhausted=true model={model_id} stage=launch-or-idle budget_seconds={} — the cell never became servable inside the budget",
                journey.journey_timeout.as_secs(),
            );
            return JourneyRunOutcome {
                measurement: ThroughputMeasurement {
                    model_id: model_id.to_owned(),
                    total_input_tokens: 0,
                    total_output_tokens: 0,
                    cached_tokens: 0,
                    prefill_time_seconds: 0.0,
                    decode_time_seconds: 0.0,
                    prefill_tokens_per_second: 0.0,
                    decode_tokens_per_second: 0.0,
                },
                measured_budget_exhausted: true,
                budget_exhausted_stage: Some(MemoryCellBudgetExhaustedStage::LaunchOrIdle),
                final_health_snapshot: WorkerHandle::unavailable().worker_health_snapshot(),
                last_observed_active_request_progress: None,
                logging_directory,
            };
        }
    };

    // The progress sampler preserves the LAST observed per-phase progress even
    // after cancellation clears the health snapshot, so a budget-exhausted
    // cell still reports the prefill or decode rate it was dying at.
    let last_observed_active_request_progress = std::sync::Arc::new(std::sync::Mutex::new(
        None::<astronomical_supervisor::ActiveRequestProgress>,
    ));
    let sampler_stop_signal = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let progress_sampler = tokio::spawn({
        let worker_handle = worker_handle.clone();
        let last_observed_active_request_progress =
            std::sync::Arc::clone(&last_observed_active_request_progress);
        let sampler_stop_signal = std::sync::Arc::clone(&sampler_stop_signal);
        async move {
            while !sampler_stop_signal.load(std::sync::atomic::Ordering::Relaxed) {
                sleep(Duration::from_secs(2)).await;
                if sampler_stop_signal.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
                if let Some(active_request_progress) = worker_handle
                    .worker_health_snapshot()
                    .active_request_progress
                {
                    *last_observed_active_request_progress
                        .lock()
                        .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner()) =
                        Some(active_request_progress);
                }
            }
        }
    });

    eprintln!("[performance-throughput] warmup model={model_id}");
    let warmup_completed = tokio::time::timeout(
        remaining_budget(),
        drive_completion(
            &worker_handle,
            RequestId::new(1),
            model_id,
            &journey.warmup_input_prompt,
            journey.warmup_images.clone(),
            journey.warmup_output_tokens,
            journey.temperature_thousandths,
        ),
    )
    .await
    .map(|warmup_result| match warmup_result {
        Ok(()) => true,
        Err(drive_error) => {
            dump_worker_logging_directory(&logging_directory, WORKER_LOG_DUMP_LINE_LIMIT);
            panic!("the throughput completion for {model_id} failed: {drive_error}");
        }
    })
    .unwrap_or_else(|_| {
        // Dropping the pending drive future drops its event receiver, which is
        // the client-disconnect cancellation signal; the worker releases its
        // generation permit and finalizes a cancelled attribution report.
        eprintln!(
            "[performance-throughput] budget_exhausted=true model={model_id} stage=warmup budget_seconds={} — a warmup that cannot finish inside the budget is itself the collapse signal",
            journey.journey_timeout.as_secs(),
        );
        false
    });
    let mut measured_completed = false;
    if warmup_completed {
        eprintln!("[performance-throughput] measured model={model_id}");
        measured_completed = tokio::time::timeout(
            remaining_budget(),
            drive_completion(
                &worker_handle,
                RequestId::new(MEASURED_REQUEST_ID),
                model_id,
                &journey.measured_input_prompt,
                journey.measured_images.clone(),
                journey.measured_output_tokens,
                journey.temperature_thousandths,
            ),
        )
        .await
        .map(|measured_result| match measured_result {
            Ok(()) => true,
            Err(drive_error) => {
                dump_worker_logging_directory(&logging_directory, WORKER_LOG_DUMP_LINE_LIMIT);
                panic!("the throughput completion for {model_id} failed: {drive_error}");
            }
        })
        .unwrap_or_else(|_| {
            eprintln!(
                "[performance-throughput] budget_exhausted=true model={model_id} stage=measured budget_seconds={} — dropping the stream receiver to cancel via client disconnect",
                journey.journey_timeout.as_secs(),
            );
            false
        });
    }
    let budget_exhausted_stage = if !warmup_completed {
        Some(MemoryCellBudgetExhaustedStage::Warmup)
    } else if !measured_completed {
        Some(MemoryCellBudgetExhaustedStage::Measured)
    } else {
        None
    };
    let measured_budget_exhausted = budget_exhausted_stage.is_some();
    let final_health_snapshot = worker_handle.worker_health_snapshot();
    // The sampler holds one final poll's worth of lag; stop it, then prefer the
    // live snapshot's progress and fall back to what the sampler preserved.
    sampler_stop_signal.store(true, std::sync::atomic::Ordering::Relaxed);
    progress_sampler.abort();
    let last_observed_active_request_progress = final_health_snapshot
        .active_request_progress
        .clone()
        .or_else(|| {
            last_observed_active_request_progress
                .lock()
                .unwrap_or_else(|poisoned_lock| poisoned_lock.into_inner())
                .take()
        });
    worker_handle
        .shutdown()
        .await
        .expect("the measured worker should terminate and be reaped");
    // A budget-exhausted completion wrote no performance-log sample, so its rates
    // are zero here. Callers that own an evidence lane (the memory-ceiling
    // sweep) recover partial rates from the preserved request progress and the
    // cancelled attribution report; this driver only reports the shortfall.
    let measurement = match read_measured_completion(&performance_log_path, model_id) {
        Some(measurement) => measurement,
        None if measured_budget_exhausted => ThroughputMeasurement {
            model_id: model_id.to_owned(),
            total_input_tokens: 0,
            total_output_tokens: 0,
            cached_tokens: 0,
            prefill_time_seconds: 0.0,
            decode_time_seconds: 0.0,
            prefill_tokens_per_second: 0.0,
            decode_tokens_per_second: 0.0,
        },
        None => panic!("the measured completion should produce a performance sample"),
    };
    JourneyRunOutcome {
        measurement,
        measured_budget_exhausted,
        budget_exhausted_stage,
        final_health_snapshot,
        last_observed_active_request_progress,
        logging_directory,
    }
}
