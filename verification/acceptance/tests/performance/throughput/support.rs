//! Journey driver: launch, warm up, measure, and record one throughput journey.
//!
//! The driver launches the production inference-worker process built with the
//! same Cargo profile as this test binary (a release journey launches the
//! release worker or fails before launch) over the daemon IPC boundary, runs
//! one short warmup completion to spin up first-use
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
use std::time::{Duration, Instant};

use astronomical_daemon::{ChatGenerationExecutor, WorkerHealthSnapshot};
use ipc_protocol::{ChatImageInput, WorkerDecodeAstroExpertCachePolicy};

pub(crate) use crate::performance_throughput::completion::{
    drive_completion, dump_worker_logging_directory,
    read_decode_astro_expert_cache_diagnostic_evidence, read_measured_completion, wait_until_idle,
};

// The evidence lane composes the same measurement type, so it stays reachable
// through the journey driver rather than through its own module path.
pub(crate) use crate::performance_throughput::completion::{
    CompletionIndex, ThroughputMeasurement, read_completion_by_index,
};
use crate::performance_throughput::machine_specs::MachineSpecs;
use crate::performance_throughput::throughput_record::{
    MemoryCellBudgetExhaustedStage, ThroughputJourneyKind, ThroughputRecord,
    current_unix_epoch_millis, format_utc_timestamp, recorded_git_commit, throughput_record_json,
};
pub(crate) use crate::performance_throughput::worker_environment::{
    PromptChunking, diagnostics_enabled, perf_worker_environment,
};

pub(crate) const MAXIMUM_MODEL_MEMORY_BYTES: u64 = 36 * (1 << 30);
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
        "the measured completion must not reuse the astro SSD prompt cache",
    );
}

pub(crate) const JOURNEY_TIMEOUT: Duration = Duration::from_secs(115);
pub(crate) const MODEL_LOAD_TIMEOUT: Duration = Duration::from_secs(60);
pub(crate) const READY_ATTEMPT_LIMIT: u8 = 70;
const WORKER_LOG_DUMP_LINE_LIMIT: usize = 60;

/// The test case for one throughput journey: the journey family recorded in
/// its printed record, the internal decode `astro_expert_cache` policy arm, the short
/// warmup completion (its prompt, images, and output cap), the measured
/// completion (its prompt, images, and output cap), the sampling temperature,
/// the per-cell MLX ceiling override, whether the worker runs with performance
/// attribution, and the built-in journey deadline. The test that defines these
/// values lives beside this driver so the test case is readable where the test
/// is.
#[derive(Clone, Debug)]
pub(crate) struct ThroughputJourney {
    pub(crate) journey_kind: ThroughputJourneyKind,
    /// Internal experiment selector passed to the worker, never a user setting.
    pub(crate) decode_astro_expert_cache_policy: WorkerDecodeAstroExpertCachePolicy,
    pub(crate) warmup_input_prompt: String,
    pub(crate) warmup_images: Vec<ChatImageInput>,
    pub(crate) warmup_output_tokens: u16,
    pub(crate) measured_input_prompt: String,
    pub(crate) measured_images: Vec<ChatImageInput>,
    pub(crate) measured_output_tokens: u16,
    pub(crate) temperature_thousandths: u16,
    pub(crate) maximum_mlx_memory_bytes: Option<u64>,
    pub(crate) attribution_enabled: bool,
    /// Which prompt chunk size the worker serves with.
    pub(crate) prompt_chunking: PromptChunking,
    pub(crate) journey_timeout: Duration,
    /// The measured model's directory when the journey locates it itself
    /// (an environment-addressed pack); `None` resolves the model id through
    /// the Development configuration's model directories.
    pub(crate) model_directory: Option<PathBuf>,
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
            decode_astro_expert_cache_policy: WorkerDecodeAstroExpertCachePolicy::Baseline,
            warmup_input_prompt,
            warmup_images,
            warmup_output_tokens,
            measured_input_prompt,
            measured_images,
            measured_output_tokens,
            temperature_thousandths,
            maximum_mlx_memory_bytes: None,
            attribution_enabled: false,
            prompt_chunking: PromptChunking::MeasurementLane,
            journey_timeout: JOURNEY_TIMEOUT,
            model_directory: None,
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
    /// The worker's health between the warmup and the measured completion.
    /// Session counters only ever accumulate, so a lane that reports what the
    /// measured completion alone read from storage subtracts this snapshot's
    /// counters from the final one. `None` when no warmup ran or it did not
    /// finish.
    pub(crate) health_snapshot_after_warmup: Option<WorkerHealthSnapshot>,
    /// The last active-request progress observed by the background sampler,
    /// which survives cancellation: a cancelled generation clears the health
    /// snapshot's progress, so a cell that died mid-prefill still reports its
    /// partial prefill rate from here.
    pub(crate) last_observed_active_request_progress:
        Option<astronomical_daemon::ActiveRequestProgress>,
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
    let journey_started_at = Instant::now();
    let measured_throughput = loop {
        let journey_elapsed = journey_started_at.elapsed();
        let remaining_budget = journey_timeout.saturating_sub(journey_elapsed);
        if remaining_budget.is_zero() {
            panic!(
                "the throughput journey for {model_id} exceeded the {journey_timeout:?} deadline"
            );
        }
        let status_interval = remaining_budget.min(Duration::from_secs(5));
        match receiver.recv_timeout(status_interval) {
            Ok(measured_throughput) => break measured_throughput,
            Err(RecvTimeoutError::Timeout) => {
                let journey_elapsed = journey_started_at.elapsed();
                if journey_elapsed >= journey_timeout {
                    panic!(
                        "the throughput journey for {model_id} exceeded the {journey_timeout:?} deadline"
                    );
                }
                let journey_budget_millis = journey_timeout.as_millis().max(1);
                let time_budget_progress_percent = journey_elapsed
                    .as_millis()
                    .saturating_mul(100)
                    .checked_div(journey_budget_millis)
                    .unwrap_or(100)
                    .min(100);
                let remaining_budget = journey_timeout.saturating_sub(journey_elapsed);
                eprintln!(
                    "[performance-throughput] progress model={model_id} time_budget_progress_percent={time_budget_progress_percent} eta_seconds={} stage=bounded_journey",
                    remaining_budget.as_secs()
                );
            }
            Err(RecvTimeoutError::Disconnected) => {
                panic!("the throughput worker thread terminated before completing")
            }
        }
    };
    worker
        .join()
        .expect("the throughput worker thread should join cleanly");
    measured_throughput
}

/// Runs one full throughput journey and prints its record.
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
        decode_astro_expert_cache_policy: measurement.decode_astro_expert_cache_policy,
        prefill_tokens_per_second: (measurement.prefill_tokens_per_second.round()) as u32,
        decode_tokens_per_second: (measurement.decode_tokens_per_second.round()) as u32,
        git_commit: recorded_git_commit(),
        worker_build_identity: super::throughput_record::recorded_worker_build_identity(),
        machine: machine_specs,
        total_input_tokens: measurement.total_input_tokens,
        total_output_tokens: measurement.total_output_tokens,
        cached_tokens: measurement.cached_tokens,
        prefill_time_seconds: measurement.prefill_time_seconds,
        decode_time_seconds: measurement.decode_time_seconds,
        configured_maximum_mlx_memory_bytes: journey.maximum_mlx_memory_bytes,
        completion_ordinal_in_worker: measurement.completion_ordinal_in_worker,
        memory_cell: None,
    };
    eprintln!(
        "[performance-throughput] {}",
        serde_json::to_string_pretty(&throughput_record_json(&record))
            .unwrap_or_else(|_| "{ \"error\": \"record serialization failed\" }".to_owned())
    );
    measurement
}

/// Reads the measured completion's rates from the worker's performance log.
/// A completion that ran out of budget wrote no sample, so its rates are
/// reported as the shortfall; any other missing sample is a broken lane.
fn measured_completion_measurement(
    model_id: &str,
    journey: &ThroughputJourney,
    logging_directory: &std::path::Path,
    measured_completion_ordinal_in_worker: u32,
    measured_budget_exhausted: bool,
) -> ThroughputMeasurement {
    let mut measurement = match read_measured_completion(
        &logging_directory.join("performance.jsonl"),
        model_id,
        journey.decode_astro_expert_cache_policy,
        measured_completion_ordinal_in_worker,
    ) {
        Some(measurement) => measurement,
        None if measured_budget_exhausted => budget_exhausted_measurement(model_id, journey),
        None => panic!("the measured completion should produce a performance sample"),
    };
    if journey.attribution_enabled {
        measurement.decode_astro_expert_cache_diagnostic_evidence =
            read_decode_astro_expert_cache_diagnostic_evidence(
                logging_directory,
                MEASURED_REQUEST_ID,
            );
    }
    measurement
}

mod journey_stages;
mod progress_report;

use journey_stages::{
    JourneyCompletions, apply_journey_overrides, budget_exhausted_at_launch,
    budget_exhausted_measurement, drive_journey_completions, launch_servable_worker_within_budget,
    resolve_budget_exhausted_stage, resolve_journey_model_directory, spawn_progress_sampler,
    stop_progress_sampler_and_shut_down_worker,
};

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
    let model_directory = resolve_journey_model_directory(model_id, journey);
    let (
        _isolated_home,
        worker_executable_path,
        logging_directory,
        model_policy_catalog,
        mut worker_startup_configuration,
    ) = perf_worker_environment(
        model_id,
        &model_directory,
        journey.attribution_enabled,
        journey.prompt_chunking,
    );
    apply_journey_overrides(&mut worker_startup_configuration, journey);
    fs::create_dir_all(&logging_directory)
        .expect("the throughput performance log directory should be created");

    let Some(worker_handle) = launch_servable_worker_within_budget(
        cell_deadline,
        worker_executable_path,
        &logging_directory,
        model_policy_catalog,
        worker_startup_configuration,
        journey,
    )
    .await
    else {
        return budget_exhausted_at_launch(model_id, journey, logging_directory);
    };

    let progress_sampler = spawn_progress_sampler(&worker_handle);

    let journey_completions = drive_journey_completions(
        &worker_handle,
        model_id,
        journey,
        cell_deadline,
        &logging_directory,
    )
    .await;
    let JourneyCompletions {
        warmup_completed,
        measured_completed,
        measured_output_text,
        measured_completion_ordinal_in_worker,
        health_snapshot_after_warmup,
    } = journey_completions;
    let budget_exhausted_stage =
        resolve_budget_exhausted_stage(warmup_completed, measured_completed);
    let measured_budget_exhausted = budget_exhausted_stage.is_some();
    let final_health_snapshot = worker_handle.worker_health_snapshot();
    let last_observed_active_request_progress = stop_progress_sampler_and_shut_down_worker(
        worker_handle,
        &final_health_snapshot,
        progress_sampler,
    )
    .await;
    let mut measurement = measured_completion_measurement(
        model_id,
        journey,
        &logging_directory,
        measured_completion_ordinal_in_worker,
        measured_budget_exhausted,
    );
    measurement.measured_output_text = measured_output_text;
    JourneyRunOutcome {
        measurement,
        measured_budget_exhausted,
        budget_exhausted_stage,
        final_health_snapshot,
        health_snapshot_after_warmup,
        last_observed_active_request_progress,
        logging_directory,
    }
}

/// The Romeo and Juliet source token-bounded prompts are cut from, word by word.
const ROMEO_AND_JULIET_SOURCE: &str =
    include_str!("../../fixtures/model_metrics_10000_romeo_and_juliet_words.txt");

/// The model id discovery derives for a model directory: the leaf of the
/// decoded `models--org--repo` `huggingface_hub_cache` entry a snapshot lives
/// under, or, in a plain library (`<organization>/<model>`), the folder's
/// own name, as Astronomical's own library names its models.
pub(crate) fn discovered_model_id(pack_directory: &std::path::Path) -> String {
    huggingface_hub_cache_model_id(pack_directory)
        .unwrap_or_else(|| plain_library_model_id(pack_directory))
}

/// The repository leaf of the `huggingface_hub_cache` entry above a snapshot.
fn huggingface_hub_cache_model_id(pack_directory: &std::path::Path) -> Option<String> {
    let entry_name = pack_directory.ancestors().find_map(|ancestor| {
        ancestor
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| name.starts_with("models--"))
    })?;
    entry_name
        .trim_start_matches("models--")
        .rsplit("--")
        .next()
        .map(str::to_owned)
}

fn plain_library_model_id(pack_directory: &std::path::Path) -> String {
    pack_directory
        .file_name()
        .and_then(|name| name.to_str())
        .expect("a plain library model directory has a folder name")
        .to_owned()
}

/// Cuts the Romeo and Juliet source at the fewest words whose prompt encodes
/// to at least `minimum_token_count` tokens under the tokenizer in
/// `model_directory`, so a measured prompt clears its token floor by at most
/// a word.
pub(crate) fn romeo_and_juliet_prompt_of_at_least(
    model_directory: &std::path::Path,
    instruction: &str,
    minimum_token_count: usize,
) -> String {
    let tokenizer = tokenizers::Tokenizer::from_file(model_directory.join("tokenizer.json"))
        .expect("the pack tokenizer should load");
    let render = |word_count: usize| {
        let excerpt = ROMEO_AND_JULIET_SOURCE
            .split_whitespace()
            .take(word_count)
            .collect::<Vec<_>>()
            .join(" ");
        format!("{instruction}\n\n{excerpt}")
    };
    let encoded_length = |word_count: usize| {
        tokenizer
            .encode(render(word_count).as_str(), false)
            .expect("the prompt should encode")
            .get_ids()
            .len()
    };
    let total_word_count = ROMEO_AND_JULIET_SOURCE.split_whitespace().count();
    assert!(
        encoded_length(total_word_count) >= minimum_token_count,
        "the Romeo and Juliet source must encode to at least {minimum_token_count} tokens"
    );
    let (mut low, mut high) = (1_usize, total_word_count);
    while low < high {
        let middle = low + (high - low) / 2;
        if encoded_length(middle) >= minimum_token_count {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    render(low)
}
