//! `cargo xtask measure-speed [--base-url URL] [--temperature T]
//! [--reference-set] [--write-ledger] [MODEL_ID...]` — measures how fast
//! models reply on this Mac.
//!
//! Each case loads its model with a warm-up request (about 500 tokens in, 50
//! out), then measures four tasks (an essay and code, after a short prompt
//! and after a long one), 200 tokens out: prompt reading, the first token
//! and the reply, one reply a task (`measure_each_task_once` says why there
//! are no repeats). Essays run at temperature 1.0 and code at 0.7 unless
//! --temperature sets one for all.
//!
//! `--reference-set` measures the cases of
//! `registry/release_speed/reference_set.json`, setting each case's memory
//! limit and restoring the engine's own afterwards. `--write-ledger` writes
//! them into the release speed ledger of the engine's version, the one
//! durable record of release speed. The catalog does not copy those figures.
//!
//! It runs against Development, never Stable, and refuses to start unless
//! that engine has the `astro_ssd_prompt_cache` and the skill declaration
//! switched off. AGENTS.md allows the `astro_ssd_prompt_cache` on for decode
//! measurements, but every task here also reports prompt reading, which a
//! restored prompt would skip.

#[path = "measure_speed/engine_requests.rs"]
mod engine_requests;
#[path = "measure_speed/engine_setup.rs"]
mod engine_setup;
#[path = "measure_speed/measure_speed_options.rs"]
pub(crate) mod measure_speed_options;
#[path = "measure_speed/measured_case.rs"]
mod measured_case;
#[path = "measure_speed/measurement_cases.rs"]
mod measurement_cases;
#[path = "measure_speed/reference_set.rs"]
mod reference_set;
#[path = "measure_speed/reply_drafting.rs"]
mod reply_drafting;
#[path = "measure_speed/served_conditions.rs"]
mod served_conditions;
#[path = "measure_speed/speed_catalog.rs"]
pub(crate) mod speed_catalog;
#[path = "measure_speed/speed_progress.rs"]
mod speed_progress;
#[path = "measure_speed/speed_prompts.rs"]
pub(crate) mod speed_prompts;
#[path = "measure_speed/speed_rounds.rs"]
mod speed_rounds;
#[path = "measure_speed/stream_timing.rs"]
mod stream_timing;

use std::path::Path;

use engine_setup::{
    EngineBuild, engine_build_from_status, refuse_dirty_build_for_ledger,
    refuse_unless_set_up_for_measuring,
};
use measure_speed_options::{MeasureSpeedOptions, this_mac};
use measured_case::MeasuredCase;
use measurement_cases::{
    MeasurementCase, MemoryLimitRequest, measurement_cases, refuse_unless_served_as_requested,
};
use speed_catalog::catalog_identity;
use speed_progress::SpeedProgress;
use speed_prompts::{SpeedPrompts, SpeedTask};
use speed_rounds::ModelSpeedRounds;

use crate::release_speed::speed_ledger::{DrafterInUse, ReleaseSpeedLedger, write_ledger};

const STATUS_PREFIX: &str = "[measure-speed]";
const WARMUP_OUTPUT_TOKENS: u32 = 50;
/// Enough tokens for a steady reply rate; more only lengthens the wait (a
/// dense 27B writes 500 in over half a minute).
const MEASURED_OUTPUT_TOKENS: u32 = 200;
/// A reply that stopped on its own before this many tokens is too short to
/// time fairly and is left out of the figures.
const MINIMUM_TIMED_TOKENS: u64 = 100;
const CATALOG_PATH: &str = "registry/download_catalog.json";

pub fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    let measure_speed_options = MeasureSpeedOptions::parse(arguments)?;
    let engine_build = check_engine_is_set_up(&measure_speed_options.base_url)?;
    refuse_dirty_build_for_ledger(measure_speed_options.should_write_ledger, &engine_build)?;
    let measurement_cases = measurement_cases(repository_root, &measure_speed_options)?;
    let speed_prompts = SpeedPrompts::load(repository_root)?;
    let measured_cases = measure_cases(&measure_speed_options, &speed_prompts, &measurement_cases);
    restore_memory_limit(
        &measure_speed_options.base_url,
        &measurement_cases,
        &engine_build,
    );
    if measured_cases.is_empty() {
        return Err(format!(
            "{STATUS_PREFIX} status=failed reason=nothing-measured"
        ));
    }
    write_results(
        repository_root,
        &measure_speed_options,
        &engine_build,
        &measured_cases,
    )
}

/// Measures every case in turn; a case that fails (its model is missing, or
/// the Mac refuses its memory limit) is reported and the rest still run.
fn measure_cases(
    measure_speed_options: &MeasureSpeedOptions,
    speed_prompts: &SpeedPrompts,
    measurement_cases: &[MeasurementCase],
) -> Vec<MeasuredCase> {
    let speed_progress = SpeedProgress::start(measurement_cases.len() * STEPS_A_CASE);
    let mut measured_cases = Vec::new();
    for measurement_case in measurement_cases {
        match measure_one_case(
            measure_speed_options,
            speed_prompts,
            measurement_case,
            &speed_progress,
        ) {
            Ok(measured_case) => measured_cases.push(measured_case),
            Err(case_error) => println!(
                "{STATUS_PREFIX} not-measured model={} memory_limit={:?} reason={case_error}",
                measurement_case.model_id, measurement_case.memory_limit_request
            ),
        }
    }
    speed_progress.finish();
    measured_cases
}

/// The warm-up, then one reply a task.
const STEPS_A_CASE: usize = 1 + SpeedTask::ALL.len();

/// The engine's build, once its status shows it is set up for measuring.
fn check_engine_is_set_up(base_url: &str) -> Result<EngineBuild, String> {
    let status_document = engine_requests::fetch_status(base_url)?;
    refuse_unless_set_up_for_measuring(&status_document)?;
    engine_build_from_status(&status_document)
}

fn measure_one_case(
    measure_speed_options: &MeasureSpeedOptions,
    speed_prompts: &SpeedPrompts,
    measurement_case: &MeasurementCase,
    speed_progress: &SpeedProgress,
) -> Result<MeasuredCase, String> {
    let base_url = &measure_speed_options.base_url;
    let model_id = &measurement_case.model_id;
    speed_progress.set_stage(&format!("{model_id} warm-up, which also loads it"));
    let warmup_outcome =
        load_under_requested_limit(measure_speed_options, speed_prompts, measurement_case)
            .and_then(|()| engine_requests::fetch_status(base_url))
            .and_then(|status_document| {
                served_conditions::served_conditions(&status_document, model_id)
            })
            .and_then(|served_conditions| {
                refuse_unless_served_as_requested(
                    measurement_case.memory_limit_request,
                    served_conditions.maximum_mlx_memory_gb,
                )
                .map(|()| served_conditions)
            });
    speed_progress.complete_step();
    let served_conditions = match warmup_outcome {
        Ok(served_conditions) => served_conditions,
        Err(warmup_error) => {
            speed_progress.skip_steps(STEPS_A_CASE - 1);
            return Err(warmup_error);
        }
    };
    let mut measured_case = MeasuredCase {
        model_id: model_id.clone(),
        served_conditions,
        speed_rounds: ModelSpeedRounds::default(),
        temperature_override: measure_speed_options.temperature_override,
        reply_tokens_requested: MEASURED_OUTPUT_TOKENS,
        long_prompt_tokens: measurement_case.long_prompt_tokens,
    };
    measure_each_task_once(
        measure_speed_options,
        speed_prompts,
        &mut measured_case,
        speed_progress,
    )?;
    println!(
        "{STATUS_PREFIX} medians model={model_id} memory_limit_gb={:?} {:?}",
        measured_case.served_conditions.maximum_mlx_memory_gb,
        measured_case.speed_rounds.medians()
    );
    Ok(measured_case)
}

/// Measures the four tasks once. A model that drafts first writes one
/// unmeasured reply, because a drafting model gives up its first reply's
/// rounds to time its plain tokens and that reply would otherwise be the one
/// measured.
///
/// One reply a task, never more. Running every task three times made the
/// release reference set take about ten minutes, and it was run again and
/// again during development when one model had changed: the person waiting
/// paid for every repeat, and none of them answered a question one reply did
/// not. When a single figure is in doubt, measure that one case again, not
/// the set.
fn measure_each_task_once(
    measure_speed_options: &MeasureSpeedOptions,
    speed_prompts: &SpeedPrompts,
    measured_case: &mut MeasuredCase,
    speed_progress: &SpeedProgress,
) -> Result<(), String> {
    write_the_learning_reply(measure_speed_options, speed_prompts, measured_case)?;
    for (task_index, speed_task) in SpeedTask::ALL.into_iter().enumerate() {
        if let Err(reply_error) = measure_one_reply(
            measure_speed_options,
            speed_prompts,
            measured_case,
            speed_task,
            speed_progress,
        ) {
            speed_progress.skip_steps(SpeedTask::ALL.len() - task_index - 1);
            return Err(reply_error);
        }
    }
    Ok(())
}

/// One unmeasured reply for a model that drafts; nothing for one that does
/// not.
fn write_the_learning_reply(
    measure_speed_options: &MeasureSpeedOptions,
    speed_prompts: &SpeedPrompts,
    measured_case: &MeasuredCase,
) -> Result<(), String> {
    if measured_case.served_conditions.drafter == DrafterInUse::None {
        return Ok(());
    }
    engine_requests::stream_chat_reply(
        &measure_speed_options.base_url,
        &measured_case.model_id,
        &speed_prompts.task_prompt(
            SpeedTask::ShortPromptEssay,
            0,
            measured_case.long_prompt_tokens,
        ),
        MEASURED_OUTPUT_TOKENS,
        measured_case.task_temperature(SpeedTask::ShortPromptEssay),
    )
    .map(|_| ())
}

/// Sets the case's memory limit when it asks for one, then loads the model
/// with the warm-up reply.
fn load_under_requested_limit(
    measure_speed_options: &MeasureSpeedOptions,
    speed_prompts: &SpeedPrompts,
    measurement_case: &MeasurementCase,
) -> Result<(), String> {
    if let MemoryLimitRequest::ServeUnder(maximum_mlx_memory_gb) =
        measurement_case.memory_limit_request
    {
        engine_requests::serve_next_model_under(
            &measure_speed_options.base_url,
            maximum_mlx_memory_gb,
        )?;
    }
    engine_requests::stream_chat_reply(
        &measure_speed_options.base_url,
        &measurement_case.model_id,
        &speed_prompts.warmup_prompt(),
        WARMUP_OUTPUT_TOKENS,
        measure_speed_options
            .temperature_override
            .unwrap_or_else(|| SpeedTask::ShortPromptEssay.default_temperature()),
    )
    .map(|_| ())
}

fn measure_one_reply(
    measure_speed_options: &MeasureSpeedOptions,
    speed_prompts: &SpeedPrompts,
    measured_case: &mut MeasuredCase,
    speed_task: SpeedTask,
    speed_progress: &SpeedProgress,
) -> Result<(), String> {
    let model_id = &measured_case.model_id;
    speed_progress.set_stage(&format!("{model_id} {}", speed_task.catalog_field()));
    let mut reply_measurement = engine_requests::stream_chat_reply(
        &measure_speed_options.base_url,
        model_id,
        &speed_prompts.task_prompt(speed_task, 0, measured_case.long_prompt_tokens),
        MEASURED_OUTPUT_TOKENS,
        measured_case.task_temperature(speed_task),
    )?;
    // Status still holds this reply's peak: the engine publishes it before
    // it ends the reply's stream, and the next request has not started.
    reply_measurement.peak_mlx_memory_bytes =
        engine_requests::fetch_status(&measure_speed_options.base_url)
            .ok()
            .as_ref()
            .and_then(served_conditions::last_reply_peak_mlx_memory_bytes);
    println!(
        "{STATUS_PREFIX} reply model={model_id} task={} {}",
        speed_task.catalog_field(),
        reply_measurement.figures_text(),
    );
    if reply_measurement.completion_tokens >= MINIMUM_TIMED_TOKENS
        && reply_measurement.reply_tokens_per_second.is_some()
    {
        measured_case
            .speed_rounds
            .record(speed_task, reply_measurement);
    }
    speed_progress.complete_step();
    Ok(())
}

/// Puts the engine's own memory limit back after reference cases changed it,
/// with nothing loaded, as the person left it.
fn restore_memory_limit(
    base_url: &str,
    measurement_cases: &[MeasurementCase],
    engine_build: &EngineBuild,
) {
    if measurement_cases.iter().all(|measurement_case| {
        measurement_case.memory_limit_request == MemoryLimitRequest::KeepEngineLimit
    }) {
        return;
    }
    if let Err(restore_error) =
        engine_requests::serve_next_model_under(base_url, engine_build.maximum_mlx_memory_gb)
    {
        println!(
            "{STATUS_PREFIX} status=warning reason=memory-limit-not-restored detail={restore_error}: set it back to {:?} GB on Observatory's Settings page",
            engine_build.maximum_mlx_memory_gb
        );
    }
}

fn write_results(
    repository_root: &Path,
    measure_speed_options: &MeasureSpeedOptions,
    engine_build: &EngineBuild,
    measured_cases: &[MeasuredCase],
) -> Result<(), String> {
    if !measure_speed_options.should_write_ledger {
        return Ok(());
    }
    let catalog_path = repository_root.join(CATALOG_PATH);
    let catalog_text = std::fs::read_to_string(&catalog_path)
        .map_err(|read_error| format!("could not read {}: {read_error}", catalog_path.display()))?;
    let ledger_file = write_ledger(
        repository_root,
        release_ledger(engine_build, measured_cases, &catalog_text),
    )?;
    println!("{STATUS_PREFIX} wrote ledger={}", ledger_file.display());
    Ok(())
}

fn release_ledger(
    engine_build: &EngineBuild,
    measured_cases: &[MeasuredCase],
    catalog_text: &str,
) -> ReleaseSpeedLedger {
    let mut ledger_measurements = Vec::new();
    for measured_case in measured_cases {
        let model_revision = catalog_identity(catalog_text, &measured_case.model_id)
            .and_then(|catalog_identity| catalog_identity.revision);
        ledger_measurements.extend(measured_case.ledger_measurements(model_revision.as_deref()));
    }
    ReleaseSpeedLedger {
        astronomical_version: engine_build.version.clone(),
        source_commit: engine_build.commit.clone(),
        measured_on_date: todays_date(),
        machine: this_mac(),
        measurements: ledger_measurements,
    }
}

/// Today's date in UTC, as `2026-11-26`.
fn todays_date() -> String {
    std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%d"])
        .output()
        .ok()
        .filter(|date_output| date_output.status.success())
        .map_or_else(
            || "unknown".to_owned(),
            |date_output| {
                String::from_utf8_lossy(&date_output.stdout)
                    .trim()
                    .to_owned()
            },
        )
}
