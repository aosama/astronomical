//! `cargo xtask measure-speed [--base-url URL] [--rounds N] [--measured-on TEXT]
//! [--write-catalog] MODEL_ID...` — measures how fast catalog models reply on
//! this Mac and, with `--write-catalog`, writes the figures into
//! `registry/download_catalog.json`.
//!
//! Each model is loaded once by a warm-up request (about 500 tokens in, 50
//! out), then measured on four tasks (an essay and code, after a short prompt
//! and after about 10,000 tokens), 500 tokens out at temperature 0.6, in
//! interleaved rounds. The median of the rounds is what the catalog keeps, so
//! one slow round never sets a figure. It runs against Development, never
//! Stable, and refuses to start unless that engine has the
//! `astro_ssd_prompt_cache` and the skill declaration switched off, as
//! AGENTS.md requires of throughput measurements.

#[path = "measure_speed/engine_requests.rs"]
mod engine_requests;
#[path = "measure_speed/measure_speed_options.rs"]
mod measure_speed_options;
#[path = "measure_speed/speed_catalog.rs"]
pub(crate) mod speed_catalog;
#[path = "measure_speed/speed_progress.rs"]
mod speed_progress;
#[path = "measure_speed/speed_prompts.rs"]
pub(crate) mod speed_prompts;
#[path = "measure_speed/stream_timing.rs"]
mod stream_timing;

use std::path::Path;

use measure_speed_options::{MeasureSpeedOptions, this_mac_description};
use speed_catalog::{MeasurementPlace, ModelSpeedRounds, write_measured_speed};
use speed_progress::SpeedProgress;
use speed_prompts::{SpeedPrompts, SpeedTask};

const STATUS_PREFIX: &str = "[measure-speed]";
const WARMUP_OUTPUT_TOKENS: u32 = 50;
const MEASURED_OUTPUT_TOKENS: u32 = 500;
/// A reply that stopped on its own before this many tokens is too short to
/// time fairly and is left out of the medians.
const MINIMUM_TIMED_TOKENS: u64 = 200;
const CATALOG_PATH: &str = "registry/download_catalog.json";

pub fn run(repository_root: &Path, arguments: &[String]) -> Result<(), String> {
    let measure_speed_options = MeasureSpeedOptions::parse(arguments)?;
    let measured_on = measure_speed_options
        .measured_on
        .clone()
        .unwrap_or_else(this_mac_description);
    let engine_version = check_engine_is_set_up(&measure_speed_options.base_url)?;
    let speed_prompts = SpeedPrompts::load(repository_root)?;
    let speed_progress = SpeedProgress::start(
        measure_speed_options.model_ids.len()
            * (1 + measure_speed_options.round_count * SpeedTask::ALL.len()),
    );
    let mut measured_models = Vec::new();
    for model_id in &measure_speed_options.model_ids {
        let speed_rounds = measure_one_model(
            &measure_speed_options,
            &speed_prompts,
            model_id,
            &speed_progress,
        )?;
        measured_models.push((model_id.clone(), speed_rounds));
    }
    speed_progress.finish();
    if measure_speed_options.should_write_catalog {
        let measurement_place = MeasurementPlace {
            measured_on: &measured_on,
            astronomical_version: &engine_version,
        };
        write_catalog(repository_root, &measured_models, &measurement_place)?;
    }
    Ok(())
}

/// The engine must be set up the way AGENTS.md measures throughput; it is
/// never reconfigured from here, because that is the person's own instance.
/// Returns the Astronomical version the figures will be recorded against.
fn check_engine_is_set_up(base_url: &str) -> Result<String, String> {
    let status_document = engine_requests::fetch_status(base_url)?;
    let prompt_cache_enabled =
        status_document["configuration"]["astro_ssd_prompt_cache"]["enabled"]["effective"]
            .as_bool()
            .unwrap_or(true);
    let skill_declared = status_document["skill"]["declared_to_models"]
        .as_bool()
        .unwrap_or(true);
    if prompt_cache_enabled || skill_declared {
        return Err(format!(
            "{STATUS_PREFIX} status=refused reason=engine-not-set-up: set \"ssd_prompt_cache\": {{\"enabled\": false}} and \"skill\": {{\"declare_to_models\": false}} in the engine's config.json, restart it, and run again (prompt cache on: {prompt_cache_enabled}, skill declared: {skill_declared})"
        ));
    }
    status_document["application"]["version"]
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("{STATUS_PREFIX} status=refused reason=engine-version-unknown"))
}

fn measure_one_model(
    measure_speed_options: &MeasureSpeedOptions,
    speed_prompts: &SpeedPrompts,
    model_id: &str,
    speed_progress: &SpeedProgress,
) -> Result<ModelSpeedRounds, String> {
    let base_url = &measure_speed_options.base_url;
    speed_progress.set_stage(&format!("{model_id} warm-up, which also loads it"));
    engine_requests::stream_chat_reply(
        base_url,
        model_id,
        &speed_prompts.warmup_prompt(),
        WARMUP_OUTPUT_TOKENS,
        measure_speed_options.temperature,
    )?;
    speed_progress.complete_step();
    let mut speed_rounds = ModelSpeedRounds::default();
    for round_index in 0..measure_speed_options.round_count {
        for speed_task in SpeedTask::ALL {
            speed_progress.set_stage(&format!(
                "{model_id} {} round {}",
                speed_task.catalog_field(),
                round_index + 1
            ));
            let reply_measurement = engine_requests::stream_chat_reply(
                base_url,
                model_id,
                &speed_prompts.task_prompt(speed_task, round_index),
                MEASURED_OUTPUT_TOKENS,
                measure_speed_options.temperature,
            )?;
            println!(
                "{STATUS_PREFIX} reply model={model_id} task={} round={} prompt_tokens={} reply_tokens={} seconds_to_first_token={:.2} reply_tokens_per_second={}",
                speed_task.catalog_field(),
                round_index + 1,
                reply_measurement.prompt_tokens,
                reply_measurement.completion_tokens,
                reply_measurement.seconds_to_first_token,
                reply_measurement
                    .reply_tokens_per_second
                    .map_or_else(|| "none".to_owned(), |rate| format!("{rate:.2}")),
            );
            if let Some(rate) = reply_measurement
                .reply_tokens_per_second
                .filter(|_| reply_measurement.completion_tokens >= MINIMUM_TIMED_TOKENS)
            {
                speed_rounds.record(speed_task, rate);
            }
            speed_progress.complete_step();
        }
    }
    println!(
        "{STATUS_PREFIX} medians model={model_id} {:?}",
        speed_rounds.medians()
    );
    Ok(speed_rounds)
}

/// Writes every fully measured model's medians into the catalog; a model the
/// catalog does not list, or one missing rounds, is reported and skipped.
fn write_catalog(
    repository_root: &Path,
    measured_models: &[(String, ModelSpeedRounds)],
    measurement_place: &MeasurementPlace<'_>,
) -> Result<(), String> {
    let catalog_path = repository_root.join(CATALOG_PATH);
    let mut catalog_text = std::fs::read_to_string(&catalog_path)
        .map_err(|read_error| format!("could not read {}: {read_error}", catalog_path.display()))?;
    for (model_id, speed_rounds) in measured_models {
        let Some(task_medians) = speed_rounds.medians() else {
            println!("{STATUS_PREFIX} skipped model={model_id} reason=missing-rounds");
            continue;
        };
        let Some(huggingface_id) = catalog_huggingface_id(&catalog_text, model_id) else {
            println!("{STATUS_PREFIX} skipped model={model_id} reason=not-in-catalog");
            continue;
        };
        catalog_text = write_measured_speed(
            &catalog_text,
            &huggingface_id,
            &task_medians,
            measurement_place,
        )?;
        println!("{STATUS_PREFIX} wrote model={model_id}");
    }
    std::fs::write(&catalog_path, catalog_text)
        .map_err(|write_error| format!("could not write {}: {write_error}", catalog_path.display()))
}

/// The Hugging Face identity of the catalog entry whose model ID (the leaf of
/// its identity) is this one.
fn catalog_huggingface_id(catalog_text: &str, model_id: &str) -> Option<String> {
    let catalog_document: serde_json::Value = serde_json::from_str(catalog_text).ok()?;
    catalog_document["entries"]
        .as_array()?
        .iter()
        .filter_map(|entry| entry["huggingface_id"].as_str())
        .find(|huggingface_id| huggingface_id.rsplit('/').next() == Some(model_id))
        .map(str::to_owned)
}
