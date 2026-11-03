//! Step progress for the app pipeline: start and finish lines, a heartbeat
//! with an ETA every five seconds while a step runs, and per-step timings
//! remembered between runs.
//!
//! The ETA comes from the previous run's step durations rather than a fixed
//! guess, because the edit loop's cost swings from seconds (a Swift-only
//! change) to minutes (a cold Rust build) and only history predicts which.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

pub(crate) struct StepReporter {
    /// The log prefix, so `app package` lines are not mistaken for the
    /// edit loop's in a CI log or a terminal running both.
    log_label: &'static str,
    step_names: Vec<&'static str>,
    previous_step_seconds: BTreeMap<String, f64>,
    measured_step_seconds: BTreeMap<String, f64>,
    run_started_at: Instant,
    current_step_index: usize,
    current_step_started_at: Instant,
    timings_path: PathBuf,
}

impl StepReporter {
    pub(crate) fn new(
        log_label: &'static str,
        step_names: Vec<&'static str>,
        timings_path: PathBuf,
    ) -> Self {
        let previous_step_seconds = read_step_timings(&timings_path);
        Self {
            log_label,
            step_names,
            previous_step_seconds,
            measured_step_seconds: BTreeMap::new(),
            run_started_at: Instant::now(),
            current_step_index: 0,
            current_step_started_at: Instant::now(),
            timings_path,
        }
    }

    pub(crate) fn start_step(&mut self, step_name: &'static str) {
        let mut step_index = 0;
        for (candidate_index, candidate_name) in self.step_names.iter().enumerate() {
            if *candidate_name == step_name {
                step_index = candidate_index;
                break;
            }
        }
        self.current_step_index = step_index;
        self.current_step_started_at = Instant::now();
        println!(
            "[{}] step={}/{} name={step_name} status=start eta={}",
            self.log_label,
            step_index + 1,
            self.step_names.len(),
            self.eta_text()
        );
    }

    pub(crate) fn finish_step(&mut self) {
        let step_name = self.step_names[self.current_step_index];
        let step_seconds = self.current_step_started_at.elapsed().as_secs_f64();
        self.measured_step_seconds
            .insert(step_name.to_owned(), step_seconds);
        println!(
            "[{}] step={}/{} name={step_name} status=success elapsed_seconds={step_seconds:.1} total_elapsed_seconds={:.1}",
            self.log_label,
            self.current_step_index + 1,
            self.step_names.len(),
            self.run_started_at.elapsed().as_secs_f64()
        );
    }

    pub(crate) fn print_heartbeat(&self) {
        let step_name = self.step_names[self.current_step_index];
        let completed_fraction = self.current_step_index as f64 / self.step_names.len() as f64;
        println!(
            "[{}] step={}/{} name={step_name} status=running step_elapsed_seconds={:.0} progress={:.0}% eta={}",
            self.log_label,
            self.current_step_index + 1,
            self.step_names.len(),
            self.current_step_started_at.elapsed().as_secs_f64(),
            completed_fraction * 100.0,
            self.eta_text()
        );
    }

    /// Writes this run's timings so the next run can estimate its ETA. A
    /// write failure only costs the next ETA, so it is reported, not fatal.
    pub(crate) fn save_timings(&self) {
        let mut timings_document = serde_json::Map::new();
        for (step_name, step_seconds) in &self.measured_step_seconds {
            timings_document.insert(step_name.clone(), serde_json::json!(step_seconds));
        }
        let timings_text = serde_json::Value::Object(timings_document).to_string();
        if let Err(write_error) = std::fs::write(&self.timings_path, timings_text) {
            eprintln!(
                "[{}] warning: could not save step timings: {write_error}",
                self.log_label
            );
        }
    }

    pub(crate) fn total_elapsed_seconds(&self) -> f64 {
        self.run_started_at.elapsed().as_secs_f64()
    }

    fn eta_text(&self) -> String {
        match estimate_remaining_seconds(
            &self.step_names,
            self.current_step_index,
            self.current_step_started_at.elapsed().as_secs_f64(),
            &self.previous_step_seconds,
        ) {
            Some(remaining_seconds) => format!("{remaining_seconds:.0}s"),
            None => "unknown(first-run)".to_owned(),
        }
    }
}

/// Seconds left: what the current step had left last time (never below
/// zero) plus every later step's previous duration. `None` while any step
/// still ahead has no previous timing, because a partial history would
/// understate the rest.
pub(crate) fn estimate_remaining_seconds(
    step_names: &[&'static str],
    current_step_index: usize,
    current_step_elapsed_seconds: f64,
    previous_step_seconds: &BTreeMap<String, f64>,
) -> Option<f64> {
    let mut remaining_seconds = 0.0;
    for (step_index, step_name) in step_names.iter().enumerate().skip(current_step_index) {
        let previous_seconds = *previous_step_seconds.get(*step_name)?;
        if step_index == current_step_index {
            remaining_seconds += (previous_seconds - current_step_elapsed_seconds).max(0.0);
        } else {
            remaining_seconds += previous_seconds;
        }
    }
    Some(remaining_seconds)
}

fn read_step_timings(timings_path: &PathBuf) -> BTreeMap<String, f64> {
    let mut step_timings = BTreeMap::new();
    let Ok(timings_text) = std::fs::read_to_string(timings_path) else {
        return step_timings;
    };
    let Ok(serde_json::Value::Object(timings_document)) = serde_json::from_str(&timings_text)
    else {
        return step_timings;
    };
    for (step_name, step_seconds) in timings_document {
        if let Some(step_seconds) = step_seconds.as_f64() {
            step_timings.insert(step_name, step_seconds);
        }
    }
    step_timings
}
