//! Serves every reply the engine probe recorded (`cargo measure-expert-misses`)
//! again under each keeping rule and each way of dividing the
//! `astro_expert_cache`, and prints per reply token the misses each makes and
//! the layers it served while they were only partly held (each a stop to copy
//! the route to the host): the replay lab's report. Run it as
//! `cargo replay-expert-misses EVIDENCE_DIRECTORY` (`cargo xtask replay-expert-misses`).
//!
//! Ways of dividing the capacity:
//! - `todays_engine`: the layers held whole when the reply began stay whole,
//!   layers that held nothing at either end keep nothing, and every other
//!   layer gets an equal table from what is left, at most two replacements per
//!   layer per token. Its row must come within ten percent of the engine's own
//!   count for the lab to stand for the engine.
//! - `shared`: one capacity across every layer.
//! - `shared_pinning_whole_layers`: the layers held whole when the reply began
//!   stay whole (their routes never stop), the rest share what is left.
//! - `divided_by_demand`: each layer's table sized to its share of the
//!   distinct experts the reply used, an oracle for how well tables sized per
//!   layer could ever do.

use std::path::Path;
use std::time::Instant;

use astronomical_memory_policy::astro_expert_cache_model::{
    BeladyMinimum, CappedReplacements, DecayedFrequency, FrequencyAdmittedRecency, KeepingRule,
    LeastFrequentlyUsed, LeastRecentlyUsed, ReplayCase, SavedReplayCase, find_saved_case_documents,
    read_saved_replay_case, replay,
};

mod capacity_divisions;

use capacity_divisions::{
    divided_by_demand, divided_like_todays_engine, shared_pinning_whole_layers,
};

/// Runs the report over `arguments[0]`, the evidence directory.
pub(crate) fn run(arguments: &[String]) -> Result<(), String> {
    let [evidence_directory] = arguments else {
        return Err("usage: cargo xtask replay-expert-misses EVIDENCE_DIRECTORY".to_owned());
    };
    for report_line in replay_report(Path::new(evidence_directory))? {
        println!("{report_line}");
    }
    Ok(())
}

/// Every report line for every saved case under `evidence_directory`.
fn replay_report(evidence_directory: &Path) -> Result<Vec<String>, String> {
    let case_document_paths = find_saved_case_documents(evidence_directory)
        .map_err(|listing_error| format!("replay-expert-misses: {listing_error}"))?;
    if case_document_paths.is_empty() {
        return Err("replay-expert-misses: no *.case.json under the evidence directory".to_owned());
    }
    let started_at = Instant::now();
    let mut report_lines = Vec::new();
    for (case_index, case_document_path) in case_document_paths.iter().enumerate() {
        let saved_replay_case = read_saved_replay_case(case_document_path)
            .map_err(|reading_error| format!("replay-expert-misses: {reading_error}"))?;
        report_lines.extend(report_saved_case(&saved_replay_case));
        report_progress(case_index + 1, case_document_paths.len(), started_at);
    }
    Ok(report_lines)
}

fn every_rule() -> Vec<Box<dyn KeepingRule>> {
    vec![
        Box::new(LeastRecentlyUsed::default()),
        Box::new(LeastFrequentlyUsed::default()),
        Box::new(DecayedFrequency::new(8.0)),
        Box::new(DecayedFrequency::new(32.0)),
        Box::new(DecayedFrequency::new(128.0)),
        Box::new(FrequencyAdmittedRecency::default()),
        Box::new(BeladyMinimum::default()),
    ]
}

/// Today's engine keeps a table's experts by a reuse score that halves every
/// eight reply tokens (`RECENT_REUSE_SCORE_HALF_LIFE_DECODE_STEPS`) and swaps
/// at most two per layer per token.
fn todays_engine_rule() -> Box<dyn KeepingRule> {
    Box::new(CappedReplacements::new(
        Box::new(DecayedFrequency::new(8.0)),
        2,
    ))
}

fn report_saved_case(saved_replay_case: &SavedReplayCase) -> Vec<String> {
    let replay_case = &saved_replay_case.replay_case;
    let held_at_start: usize = replay_case
        .held_expert_ids_by_layer_at_start
        .iter()
        .map(Vec::len)
        .sum();
    let mut report_lines = vec![format!(
        "[replay-expert-misses] workload={} reply_tokens={} capacity_gb={:.2} held_at_start={held_at_start}",
        saved_replay_case.workload_name,
        replay_case.routed_reply.token_count(),
        replay_case.capacity_bytes as f64 / f64::from(1_u32 << 30),
    )];
    let workload_name = &saved_replay_case.workload_name;
    report_lines.push(report_replay(
        workload_name,
        "todays_engine",
        &divided_like_todays_engine(saved_replay_case),
        todays_engine_rule(),
    ));
    for keeping_rule in every_rule() {
        report_lines.push(report_replay(
            workload_name,
            "shared",
            replay_case,
            keeping_rule,
        ));
    }
    for division_name in ["shared_pinning_whole_layers", "divided_by_demand"] {
        let divided_case = if division_name == "divided_by_demand" {
            divided_by_demand(replay_case)
        } else {
            shared_pinning_whole_layers(replay_case)
        };
        for keeping_rule in [
            Box::new(LeastRecentlyUsed::default()) as Box<dyn KeepingRule>,
            Box::new(DecayedFrequency::new(32.0)),
            Box::new(BeladyMinimum::default()),
        ] {
            report_lines.push(report_replay(
                workload_name,
                division_name,
                &divided_case,
                keeping_rule,
            ));
        }
    }
    report_lines
}

fn report_replay(
    workload_name: &str,
    division_name: &str,
    replay_case: &ReplayCase,
    mut keeping_rule: Box<dyn KeepingRule>,
) -> String {
    let replay_outcome = replay(replay_case, keeping_rule.as_mut());
    format!(
        "[replay-expert-misses] workload={workload_name} division={division_name} rule=\"{}\" misses_per_token={:.2} partly_held_layers_per_token={:.2}",
        keeping_rule.name(),
        replay_outcome.misses_per_token(),
        replay_outcome.partly_held_layers_per_token(),
    )
}

fn report_progress(completed_case_count: usize, total_case_count: usize, started_at: Instant) {
    let elapsed_seconds = started_at.elapsed().as_secs_f64();
    let seconds_per_case = elapsed_seconds / completed_case_count as f64;
    let remaining_seconds = seconds_per_case * (total_case_count - completed_case_count) as f64;
    eprintln!(
        "[replay-expert-misses] status=progress cases={completed_case_count}/{total_case_count} progress_percent={:.0} elapsed_seconds={elapsed_seconds:.1} ETA_seconds={remaining_seconds:.1}",
        100.0 * completed_case_count as f64 / total_case_count as f64
    );
}

#[cfg(test)]
#[path = "replay_expert_misses_tests.rs"]
mod replay_expert_misses_tests;
