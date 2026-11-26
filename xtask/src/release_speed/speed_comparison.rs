//! Compares two releases' speed ledgers and prints the "Speed" section of the
//! release notes: a caption with the Mac and the conditions, then one row a
//! workload with the previous and the current figure.
//!
//! The rules keep the section honest without anyone editing it:
//! - a change is stated only when the two releases' ranges do not overlap,
//!   otherwise the cell says "unchanged", so run-to-run spread is never
//!   published as a gain or a loss;
//! - a slower figure is printed exactly as a faster one is;
//! - figures taken on different Macs are listed side by side with no
//!   percentages, since the difference would be the Mac's;
//! - a workload measured in only one of the two releases is labelled, never
//!   compared with something else;
//! - a long prompt is compared only with one of about the same length, since
//!   a prompt twice as long takes twice as long to read and slows the reply.

use std::fmt::Write as _;

use super::speed_ledger::{
    ComparisonKey, DrafterInUse, FigureSpread, ReleaseSpeedLedger, SpeedMeasurement,
    WeightsPlacement,
};
use crate::measure_speed::measure_speed_options::describe_mac;

const BYTES_PER_GIGABYTE: u64 = 1_073_741_824;
/// Below this, a prompt is read in a fraction of a second and its speed says
/// nothing about prompt reading; only the first token is shown. Short
/// prompts are about 30 tokens and the shortest long one about 1,000.
const LONG_PROMPT_TOKENS: u64 = 500;
/// How far apart two long prompts may be and still be compared. The prompts
/// are cut from the same fixture to a target length, so one workload lands
/// within a few percent of itself between releases; a changed target (10,000
/// to 5,000 tokens in 0.2.135) lands far outside, and comparing across it
/// published a halved prompt as a 58% faster first token.
const COMPARABLE_PROMPT_LENGTH_FRACTION: f64 = 0.10;

/// Whether a larger figure is better (tokens a second) or a smaller one is
/// (seconds to the first token): it decides nothing about the wording, only
/// the unit a figure is printed in.
#[derive(Clone, Copy)]
enum FigureUnit {
    TokensPerSecond,
    Seconds,
    /// Bytes, printed in GB of 2^30 bytes.
    MemoryBytes,
}

/// The markdown section the release notes and the release issue open with.
#[must_use]
pub fn release_speed_section(
    previous_ledger: &ReleaseSpeedLedger,
    current_ledger: &ReleaseSpeedLedger,
) -> String {
    let is_same_machine = previous_ledger.machine == current_ledger.machine;
    let mut section_text = String::from("## Speed\n\n");
    let _ = writeln!(
        section_text,
        "> {}\n",
        caption(previous_ledger, current_ledger, is_same_machine)
    );
    let _ = writeln!(
        section_text,
        "| Model and workload | Prompt reading (tokens a second) | First token | Reply (tokens a second) | Peak memory |\n|---|---|---|---|---|"
    );
    for current_measurement in &current_ledger.measurements {
        let previous_measurement = matching_measurement(previous_ledger, current_measurement);
        let _ = writeln!(
            section_text,
            "{}",
            comparison_row(previous_measurement, current_measurement, is_same_machine)
        );
    }
    let dropped_workloads = workloads_measured_only_before(previous_ledger, current_ledger);
    if !dropped_workloads.is_empty() {
        let _ = writeln!(
            section_text,
            "\nNot measured in this release: {}.",
            dropped_workloads.join("; ")
        );
    }
    section_text
}

fn caption(
    previous_ledger: &ReleaseSpeedLedger,
    current_ledger: &ReleaseSpeedLedger,
    is_same_machine: bool,
) -> String {
    let machine_text = describe_mac(
        &current_ledger.machine.chip,
        current_ledger.machine.memory_bytes / BYTES_PER_GIGABYTE,
    );
    let mut caption_text = format!(
        "{} against {}, measured on a {machine_text}. {}. A change is shown only when the two releases' ranges do not overlap.",
        current_ledger.astronomical_version,
        previous_ledger.astronomical_version,
        reply_conditions_text(current_ledger)
    );
    if !is_same_machine {
        let previous_machine_text = describe_mac(
            &previous_ledger.machine.chip,
            previous_ledger.machine.memory_bytes / BYTES_PER_GIGABYTE,
        );
        let _ = write!(
            caption_text,
            " The previous figures were taken on a {previous_machine_text}, so no change is stated."
        );
    }
    caption_text
}

/// "Replies of 500 tokens, essays at temperature 1.0 and code at 0.7, 3
/// interleaved rounds, median shown" (or "one reply a task" when each task
/// was measured once), read from the measurements themselves.
fn reply_conditions_text(current_ledger: &ReleaseSpeedLedger) -> String {
    let measurements = &current_ledger.measurements;
    let reply_tokens = measurements
        .iter()
        .map(|measurement| measurement.reply_tokens_requested)
        .max()
        .unwrap_or_default();
    let round_count = measurements
        .iter()
        .map(|measurement| measurement.figures.round_count)
        .min()
        .unwrap_or_default();
    let mut temperature_texts = Vec::new();
    for (task_prefix, task_name) in [("essay", "essays"), ("code", "code")] {
        if let Some(task_measurement) = measurements
            .iter()
            .find(|measurement| measurement.task.starts_with(task_prefix))
        {
            temperature_texts.push(format!(
                "{task_name} at temperature {:.1}",
                task_measurement.temperature
            ));
        }
    }
    let rounds_text = if round_count == 1 {
        "one reply a task".to_owned()
    } else {
        format!("{round_count} interleaved rounds, median shown")
    };
    format!(
        "Replies of {reply_tokens} tokens, {}, {rounds_text}",
        temperature_texts.join(" and ")
    )
}

/// The other release's measurement of the same workload: the same model,
/// revision, memory limit and task, after a prompt of about the same length.
fn matching_measurement<'ledger>(
    ledger: &'ledger ReleaseSpeedLedger,
    wanted_measurement: &SpeedMeasurement,
) -> Option<&'ledger SpeedMeasurement> {
    let wanted_key: ComparisonKey = wanted_measurement.comparison_key();
    ledger.measurements.iter().find(|measurement| {
        measurement.comparison_key() == wanted_key
            && prompts_are_comparable(
                measurement.figures.prompt_tokens,
                wanted_measurement.figures.prompt_tokens,
            )
    })
}

/// Two short prompts always compare; two long ones only within
/// `COMPARABLE_PROMPT_LENGTH_FRACTION` of the longer.
#[expect(
    clippy::cast_precision_loss,
    reason = "prompt token counts, far inside f64's exact range"
)]
fn prompts_are_comparable(first_prompt_tokens: u64, second_prompt_tokens: u64) -> bool {
    let is_first_short = first_prompt_tokens < LONG_PROMPT_TOKENS;
    let is_second_short = second_prompt_tokens < LONG_PROMPT_TOKENS;
    if is_first_short || is_second_short {
        return is_first_short == is_second_short;
    }
    let longer_prompt_tokens = first_prompt_tokens.max(second_prompt_tokens) as f64;
    let length_difference = first_prompt_tokens.abs_diff(second_prompt_tokens) as f64;
    length_difference <= longer_prompt_tokens * COMPARABLE_PROMPT_LENGTH_FRACTION
}

fn comparison_row(
    previous_measurement: Option<&SpeedMeasurement>,
    current_measurement: &SpeedMeasurement,
    is_same_machine: bool,
) -> String {
    let prompt_reading_cell = if current_measurement.figures.prompt_tokens < LONG_PROMPT_TOKENS {
        "–".to_owned()
    } else {
        figure_cell(
            previous_measurement.map(|previous| &previous.figures.prefill_tokens_per_second),
            &current_measurement.figures.prefill_tokens_per_second,
            FigureUnit::TokensPerSecond,
            is_same_machine,
        )
    };
    let first_token_cell = figure_cell(
        previous_measurement.map(|previous| &previous.figures.seconds_to_first_token),
        &current_measurement.figures.seconds_to_first_token,
        FigureUnit::Seconds,
        is_same_machine,
    );
    let reply_cell = figure_cell(
        previous_measurement.map(|previous| &previous.figures.reply_tokens_per_second),
        &current_measurement.figures.reply_tokens_per_second,
        FigureUnit::TokensPerSecond,
        is_same_machine,
    );
    format!(
        "| {} | {prompt_reading_cell} | {first_token_cell} | {reply_cell} | {} |",
        workload_label(previous_measurement, current_measurement),
        peak_memory_cell(previous_measurement, current_measurement, is_same_machine)
    )
}

/// The most MLX memory a reply had in use; a dash for a release measured
/// before the engine reported it.
fn peak_memory_cell(
    previous_measurement: Option<&SpeedMeasurement>,
    current_measurement: &SpeedMeasurement,
    is_same_machine: bool,
) -> String {
    let Some(current_peak) = current_measurement.figures.peak_mlx_memory_bytes.as_ref() else {
        return "–".to_owned();
    };
    let previous_peak =
        previous_measurement.and_then(|previous| previous.figures.peak_mlx_memory_bytes.as_ref());
    if previous_measurement.is_some() && previous_peak.is_none() {
        // The row is not new; only its memory figure is.
        return figure_text(current_peak.median, FigureUnit::MemoryBytes);
    }
    figure_cell(
        previous_peak,
        current_peak,
        FigureUnit::MemoryBytes,
        is_same_machine,
    )
}

fn figure_cell(
    previous_spread: Option<&FigureSpread>,
    current_spread: &FigureSpread,
    figure_unit: FigureUnit,
    is_same_machine: bool,
) -> String {
    let current_text = figure_text(current_spread.median, figure_unit);
    let Some(previous_spread) = previous_spread else {
        return format!("{current_text} (new)");
    };
    let previous_text = figure_text(previous_spread.median, figure_unit);
    if !is_same_machine {
        return format!("{previous_text} → {current_text}");
    }
    if !current_spread.is_apart_from(previous_spread) {
        return format!("{current_text} (unchanged)");
    }
    format!(
        "{previous_text} → {current_text} ({})",
        percent_change(previous_spread.median, current_spread.median)
    )
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "a rounded figure of tokens a second, far inside i64"
)]
fn figure_text(figure: f64, figure_unit: FigureUnit) -> String {
    match figure_unit {
        FigureUnit::Seconds => format!("{figure:.1} s"),
        FigureUnit::MemoryBytes => format!("{:.1} GB", figure / BYTES_PER_GIGABYTE as f64),
        FigureUnit::TokensPerSecond => with_thousands_separators(figure.round() as i64),
    }
}

fn with_thousands_separators(whole_figure: i64) -> String {
    let digits_text = whole_figure.unsigned_abs().to_string();
    let mut grouped_text = String::new();
    for (digit_index, digit) in digits_text.chars().enumerate() {
        if digit_index > 0 && (digits_text.len() - digit_index) % 3 == 0 {
            grouped_text.push(',');
        }
        grouped_text.push(digit);
    }
    if whole_figure < 0 {
        grouped_text.insert(0, '-');
    }
    grouped_text
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "a whole percentage of two positive figures"
)]
fn percent_change(previous_figure: f64, current_figure: f64) -> String {
    if previous_figure <= 0.0 {
        return "new".to_owned();
    }
    let whole_percent =
        ((current_figure - previous_figure) / previous_figure * 100.0).round() as i64;
    if whole_percent > 0 {
        format!("+{whole_percent}%")
    } else {
        format!("{whole_percent}%")
    }
}

fn workload_label(
    previous_measurement: Option<&SpeedMeasurement>,
    current_measurement: &SpeedMeasurement,
) -> String {
    let memory_text = memory_setting_text(current_measurement.maximum_mlx_memory_gb);
    let placement_text = match current_measurement.weights_placement {
        WeightsPlacement::Streamed => ", streamed from the SSD",
        WeightsPlacement::InMemory => "",
    };
    format!(
        "{}, {memory_text}{placement_text}, {}{}",
        current_measurement.model_id,
        task_text(
            &current_measurement.task,
            current_measurement.figures.prompt_tokens
        ),
        drafter_text(
            previous_measurement.map(|previous| previous.drafter),
            current_measurement.drafter
        )
    )
}

fn memory_setting_text(maximum_mlx_memory_gb: Option<u64>) -> String {
    maximum_mlx_memory_gb.map_or_else(
        || "this Mac's own memory limit".to_owned(),
        |memory_gb| format!("{memory_gb} GB limit"),
    )
}

/// "writing code after a 10,000-token prompt": the task in the plain words
/// the model cards use, with the prompt's real size.
fn task_text(task: &str, prompt_tokens: u64) -> String {
    let writing_text = if task.starts_with("code") {
        "writing code"
    } else {
        "writing an essay"
    };
    if prompt_tokens < LONG_PROMPT_TOKENS {
        return format!("{writing_text} after a short prompt");
    }
    #[expect(
        clippy::cast_possible_wrap,
        reason = "a prompt's token count, far inside i64"
    )]
    let rounded_prompt_tokens = ((prompt_tokens + 500) / 1_000 * 1_000) as i64;
    format!(
        "{writing_text} after a {}-token prompt",
        with_thousands_separators(rounded_prompt_tokens)
    )
}

/// The drafter in use, and whether that changed since the previous release:
/// the reason a row moved is as useful to a reader as the figure.
fn drafter_text(previous_drafter: Option<DrafterInUse>, current_drafter: DrafterInUse) -> String {
    let drafter_name = match current_drafter {
        DrafterInUse::None => None,
        DrafterInUse::Dflash => Some("DFlash"),
        DrafterInUse::Dflash2 => Some("DFlash2"),
    };
    match (previous_drafter, drafter_name) {
        (Some(previous), Some(name)) if previous != current_drafter => {
            format!(", now drafting with {name}")
        }
        (Some(previous), None) if previous != DrafterInUse::None => {
            ", no longer drafting".to_owned()
        }
        (_, Some(name)) => format!(", {name} drafter"),
        (_, None) => String::new(),
    }
}

fn workloads_measured_only_before(
    previous_ledger: &ReleaseSpeedLedger,
    current_ledger: &ReleaseSpeedLedger,
) -> Vec<String> {
    previous_ledger
        .measurements
        .iter()
        .filter(|previous| matching_measurement(current_ledger, previous).is_none())
        .map(|previous| {
            format!(
                "{}, {}, {}",
                previous.model_id,
                memory_setting_text(previous.maximum_mlx_memory_gb),
                task_text(&previous.task, previous.figures.prompt_tokens)
            )
        })
        .collect()
}

#[cfg(test)]
#[path = "speed_comparison_tests.rs"]
mod speed_comparison_tests;
