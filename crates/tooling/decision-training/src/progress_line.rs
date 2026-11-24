//! The one-line progress report long calibration steps print.
//!
//! Scoring hundreds of examples on a 9B model, or proving a 10 GB copy,
//! takes minutes, so every such step reports how far it has got, its
//! percentage and an estimate of the time left. The estimate assumes the
//! rest goes at the average pace so far, which is steady for both steps:
//! examples are similar in size, and bytes are bytes.

use std::time::Duration;

/// Below this much progress the pace is too uncertain to project from.
const MINIMUM_FRACTION_FOR_ESTIMATE: f64 = 0.01;

/// `"{stage}: {done} of {total} {unit} (37%), about 2 min left"`.
#[must_use]
pub fn progress_line(
    stage_name: &str,
    completed_count: u64,
    total_count: u64,
    unit_name: &str,
    elapsed: Duration,
) -> String {
    let completed_fraction = if total_count == 0 {
        1.0
    } else {
        completed_count.min(total_count) as f64 / total_count as f64
    };
    let percentage = (completed_fraction * 100.0).floor();
    let time_left_text = time_left_text(completed_fraction, elapsed);
    format!(
        "{stage_name}: {completed_count} of {total_count} {unit_name} ({percentage:.0}%), {time_left_text}"
    )
}

fn time_left_text(completed_fraction: f64, elapsed: Duration) -> String {
    if completed_fraction >= 1.0 {
        return "done".to_owned();
    }
    if completed_fraction < MINIMUM_FRACTION_FOR_ESTIMATE {
        return "estimating time left".to_owned();
    }
    let elapsed_seconds = elapsed.as_secs_f64();
    let seconds_left = elapsed_seconds / completed_fraction - elapsed_seconds;
    format!("about {} left", duration_words(seconds_left))
}

fn duration_words(seconds: f64) -> String {
    let whole_seconds = seconds.max(0.0).round() as u64;
    if whole_seconds < 60 {
        return format!("{whole_seconds} s");
    }
    let whole_minutes = whole_seconds.div_ceil(60);
    if whole_minutes < 60 {
        return format!("{whole_minutes} min");
    }
    format!("{} h {} min", whole_minutes / 60, whole_minutes % 60)
}

#[cfg(test)]
#[path = "progress_line_tests.rs"]
mod tests;
