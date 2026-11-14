//! One line of progress for a running throughput journey.
//!
//! A measurement on a large model runs for minutes with nothing printed,
//! which reads as a hang. The background sampler prints one of these lines
//! each time it polls the worker, so whoever started the run can see how far
//! it is and roughly how long is left.

use astronomical_daemon::ActiveRequestProgress;

/// The progress line for a request that is reading its prompt or writing its
/// reply; `None` for stages that carry no token counts.
pub(super) fn progress_line_for(active_request_progress: &ActiveRequestProgress) -> Option<String> {
    match active_request_progress {
        ActiveRequestProgress::Prefill {
            processed_tokens,
            total_tokens,
            elapsed_millis,
            ..
        } => Some(progress_line(
            "prompt",
            u64::from(*processed_tokens),
            u64::from(*total_tokens),
            *elapsed_millis,
        )),
        ActiveRequestProgress::Generation {
            generated_token_count,
            maximum_output_tokens,
            elapsed_millis,
        } => Some(progress_line(
            "reply",
            u64::from(*generated_token_count),
            u64::from(*maximum_output_tokens),
            *elapsed_millis,
        )),
        _ => None,
    }
}

/// Prints the progress line for a request that carries token counts.
pub(super) fn print_progress_line(active_request_progress: &ActiveRequestProgress) {
    if let Some(progress_line) = progress_line_for(active_request_progress) {
        eprintln!("[performance-throughput] {progress_line}");
    }
}

/// Formats completed-of-total tokens with a percentage and the time left at
/// the rate seen so far. Before the first token completes there is no rate,
/// so the time left is reported as unknown rather than as zero.
pub(super) fn progress_line(
    stage_name: &str,
    completed_token_count: u64,
    total_token_count: u64,
    elapsed_millis: u64,
) -> String {
    let completed_percent = if total_token_count == 0 {
        0
    } else {
        completed_token_count.saturating_mul(100) / total_token_count
    };
    let elapsed_seconds = elapsed_millis / 1_000;
    let remaining_token_count = total_token_count.saturating_sub(completed_token_count);
    let time_left = if completed_token_count == 0 {
        "unknown".to_owned()
    } else {
        let remaining_millis =
            elapsed_millis.saturating_mul(remaining_token_count) / completed_token_count;
        format!("{}s", remaining_millis / 1_000)
    };
    format!(
        "progress stage={stage_name} {completed_token_count}/{total_token_count} tokens \
         {completed_percent}% elapsed {elapsed_seconds}s left {time_left}"
    )
}

#[cfg(test)]
#[path = "progress_report_tests.rs"]
mod tests;
