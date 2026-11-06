//! Display formatting shared by the CLI verbs.

use astronomical_ipc_protocol::NeuralEnginePrefillStatus;

/// The user-facing gigabyte is the one Apple prints on the box (AGENTS.md):
/// 1 GB = 1,073,741,824 bytes (2³⁰). Two decimals with trailing zeros and
/// the trailing dot trimmed, so `1610612736` renders as `1.5` and
/// `2147483648` as `2`.
pub(crate) fn format_gigabytes(size_bytes: u64) -> String {
    let gigabytes = size_bytes as f64 / 1_073_741_824.0;
    format!("{gigabytes:.2}")
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_owned()
}

/// The line `respond` keeps redrawing while it waits for a load, and `status` shows.
pub(crate) const MODEL_LOAD_LINE: &str = "Loading the model";

/// What the Neural Engine is doing for the loaded model. Nothing for `Off`, so a
/// user who never enabled the setting keeps the usual status lines.
pub(crate) fn render_neural_engine_prefill(
    neural_engine_prefill: &NeuralEnginePrefillStatus,
) -> Option<String> {
    match neural_engine_prefill {
        NeuralEnginePrefillStatus::Off => None,
        NeuralEnginePrefillStatus::Preparing {
            completed_program_count,
            total_program_count,
            estimated_remaining_ms,
            is_waiting_to_measure,
            ..
        } => {
            let count_line = if *is_waiting_to_measure {
                format!(
                    "{total_program_count} programs built, measuring their memory at the next pause in requests; prefill on the GPU until then"
                )
            } else {
                format!(
                    "preparing, {completed_program_count}/{total_program_count} programs; prefill on the GPU until ready"
                )
            };
            Some(match estimated_remaining_ms {
                Some(estimated_remaining_ms) if *estimated_remaining_ms > 0 => format!(
                    "{count_line} (about {} left)",
                    render_remaining_time(*estimated_remaining_ms)
                ),
                _ => count_line,
            })
        }
        NeuralEnginePrefillStatus::Active {
            layer_program_count,
            measured_speed_percent,
            ..
        } => Some(match measured_speed_percent {
            Some(measured_speed_percent) => format!(
                "engaged, {layer_program_count} layer programs, measured {:.2}x the GPU alone",
                f64::from(*measured_speed_percent) / 100.0
            ),
            None => format!(
                "engaged, {layer_program_count} layer programs; comparing with the GPU alone during long prompts"
            ),
        }),
        NeuralEnginePrefillStatus::Unavailable { reason } => {
            Some(format!("unavailable, prefill stays on the GPU: {reason}"))
        }
    }
}

/// Rounded up, so the last moments of a preparation never claim "0 s left".
fn render_remaining_time(remaining_ms: u64) -> String {
    let remaining_seconds = remaining_ms.div_ceil(1000);
    if remaining_seconds < 60 {
        return format!("{remaining_seconds} s");
    }
    format!(
        "{} min {} s",
        remaining_seconds / 60,
        remaining_seconds % 60
    )
}

#[cfg(test)]
#[path = "formatting_tests.rs"]
mod tests;
