//! Human-readable ranking of the most sensitive modules.

use std::fmt::Write as _;

use super::{SensitivityResult, SensitivityTable};

#[cfg(test)]
#[path = "report_tests.rs"]
mod report_tests;

/// Renders, for each observed bit width ascending, the `top_n` modules with
/// the highest KL at that width, in OptiQ's `print_sensitivity_report` line
/// format: `rank. module_name  sens=1.234e-03  params=1,234`, with the
/// module name padded to 60 columns.
#[must_use]
pub fn render_sensitivity_report(table: &SensitivityTable, top_n: usize) -> String {
    let mut report = String::new();
    if table.results.is_empty() {
        let _ = writeln!(report, "  (no modules to report)");
        return report;
    }
    for bit_width in table.observed_bit_widths() {
        let mut ranked_results = Vec::new();
        for result in &table.results {
            if let Some(kl_divergence) = result.kl_divergence_at(bit_width) {
                ranked_results.push((kl_divergence, result));
            }
        }
        ranked_results.sort_by(compare_by_descending_divergence);
        let _ = writeln!(
            report,
            "\n  Top {top_n} most-sensitive modules at {}-bit:",
            bit_width.bits()
        );
        for (rank_index, (kl_divergence, result)) in ranked_results.iter().take(top_n).enumerate() {
            let _ = writeln!(
                report,
                "    {:3}. {:60}  sens={}  params={}",
                rank_index + 1,
                result.module_name,
                python_scientific(*kl_divergence),
                grouped_thousands(result.logical_parameter_count)
            );
        }
    }
    report
}

/// Descending by divergence; NaN sorts first, as the most sensitive.
fn compare_by_descending_divergence(
    left: &(f64, &SensitivityResult),
    right: &(f64, &SensitivityResult),
) -> std::cmp::Ordering {
    right.0.total_cmp(&left.0)
}

/// Python's `{:.3e}`: three fraction digits and a signed exponent of at
/// least two digits (`1.234e-03`), where Rust would print `1.234e-3`.
fn python_scientific(value: f64) -> String {
    if !value.is_finite() {
        return if value.is_nan() {
            "nan".to_owned()
        } else if value > 0.0 {
            "inf".to_owned()
        } else {
            "-inf".to_owned()
        };
    }
    let rust_scientific = format!("{value:.3e}");
    let Some((mantissa, exponent)) = rust_scientific.split_once('e') else {
        return rust_scientific;
    };
    let exponent_value: i32 = exponent.parse().unwrap_or(0);
    let exponent_sign = if exponent_value < 0 { '-' } else { '+' };
    format!("{mantissa}e{exponent_sign}{:02}", exponent_value.abs())
}

/// Python's `{:,}` for a count: digits grouped by three with commas.
fn grouped_thousands(count: u64) -> String {
    let digits = count.to_string();
    let mut grouped = String::with_capacity(digits.len() + digits.len() / 3);
    for (digit_index, digit) in digits.chars().enumerate() {
        let digits_remaining = digits.len() - digit_index;
        if digit_index > 0 && digits_remaining % 3 == 0 {
            grouped.push(',');
        }
        grouped.push(digit);
    }
    grouped
}
