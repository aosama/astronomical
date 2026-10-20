//! Human-readable ranking of the most sensitive modules.

use std::fmt::Write as _;

use super::SensitivityTable;

#[cfg(test)]
#[path = "report_tests.rs"]
mod report_tests;

/// Renders, for each observed bit width, the `top_n` modules with the highest
/// KL at that width, with their parameter counts.
///
/// Work to port (OptiQ's `print_sensitivity_report`): for each width
/// ascending, sort the results that have that width by descending KL, and
/// emit one line per module `rank. module_name  sens=1.234e-03  params=1,234`.
#[must_use]
pub fn render_sensitivity_report(table: &SensitivityTable, top_n: usize) -> String {
    let mut report = String::new();
    if table.results.is_empty() {
        let _ = writeln!(report, "  (no modules to report)");
        return report;
    }
    let _ = top_n;
    todo!("rank per width and render the top-N lines")
}
