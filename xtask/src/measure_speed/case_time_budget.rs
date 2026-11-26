//! How long one measured case may take, and how many rounds that buys.
//!
//! Waiting on a measurement is a cost the person measuring pays every time,
//! so a case sizes itself to two minutes instead of running a fixed number of
//! replies: one round always, and another only while the longest round so far
//! still fits. A fast model gets up to three rounds and a range; a slow one
//! (a dense 27B writes about 16 tokens a second) gets one round, and the
//! release table then asks more of a difference before it states one
//! (`speed_comparison`).

/// Wall-clock seconds one case may take, its model load and warm-up included.
pub const CASE_TIME_BUDGET_SECONDS: f64 = 120.0;
/// Rounds after which a range is as good as it gets for this purpose.
pub const MOST_ROUNDS_A_CASE: usize = 3;

/// Whether another round fits the case's budget, judged by its longest round
/// so far.
#[must_use]
pub fn another_round_fits(case_elapsed_seconds: f64, longest_round_seconds: f64) -> bool {
    case_elapsed_seconds + longest_round_seconds <= CASE_TIME_BUDGET_SECONDS
}

#[cfg(test)]
#[path = "case_time_budget_tests.rs"]
mod case_time_budget_tests;
