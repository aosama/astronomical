//! The replay lab's rows for a reply served under the capacity it really
//! had (`capacity_schedule`): today's engine first, which must come near the
//! engine's own misses and table drops for the lab to stand for it, then the
//! same routes under other trim rules and other reserves for macOS's wired
//! line.

use memory_policy::astro_expert_cache_model::{
    CapacityMoment, CapacityTrim, ReplayCase, SavedReplayCase, capacity_by_token,
    capacity_by_token_under_reserve, limit_past_wired_line_bytes, replay_under_capacity_schedule,
};

use memory_policy::ledger::wired_outside_mlx_window::{
    RELEASE_LAG_WINDOW_MILLISECONDS, WiredOutsideMlxWindow,
};

use super::capacity_divisions::divided_like_todays_engine;
use super::todays_engine_rule;

const GIBIBYTE: f64 = (1_u64 << 30) as f64;

/// How the memory wired outside MLX is read before the reserve is computed.
#[derive(Clone, Copy)]
enum WiredOutsideMlxReading {
    /// Each reading as read: the engine before the release-lag window.
    AsRead,
    /// The lowest reading of a window of this many milliseconds, as the
    /// ledger's `WiredOutsideMlxWindow` reads it; 300 is today's engine.
    LowestOfWindow(u64),
    /// No reserve at all: the bound on what the reserve costs.
    NoReserve,
}

impl WiredOutsideMlxReading {
    fn name(self) -> String {
        match self {
            Self::AsRead => "reserve_as_read".to_owned(),
            Self::LowestOfWindow(window_milliseconds) => {
                format!("reserve_lowest_of_{window_milliseconds}_ms")
            }
            Self::NoReserve => "no_reserve".to_owned(),
        }
    }

    const fn window_milliseconds(self) -> u64 {
        match self {
            Self::AsRead | Self::NoReserve => 0,
            Self::LowestOfWindow(window_milliseconds) => window_milliseconds,
        }
    }
}

/// Every row for one saved case with capacity moments; none without them.
pub(super) fn report_capacity_schedule(saved_replay_case: &SavedReplayCase) -> Vec<String> {
    let capacity_moments = &saved_replay_case.capacity_moments;
    if capacity_moments.is_empty() {
        return Vec::new();
    }
    let engine_division = divided_like_todays_engine(saved_replay_case);
    let reply_token_count = engine_division.routed_reply.token_count();
    let recorded_schedule = capacity_by_token(
        capacity_moments,
        reply_token_count,
        engine_division.capacity_bytes,
    );
    let mut report_lines = vec![schedule_summary_line(saved_replay_case, &recorded_schedule)];
    for capacity_trim in [
        CapacityTrim::DropLeastOccupiedLayer,
        CapacityTrim::GiveUpLowestPriorityExpert,
    ] {
        report_lines.push(scheduled_row(
            saved_replay_case,
            &engine_division,
            ("recorded", &recorded_schedule),
            capacity_trim,
        ));
    }
    for wired_outside_mlx_reading in [
        WiredOutsideMlxReading::AsRead,
        WiredOutsideMlxReading::LowestOfWindow(RELEASE_LAG_WINDOW_MILLISECONDS),
        WiredOutsideMlxReading::LowestOfWindow(2 * RELEASE_LAG_WINDOW_MILLISECONDS),
        WiredOutsideMlxReading::NoReserve,
    ] {
        let reserve_by_moment = reserve_by_moment(capacity_moments, wired_outside_mlx_reading);
        let schedule = capacity_by_token_under_reserve(
            capacity_moments,
            &reserve_by_moment,
            reply_token_count,
            engine_division.capacity_bytes,
        );
        report_lines.push(scheduled_row(
            saved_replay_case,
            &engine_division,
            (&wired_outside_mlx_reading.name(), &schedule),
            CapacityTrim::DropLeastOccupiedLayer,
        ));
    }
    report_lines
}

/// The schedule itself: how far it moved, and what the engine recorded
/// giving back, for comparison with the lab's rows.
fn schedule_summary_line(saved_replay_case: &SavedReplayCase, recorded_schedule: &[u64]) -> String {
    let mut lowest_bytes = u64::MAX;
    let mut highest_bytes = 0;
    let mut falling_token_count = 0;
    let mut previous_bytes: Option<u64> = None;
    for &capacity_bytes in recorded_schedule {
        lowest_bytes = lowest_bytes.min(capacity_bytes);
        highest_bytes = highest_bytes.max(capacity_bytes);
        if let Some(previous) = previous_bytes
            && capacity_bytes < previous
        {
            falling_token_count += 1;
        }
        previous_bytes = Some(capacity_bytes);
    }
    let mut engine_released_bytes = 0_u64;
    for moment in &saved_replay_case.capacity_moments {
        engine_released_bytes = engine_released_bytes.saturating_add(moment.released_expert_bytes);
    }
    if recorded_schedule.is_empty() {
        lowest_bytes = 0;
    }
    format!(
        "[replay-expert-misses] workload={} capacity_schedule moments={} capacity_gb=[{:.2}, {:.2}] tokens_where_capacity_fell={falling_token_count} engine_released_gb={:.2}",
        saved_replay_case.workload_name,
        saved_replay_case.capacity_moments.len(),
        lowest_bytes as f64 / GIBIBYTE,
        highest_bytes as f64 / GIBIBYTE,
        engine_released_bytes as f64 / GIBIBYTE,
    )
}

fn scheduled_row(
    saved_replay_case: &SavedReplayCase,
    engine_division: &ReplayCase,
    (schedule_name, schedule): (&str, &[u64]),
    capacity_trim: CapacityTrim,
) -> String {
    let mut keeping_rule = todays_engine_rule(saved_replay_case);
    let scheduled_outcome = replay_under_capacity_schedule(
        engine_division,
        schedule,
        keeping_rule.as_mut(),
        capacity_trim,
    );
    format!(
        "[replay-expert-misses] workload={} division=todays_engine schedule={schedule_name} trim={capacity_trim:?} misses_per_token={:.2} dropped_layers={} trimmed_experts={} misses_per_token_in_the_first_48={:.2} misses_per_token_after_them={:.2}",
        saved_replay_case.workload_name,
        scheduled_outcome.replay_outcome.misses_per_token(),
        scheduled_outcome.dropped_layer_count,
        scheduled_outcome.trimmed_expert_count,
        scheduled_outcome.replay_outcome.early_misses_per_token(),
        scheduled_outcome.replay_outcome.later_misses_per_token(),
    )
}

/// The reserve each moment would have held had the memory wired outside MLX
/// been read under `wired_outside_mlx_reading`.
///
/// A moment is recorded every time a capacity is applied, so several moments
/// share one wired reading; the window takes each reading once, by its time.
/// A moment made before any wired reading keeps the reserve the engine held.
fn reserve_by_moment(
    capacity_moments: &[CapacityMoment],
    wired_outside_mlx_reading: WiredOutsideMlxReading,
) -> Vec<u64> {
    let mut reserves = Vec::with_capacity(capacity_moments.len());
    let mut window = WiredOutsideMlxWindow::with_window_milliseconds(
        wired_outside_mlx_reading.window_milliseconds(),
    );
    let mut counted_outside_bytes = 0;
    let mut last_reading_milliseconds: Option<u64> = None;
    for moment in capacity_moments {
        if moment.system_wire_limit_bytes == 0 {
            reserves.push(moment.system_wired_reserve_bytes);
            continue;
        }
        if last_reading_milliseconds != Some(moment.wired_reading_milliseconds) {
            counted_outside_bytes = window.observe(
                moment.wired_reading_milliseconds,
                moment.wired_outside_mlx_bytes,
            );
            last_reading_milliseconds = Some(moment.wired_reading_milliseconds);
        }
        reserves.push(match wired_outside_mlx_reading {
            WiredOutsideMlxReading::NoReserve => 0,
            _ => limit_past_wired_line_bytes(moment, counted_outside_bytes),
        });
    }
    reserves
}

#[cfg(test)]
#[path = "capacity_schedule_report_tests.rs"]
mod tests;
