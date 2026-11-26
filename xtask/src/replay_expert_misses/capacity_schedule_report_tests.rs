use memory_policy::astro_expert_cache_model::CapacityMoment;

use super::{WiredOutsideMlxReading, reserve_by_moment};

const GIBIBYTE: u64 = 1 << 30;

/// A moment at a 34 GiB limit under a 100 GiB wire line, from a reading at
/// `reading_milliseconds`: 1 GiB of slack leaves MLX 99 GiB less what is
/// wired outside it.
fn moment_at(reading_milliseconds: u64, wired_outside_mlx_gibibytes: u64) -> CapacityMoment {
    CapacityMoment {
        reply_token_index: 0,
        kept_expert_capacity_bytes: 20 * GIBIBYTE,
        system_wired_reserve_bytes: 0,
        wired_outside_mlx_bytes: wired_outside_mlx_gibibytes * GIBIBYTE,
        wired_reading_milliseconds: reading_milliseconds,
        limit_bytes: 34 * GIBIBYTE,
        system_wire_limit_bytes: 100 * GIBIBYTE,
        released_expert_bytes: 0,
    }
}

/// Steady at 63 GiB outside MLX (no reserve), a release lag inflating two
/// readings to 67 GiB (2 GiB reserve as read), then steady again.
fn moments_with_a_release_lag() -> Vec<CapacityMoment> {
    vec![
        moment_at(0, 63),
        moment_at(50, 67),
        moment_at(100, 67),
        moment_at(250, 63),
    ]
}

#[test]
fn should_reserve_for_a_release_lag_as_read() {
    let reserves = reserve_by_moment(
        &moments_with_a_release_lag(),
        WiredOutsideMlxReading::AsRead,
    );
    assert_eq!(reserves, vec![0, 2 * GIBIBYTE, 2 * GIBIBYTE, 0]);
}

#[test]
fn should_reserve_nothing_for_a_release_lag_inside_the_window() {
    let reserves = reserve_by_moment(
        &moments_with_a_release_lag(),
        WiredOutsideMlxReading::LowestOfWindow(300),
    );
    assert_eq!(reserves, vec![0; 4]);
}

#[test]
fn should_take_moments_that_share_a_reading_as_one_reading() {
    let moments = [moment_at(0, 63), moment_at(400, 67), moment_at(400, 67)];
    let reserves = reserve_by_moment(&moments, WiredOutsideMlxReading::LowestOfWindow(300));
    assert_eq!(
        reserves,
        vec![0, 0, 0],
        "the reading at 0 ms still bounds the window"
    );
}

#[test]
fn should_reserve_once_a_rise_outlasts_the_window() {
    let moments = [moment_at(0, 63), moment_at(50, 67), moment_at(400, 67)];
    let reserves = reserve_by_moment(&moments, WiredOutsideMlxReading::LowestOfWindow(300));
    assert_eq!(reserves, vec![0, 0, 2 * GIBIBYTE]);
}

#[test]
fn should_keep_the_engines_reserve_before_any_wired_reading() {
    let mut before_any_reading = moment_at(0, 0);
    before_any_reading.system_wire_limit_bytes = 0;
    before_any_reading.system_wired_reserve_bytes = GIBIBYTE;
    let reserves = reserve_by_moment(&[before_any_reading], WiredOutsideMlxReading::AsRead);
    assert_eq!(reserves, vec![GIBIBYTE]);
}

#[test]
fn should_reserve_nothing_at_all_as_the_bound() {
    let reserves = reserve_by_moment(
        &moments_with_a_release_lag(),
        WiredOutsideMlxReading::NoReserve,
    );
    assert_eq!(reserves, vec![0; 4]);
}

#[test]
fn should_print_no_schedule_rows_for_a_recording_without_moments() {
    let saved_replay_case = super::super::replay_expert_misses_tests::saved_case(Vec::new());
    assert!(super::report_capacity_schedule(&saved_replay_case).is_empty());
}

#[test]
fn should_print_the_schedule_and_every_rule_for_a_recording_with_moments() {
    let mut saved_replay_case = super::super::replay_expert_misses_tests::saved_case(Vec::new());
    saved_replay_case.capacity_moments = moments_with_a_release_lag();
    let report_lines = super::report_capacity_schedule(&saved_replay_case);
    assert!(report_lines[0].contains("capacity_schedule moments=4"));
    for expected_row in [
        "schedule=recorded trim=DropLeastOccupiedLayer",
        "schedule=recorded trim=GiveUpLowestPriorityExpert",
        "schedule=reserve_as_read",
        "schedule=reserve_lowest_of_300_ms",
        "schedule=reserve_lowest_of_600_ms",
        "schedule=no_reserve",
    ] {
        assert!(
            report_lines
                .iter()
                .any(|report_line| report_line.contains(expected_row)),
            "missing row {expected_row}: {report_lines:#?}"
        );
    }
}
