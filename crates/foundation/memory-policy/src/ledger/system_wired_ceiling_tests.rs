use super::{
    SystemWiredReading, limit_past_system_wired_ceiling_bytes, next_system_wired_reserve_bytes,
    wired_outside_mlx_bytes, with_wired_outside_mlx_bytes,
};

const GIBIBYTE: u64 = 1 << 30;
/// A kernel limit of 100 GiB keeps the arithmetic readable: the slack is
/// 1 GiB.
const CEILING_BYTES: u64 = 100 * GIBIBYTE;

#[test]
fn should_grow_the_reserve_by_what_the_system_is_over_the_ceiling() {
    assert_eq!(
        next_system_wired_reserve_bytes(GIBIBYTE / 4, CEILING_BYTES + GIBIBYTE / 2, CEILING_BYTES),
        GIBIBYTE / 4 + GIBIBYTE / 2
    );
}

#[test]
fn should_keep_the_reserve_while_the_system_is_within_the_slack_under_the_ceiling() {
    assert_eq!(
        next_system_wired_reserve_bytes(GIBIBYTE, CEILING_BYTES, CEILING_BYTES),
        GIBIBYTE
    );
    assert_eq!(
        next_system_wired_reserve_bytes(GIBIBYTE, CEILING_BYTES - GIBIBYTE, CEILING_BYTES),
        GIBIBYTE,
        "exactly the slack under the ceiling gives nothing back"
    );
}

#[test]
fn should_give_back_only_what_the_system_is_under_beyond_the_slack_and_stop_at_zero() {
    assert_eq!(
        next_system_wired_reserve_bytes(
            2 * GIBIBYTE,
            CEILING_BYTES - GIBIBYTE - GIBIBYTE / 2,
            CEILING_BYTES
        ),
        2 * GIBIBYTE - GIBIBYTE / 2
    );
    assert_eq!(
        next_system_wired_reserve_bytes(GIBIBYTE / 8, 10 * GIBIBYTE, CEILING_BYTES),
        0
    );
}

#[test]
fn should_not_overflow_on_impossible_readings() {
    assert_eq!(
        next_system_wired_reserve_bytes(u64::MAX, u64::MAX, 100),
        u64::MAX
    );
    assert_eq!(next_system_wired_reserve_bytes(0, 0, 0), 0);
}

fn reading(system_wired_bytes: u64, mlx_held_bytes: u64) -> SystemWiredReading {
    SystemWiredReading {
        system_wired_bytes,
        system_wire_limit_bytes: CEILING_BYTES,
        mlx_held_bytes,
        reading_milliseconds: 0,
    }
}

#[test]
fn should_keep_unused_the_part_of_the_limit_past_the_line_less_slack() {
    // 99 GiB under the slack; 10 GiB wired outside MLX leaves MLX 89 GiB.
    assert_eq!(
        limit_past_system_wired_ceiling_bytes(95 * GIBIBYTE, reading(60 * GIBIBYTE, 50 * GIBIBYTE)),
        6 * GIBIBYTE
    );
}

#[test]
fn should_keep_nothing_unused_when_the_limit_fits_under_the_line() {
    assert_eq!(
        limit_past_system_wired_ceiling_bytes(89 * GIBIBYTE, reading(60 * GIBIBYTE, 50 * GIBIBYTE)),
        0,
        "exactly the room left is no reserve"
    );
    assert_eq!(
        limit_past_system_wired_ceiling_bytes(34 * GIBIBYTE, reading(36 * GIBIBYTE, 34 * GIBIBYTE)),
        0
    );
}

#[test]
fn should_keep_the_whole_limit_unused_when_other_wiring_fills_the_line() {
    assert_eq!(
        limit_past_system_wired_ceiling_bytes(20 * GIBIBYTE, reading(120 * GIBIBYTE, GIBIBYTE)),
        20 * GIBIBYTE
    );
    assert_eq!(
        limit_past_system_wired_ceiling_bytes(20 * GIBIBYTE, reading(GIBIBYTE, 5 * GIBIBYTE)),
        0,
        "MLX holding more than the wired total counts nothing outside MLX"
    );
}

#[test]
fn should_read_the_memory_wired_outside_mlx_and_never_below_zero() {
    assert_eq!(
        wired_outside_mlx_bytes(reading(36 * GIBIBYTE, 32 * GIBIBYTE)),
        4 * GIBIBYTE
    );
    assert_eq!(wired_outside_mlx_bytes(reading(GIBIBYTE, 5 * GIBIBYTE)), 0);
}

#[test]
fn should_rewrite_the_wired_total_to_the_steady_outside_share() {
    let rewritten =
        with_wired_outside_mlx_bytes(reading(40 * GIBIBYTE, 32 * GIBIBYTE), 3 * GIBIBYTE);
    assert_eq!(rewritten.system_wired_bytes, 35 * GIBIBYTE);
    assert_eq!(rewritten.mlx_held_bytes, 32 * GIBIBYTE);
}
