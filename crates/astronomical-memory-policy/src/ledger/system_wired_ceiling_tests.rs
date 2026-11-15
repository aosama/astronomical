use super::next_system_wired_reserve_bytes;

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
