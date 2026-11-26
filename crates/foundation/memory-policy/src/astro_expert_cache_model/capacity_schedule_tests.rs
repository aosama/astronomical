//! The capacity schedule a recorded reply is replayed under.

use super::{
    CapacityMoment, capacity_by_token, capacity_by_token_under_reserve,
    limit_past_wired_line_bytes, rebased_capacity_moments,
};

const GIBIBYTE: u64 = 1 << 30;

fn moment(
    reply_token_index: u32,
    capacity_gibibytes: u64,
    reserve_gibibytes: u64,
) -> CapacityMoment {
    CapacityMoment {
        reply_token_index,
        kept_expert_capacity_bytes: capacity_gibibytes * GIBIBYTE,
        system_wired_reserve_bytes: reserve_gibibytes * GIBIBYTE,
        wired_outside_mlx_bytes: 0,
        wired_reading_milliseconds: 0,
        limit_bytes: 34 * GIBIBYTE,
        system_wire_limit_bytes: 100 * GIBIBYTE,
        released_expert_bytes: 0,
    }
}

#[test]
fn should_serve_every_token_at_the_starting_capacity_with_no_moments() {
    assert_eq!(capacity_by_token(&[], 3, 7), vec![7, 7, 7]);
}

#[test]
fn should_carry_a_capacity_forward_from_its_token() {
    let moments = [moment(1, 20, 0), moment(3, 24, 0)];
    assert_eq!(
        capacity_by_token(&moments, 5, 22 * GIBIBYTE),
        vec![
            22 * GIBIBYTE,
            20 * GIBIBYTE,
            20 * GIBIBYTE,
            24 * GIBIBYTE,
            24 * GIBIBYTE
        ]
    );
}

#[test]
fn should_serve_a_token_at_the_lowest_capacity_applied_during_it() {
    // A dip and a recovery before token 1: what was given back stays given back.
    let moments = [moment(1, 18, 2), moment(1, 26, 0)];
    assert_eq!(
        capacity_by_token(&moments, 3, 26 * GIBIBYTE),
        vec![26 * GIBIBYTE, 18 * GIBIBYTE, 26 * GIBIBYTE]
    );
}

#[test]
fn should_ignore_moments_past_the_reply() {
    assert_eq!(capacity_by_token(&[moment(9, 1, 0)], 2, 5), vec![5, 5]);
}

#[test]
fn should_add_back_the_reserve_another_rule_would_not_have_held() {
    let moments = [moment(0, 22, 2), moment(1, 24, 0)];
    assert_eq!(
        capacity_by_token_under_reserve(&moments, &[0, 0], 2, 24 * GIBIBYTE),
        vec![24 * GIBIBYTE, 24 * GIBIBYTE]
    );
    assert_eq!(
        capacity_by_token_under_reserve(&moments, &[3 * GIBIBYTE, 0], 2, 24 * GIBIBYTE),
        vec![21 * GIBIBYTE, 24 * GIBIBYTE],
        "a larger reserve than the engine held takes from the capacity"
    );
}

#[test]
fn should_reserve_what_the_limit_has_past_the_room_under_the_wired_line() {
    let at_the_line = moment(0, 0, 0);
    // 100 GiB line, 1 GiB slack, 60 GiB wired outside MLX: MLX has 39 GiB, a
    // 34 GiB limit needs nothing kept unused.
    assert_eq!(limit_past_wired_line_bytes(&at_the_line, 60 * GIBIBYTE), 0);
    // 70 GiB outside leaves 29 GiB: 5 GiB of the limit is kept unused.
    assert_eq!(
        limit_past_wired_line_bytes(&at_the_line, 70 * GIBIBYTE),
        5 * GIBIBYTE
    );
    assert_eq!(
        limit_past_wired_line_bytes(&at_the_line, 200 * GIBIBYTE),
        34 * GIBIBYTE,
        "wiring outside MLX past the line leaves the whole limit unused"
    );
}

#[test]
fn should_keep_the_engines_reserve_for_moments_with_no_other_reserve() {
    let moments = [moment(0, 22, 2), moment(1, 24, 1)];
    assert_eq!(
        capacity_by_token_under_reserve(&moments, &[0], 2, 24 * GIBIBYTE),
        vec![24 * GIBIBYTE, 24 * GIBIBYTE],
        "the second moment keeps its own 1 GiB reserve and its recorded capacity"
    );
}

#[test]
fn should_leave_moments_as_they_are_when_no_routes_were_dropped() {
    let moments = [moment(0, 20, 0), moment(5, 21, 0)];
    assert_eq!(rebased_capacity_moments(&moments, 0), moments.to_vec());
}

#[test]
fn should_index_moments_against_the_kept_routes() {
    let moments = [moment(100, 20, 0), moment(130, 21, 0)];
    let rebased = rebased_capacity_moments(&moments, 100);
    assert_eq!(rebased[0].reply_token_index, 0);
    assert_eq!(rebased[1].reply_token_index, 30);
}

#[test]
fn should_carry_moments_from_before_the_kept_routes_to_their_first_token() {
    let moments = [moment(10, 18, 0), moment(150, 21, 0)];
    let rebased = rebased_capacity_moments(&moments, 100);
    assert_eq!(
        rebased[0].reply_token_index, 0,
        "the capacity the kept routes began under"
    );
    assert_eq!(capacity_by_token(&rebased, 51, 0)[0], 18 * GIBIBYTE);
}
