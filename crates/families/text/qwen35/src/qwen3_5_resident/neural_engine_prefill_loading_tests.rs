use super::estimated_remaining_ms;

#[test]
fn should_offer_no_estimate_before_the_first_program_finishes() {
    assert_eq!(estimated_remaining_ms(5_000, 0, 56), None);
}

#[test]
fn should_extrapolate_from_the_programs_finished_so_far() {
    assert_eq!(estimated_remaining_ms(10_000, 10, 56), Some(46_000));
}

#[test]
fn should_estimate_nothing_left_when_every_program_finished() {
    assert_eq!(estimated_remaining_ms(10_000, 56, 56), Some(0));
}

#[test]
fn should_not_underflow_on_a_count_past_the_total() {
    assert_eq!(estimated_remaining_ms(10_000, 60, 56), Some(0));
}

#[test]
fn should_saturate_rather_than_wrap_on_a_huge_clock() {
    assert_eq!(estimated_remaining_ms(u64::MAX, 1, 3), Some(u64::MAX));
}
