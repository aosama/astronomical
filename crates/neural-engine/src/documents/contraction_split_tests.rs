use super::contraction_part_width;

#[test]
fn should_split_a_deep_mlp_into_full_rate_parts() {
    assert_eq!(contraction_part_width(&[4096, 12288]), Some(2048));
}

#[test]
fn should_keep_a_depth_that_already_runs_at_the_full_rate() {
    assert_eq!(contraction_part_width(&[2048, 6144]), Some(2048));
}

#[test]
fn should_take_a_part_that_divides_both_depths_of_a_small_model() {
    assert_eq!(contraction_part_width(&[1024, 3584]), Some(512));
}

#[test]
fn should_keep_exactly_the_deepest_full_rate_depth_whole() {
    assert_eq!(contraction_part_width(&[2048]), Some(2048));
}

#[test]
fn should_split_one_past_the_deepest_full_rate_depth() {
    assert_eq!(contraction_part_width(&[2049]), Some(683));
}

#[test]
fn should_refuse_depths_that_share_almost_no_divisor() {
    assert_eq!(contraction_part_width(&[1021, 4096]), None);
}

#[test]
fn should_accept_the_shallowest_sensible_part() {
    assert_eq!(contraction_part_width(&[64, 128]), Some(64));
}

#[test]
fn should_refuse_a_part_just_below_the_shallowest_sensible_one() {
    assert_eq!(contraction_part_width(&[63]), None);
}

#[test]
fn should_handle_a_depth_with_no_even_divisor() {
    assert_eq!(contraction_part_width(&[1021]), Some(1021));
}

#[test]
fn should_refuse_a_zero_depth() {
    assert_eq!(contraction_part_width(&[4096, 0]), None);
}

#[test]
fn should_refuse_no_depths_at_all() {
    assert_eq!(contraction_part_width(&[]), None);
}
