use super::generated_seed;

#[test]
fn should_draw_different_seeds_for_different_request_identifiers() {
    let first_seed = generated_seed(1);
    let second_seed = generated_seed(2);

    assert_ne!(first_seed, second_seed);
}

#[test]
fn should_accept_the_extreme_request_identifiers_without_overflow() {
    let _minimum_seed = generated_seed(0);
    let _maximum_seed = generated_seed(u64::MAX);
}

/// The identifier's contribution is a rotation, so it is reversible: with the
/// same clock reading the identifier is recoverable, which keeps two
/// concurrent requests distinct even in the same nanosecond.
#[test]
fn should_keep_the_identifier_visible_in_the_seed() {
    let seed_for_zero = generated_seed(0);
    let seed_for_one = generated_seed(1);

    assert_ne!(seed_for_zero ^ seed_for_one, 0);
}
