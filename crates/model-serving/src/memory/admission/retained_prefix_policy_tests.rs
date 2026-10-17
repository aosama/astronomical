use crate::{RetainedPrefixRetentionDecision, RetainedPrefixRetentionRequirements};

#[test]
fn should_retain_an_adoptable_state_when_adoption_is_enabled() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let decision = RetainedPrefixRetentionRequirements {
        adoption_enabled: true,
        adoptable_token_count: 20_480,
        retained_state_payload_bytes: 4_000_000_000,
        effective_mlx_memory_ceiling_bytes: 36_000_000_000,
    }
    .decide();
    assert!(
        matches!(decision, RetainedPrefixRetentionDecision::Retain),
        "an adoptable, modestly sized state must be retained for the next turn"
    );
}

#[test]
fn should_retain_a_state_of_exactly_half_the_ceiling() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let decision = RetainedPrefixRetentionRequirements {
        adoption_enabled: true,
        adoptable_token_count: 98_304,
        retained_state_payload_bytes: 18_000_000_000,
        effective_mlx_memory_ceiling_bytes: 36_000_000_000,
    }
    .decide();
    assert!(
        matches!(decision, RetainedPrefixRetentionDecision::Retain),
        "exactly half the ceiling is the documented retention boundary and must retain"
    );
}

#[test]
fn should_drop_the_retained_state_when_adoption_is_disabled() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let decision = RetainedPrefixRetentionRequirements {
        adoption_enabled: false,
        adoptable_token_count: 20_480,
        retained_state_payload_bytes: 4_000_000_000,
        effective_mlx_memory_ceiling_bytes: 36_000_000_000,
    }
    .decide();
    assert!(
        matches!(decision, RetainedPrefixRetentionDecision::Drop),
        "the opt-out must release the live state immediately"
    );
}

#[test]
fn should_drop_a_state_with_nothing_adoptable() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let decision = RetainedPrefixRetentionRequirements {
        adoption_enabled: true,
        adoptable_token_count: 0,
        retained_state_payload_bytes: 4_000_000_000,
        effective_mlx_memory_ceiling_bytes: 36_000_000_000,
    }
    .decide();
    assert!(
        matches!(decision, RetainedPrefixRetentionDecision::Drop),
        "a generation that published no complete block has no adoptable prefix"
    );
}

#[test]
fn should_drop_a_state_that_would_pin_more_than_half_the_ceiling() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let decision = RetainedPrefixRetentionRequirements {
        adoption_enabled: true,
        adoptable_token_count: 196_352,
        retained_state_payload_bytes: 19_000_000_000,
        effective_mlx_memory_ceiling_bytes: 36_000_000_000,
    }
    .decide();
    assert!(
        matches!(decision, RetainedPrefixRetentionDecision::Drop),
        "a retained state above half the ceiling would starve an unrelated next conversation"
    );
}

#[test]
fn should_not_wrap_when_the_state_payload_doubles_past_the_word() {
    astronomical_test_harness::ensure_network_denied()
        .expect("network denial could not be applied");

    let decision = RetainedPrefixRetentionRequirements {
        adoption_enabled: true,
        adoptable_token_count: 98_304,
        retained_state_payload_bytes: u64::MAX / 2 + 1,
        effective_mlx_memory_ceiling_bytes: 36_000_000_000,
    }
    .decide();
    assert!(
        matches!(decision, RetainedPrefixRetentionDecision::Drop),
        "a state larger than half of any representable ceiling must drop, not wrap to a retain"
    );
}
