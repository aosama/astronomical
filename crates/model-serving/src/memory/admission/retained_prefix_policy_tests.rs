use astronomical_test_harness::offline_test;

use crate::{RetainedPrefixRetentionDecision, RetainedPrefixRetentionRequirements};

offline_test!(should_retain_an_adoptable_state_when_adoption_is_enabled, {
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
});

offline_test!(should_drop_the_retained_state_when_adoption_is_disabled, {
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
});

offline_test!(should_drop_a_state_with_nothing_adoptable, {
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
});

offline_test!(
    should_drop_a_state_that_would_pin_more_than_half_the_ceiling,
    {
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
);
