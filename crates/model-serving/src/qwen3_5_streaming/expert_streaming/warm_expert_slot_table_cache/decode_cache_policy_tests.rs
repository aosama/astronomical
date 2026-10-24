use super::{
    DecodeExpertSlotState, MAXIMUM_CANDIDATE_SLOT_REPLACEMENTS_PER_DECODE_STEP,
    MAXIMUM_PROBATIONARY_EXPERT_SLOTS_PER_LAYER, available_probationary_slot_count,
};

#[test]
fn should_admit_one_full_routed_set_and_bound_replacements_per_decode_step() {
    assert_eq!(MAXIMUM_PROBATIONARY_EXPERT_SLOTS_PER_LAYER, 8);
    assert_eq!(MAXIMUM_CANDIDATE_SLOT_REPLACEMENTS_PER_DECODE_STEP, 2);
}

#[test]
fn should_limit_live_probationary_experts_to_the_routed_set_per_layer() {
    let mut decode_policy_states = Vec::new();
    for _ in 0..MAXIMUM_PROBATIONARY_EXPERT_SLOTS_PER_LAYER {
        let mut decode_policy_state = DecodeExpertSlotState::default();
        decode_policy_state.mark_probationary(3);
        decode_policy_states.push(decode_policy_state);
    }

    assert_eq!(
        available_probationary_slot_count(&decode_policy_states, 3),
        0
    );
    assert_eq!(
        available_probationary_slot_count(&decode_policy_states[..7], 3),
        1
    );
}

#[test]
fn should_admit_a_historically_hot_expert_without_probation() {
    let mut decode_policy_state = DecodeExpertSlotState::default();
    decode_policy_state.mark_historically_hot(10);

    assert!(!decode_policy_state.is_probationary(10));
    assert!(
        !decode_policy_state.is_evictable(11),
        "a history-backed admission must survive immediate victim selection"
    );
}

#[test]
fn should_promote_a_probationary_expert_reused_at_the_window_boundary() {
    let mut decode_policy_state = DecodeExpertSlotState::default();
    decode_policy_state.mark_probationary(10);

    decode_policy_state.record_decode_reuse(14);

    assert!(!decode_policy_state.is_probationary(14));
    assert!(!decode_policy_state.is_evictable(14));
}

#[test]
fn should_expire_an_unreused_probationary_expert_after_its_window() {
    let mut decode_policy_state = DecodeExpertSlotState::default();
    decode_policy_state.mark_probationary(10);

    decode_policy_state.expire_probationary_admission(14);
    assert!(decode_policy_state.is_probationary(14));

    decode_policy_state.expire_probationary_admission(15);

    assert!(!decode_policy_state.is_probationary(15));
    assert!(decode_policy_state.is_evictable(15));
}

#[test]
fn should_keep_a_recently_reused_expert_out_of_victim_selection() {
    let mut decode_policy_state = DecodeExpertSlotState::default();
    decode_policy_state.record_decode_reuse(10);

    assert!(
        !decode_policy_state.is_evictable(11),
        "an entry with any positive decayed score survives probationary churn"
    );
    assert!(!decode_policy_state.is_evictable(15));
}

#[test]
fn should_admit_a_fully_decayed_entry_into_victim_selection() {
    let mut decode_policy_state = DecodeExpertSlotState::default();
    decode_policy_state.record_decode_reuse(10);

    // Eight half-lives of idleness (epoch 10 + 64) decay score 1 to zero.
    assert!(decode_policy_state.is_evictable(74));
}

#[test]
fn should_rank_a_colder_expert_before_a_more_recently_reused_expert() {
    let mut older_decode_policy_state = DecodeExpertSlotState::default();
    let mut recent_decode_policy_state = DecodeExpertSlotState::default();
    older_decode_policy_state.record_decode_reuse(0);
    recent_decode_policy_state.record_decode_reuse(8);

    assert!(
        older_decode_policy_state.compare_eviction_priority(
            0,
            recent_decode_policy_state,
            1,
            9,
            48,
            48,
        ) == std::cmp::Ordering::Less
    );
}

#[test]
fn should_rank_lower_reuse_per_byte_before_higher_raw_reuse() {
    let mut lower_value_decode_policy_state = DecodeExpertSlotState::default();
    let mut higher_value_decode_policy_state = DecodeExpertSlotState::default();
    lower_value_decode_policy_state.record_decode_reuse(0);
    higher_value_decode_policy_state.record_decode_reuse(0);
    higher_value_decode_policy_state.record_decode_reuse(0);

    assert_eq!(
        lower_value_decode_policy_state.compare_eviction_priority(
            1,
            higher_value_decode_policy_state,
            0,
            0,
            8,
            4,
        ),
        std::cmp::Ordering::Less
    );
}
