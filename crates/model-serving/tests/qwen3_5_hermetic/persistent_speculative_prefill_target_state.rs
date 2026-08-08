use astronomical_model_serving::{
    PersistentSpeculativePrefillTargetStateContract,
    longest_reusable_speculative_prefill_target_prefix,
};

fn target_state_contract() -> PersistentSpeculativePrefillTargetStateContract {
    PersistentSpeculativePrefillTargetStateContract::new(
        "target-model".to_owned(),
        "target-revision".to_owned(),
        "drafter-model".to_owned(),
        "drafter-revision".to_owned(),
        [7_u8; 32],
        20,
        32,
        512,
        8,
        13,
    )
}

#[test]
fn should_restore_the_longest_selection_bound_sparse_target_prefix_for_a_follow_up() {
    let initial_prompt_token_ids = (0..40_960)
        .map(|token_position| token_position as u32)
        .collect::<Vec<_>>();
    let mut follow_up_prompt_token_ids = initial_prompt_token_ids.clone();
    follow_up_prompt_token_ids
        .extend((0..10_240).map(|token_position| 100_000_u32 + token_position as u32));
    let ordered_image_sha256_digests = [[3_u8; 32], [5_u8; 32]];
    let persisted_target_state_identity = target_state_contract()
        .target_state_identity_hash(&initial_prompt_token_ids, &ordered_image_sha256_digests);

    let restored_prefix_token_count = longest_reusable_speculative_prefill_target_prefix(
        &target_state_contract(),
        &follow_up_prompt_token_ids,
        &ordered_image_sha256_digests,
        |candidate_target_state_identity| {
            candidate_target_state_identity == persisted_target_state_identity
        },
    );

    assert_eq!(restored_prefix_token_count, Some(40_960));
}

#[test]
fn should_never_restore_sparse_target_state_for_different_prompt_or_images() {
    let initial_prompt_token_ids = (0..8_192)
        .map(|token_position| token_position as u32)
        .collect::<Vec<_>>();
    let ordered_image_sha256_digests = [[3_u8; 32]];
    let persisted_target_state_identity = target_state_contract()
        .target_state_identity_hash(&initial_prompt_token_ids, &ordered_image_sha256_digests);
    let mut changed_prompt_token_ids = initial_prompt_token_ids.clone();
    changed_prompt_token_ids[4_096] = 999_999;

    assert_eq!(
        longest_reusable_speculative_prefill_target_prefix(
            &target_state_contract(),
            &changed_prompt_token_ids,
            &ordered_image_sha256_digests,
            |candidate_target_state_identity| {
                candidate_target_state_identity == persisted_target_state_identity
            },
        ),
        None
    );
    assert_eq!(
        longest_reusable_speculative_prefill_target_prefix(
            &target_state_contract(),
            &initial_prompt_token_ids,
            &[[9_u8; 32]],
            |candidate_target_state_identity| {
                candidate_target_state_identity == persisted_target_state_identity
            },
        ),
        None
    );
}

#[test]
fn should_invalidate_sparse_target_state_when_target_drafter_or_selection_policy_changes() {
    let prompt_token_ids = (0..8_192)
        .map(|token_position| token_position as u32)
        .collect::<Vec<_>>();
    let baseline_identity =
        target_state_contract().target_state_identity_hash(&prompt_token_ids, &[]);
    let changed_target_identity = PersistentSpeculativePrefillTargetStateContract::new(
        "different-target".to_owned(),
        "target-revision".to_owned(),
        "drafter-model".to_owned(),
        "drafter-revision".to_owned(),
        [7_u8; 32],
        20,
        32,
        512,
        8,
        13,
    )
    .target_state_identity_hash(&prompt_token_ids, &[]);
    let changed_policy_identity = PersistentSpeculativePrefillTargetStateContract::new(
        "target-model".to_owned(),
        "target-revision".to_owned(),
        "drafter-model".to_owned(),
        "drafter-revision".to_owned(),
        [7_u8; 32],
        40,
        32,
        512,
        8,
        13,
    )
    .target_state_identity_hash(&prompt_token_ids, &[]);

    assert_ne!(baseline_identity, changed_target_identity);
    assert_ne!(baseline_identity, changed_policy_identity);
}
