use astronomical_model_serving::PersistentSpeculativePrefillSelectionContract;

fn selection_contract(draft_model_id: &str) -> PersistentSpeculativePrefillSelectionContract {
    PersistentSpeculativePrefillSelectionContract::new(
        draft_model_id.to_owned(),
        "revision-a".to_owned(),
        [7_u8; 32],
        20,
        32,
        512,
        8,
        13,
        2_048,
        8_192,
    )
}

#[test]
fn should_isolate_selection_identity_when_drafter_model_changes() {
    let prompt_token_ids = (0..8_192)
        .map(|token_id| token_id as u32)
        .collect::<Vec<_>>();

    let drafter_a_identity =
        selection_contract("drafter-a").selection_identity_hash(&prompt_token_ids);
    let drafter_b_identity =
        selection_contract("drafter-b").selection_identity_hash(&prompt_token_ids);

    assert_ne!(drafter_a_identity, drafter_b_identity);
}

#[test]
fn should_invalidate_selection_identity_when_prompt_or_policy_changes() {
    let prompt_token_ids = (0..8_192)
        .map(|token_id| token_id as u32)
        .collect::<Vec<_>>();
    let mut modified_prompt_token_ids = prompt_token_ids.clone();
    modified_prompt_token_ids[32] = 99_999;

    let baseline_identity =
        selection_contract("drafter-a").selection_identity_hash(&prompt_token_ids);
    let modified_prompt_identity =
        selection_contract("drafter-a").selection_identity_hash(&modified_prompt_token_ids);
    let modified_policy_identity = PersistentSpeculativePrefillSelectionContract::new(
        "drafter-a".to_owned(),
        "revision-a".to_owned(),
        [7_u8; 32],
        40,
        32,
        512,
        8,
        13,
        2_048,
        8_192,
    )
    .selection_identity_hash(&prompt_token_ids);

    assert_ne!(baseline_identity, modified_prompt_identity);
    assert_ne!(baseline_identity, modified_policy_identity);
}
