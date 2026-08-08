use astronomical_model_serving::PersistentSpeculativePrefillSelectionContract;

fn selection_contract(draft_model_id: &str) -> PersistentSpeculativePrefillSelectionContract {
    selection_contract_for_model_identity(
        "target-model",
        "target-revision",
        draft_model_id,
        "revision-a",
    )
}

fn selection_contract_for_model_identity(
    target_model_id: &str,
    target_model_revision: &str,
    draft_model_id: &str,
    draft_model_revision: &str,
) -> PersistentSpeculativePrefillSelectionContract {
    PersistentSpeculativePrefillSelectionContract::new(
        target_model_id.to_owned(),
        target_model_revision.to_owned(),
        draft_model_id.to_owned(),
        draft_model_revision.to_owned(),
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
fn should_purge_only_a_matching_target_drafter_pairing_with_an_obsolete_keep_percentage() {
    let stored_selection_policy = selection_contract("drafter-a").policy_identity();

    assert!(stored_selection_policy.should_purge_for_active_keep_percentage(
        "target-model",
        "target-revision",
        "drafter-a",
        "revision-a",
        40,
    ));
    assert!(!stored_selection_policy.should_purge_for_active_keep_percentage(
        "target-model",
        "target-revision",
        "drafter-a",
        "revision-a",
        20,
    ));
    assert!(!stored_selection_policy.should_purge_for_active_keep_percentage(
        "other-target",
        "target-revision",
        "drafter-a",
        "revision-a",
        40,
    ));
    assert!(!stored_selection_policy.should_purge_for_active_keep_percentage(
        "target-model",
        "target-revision",
        "other-drafter",
        "revision-a",
        40,
    ));
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
fn should_isolate_selection_identity_across_target_and_drafter_revisions() {
    let prompt_token_ids = (0..8_192)
        .map(|token_id| token_id as u32)
        .collect::<Vec<_>>();
    let baseline_identity = selection_contract_for_model_identity(
        "target-model",
        "target-revision-a",
        "drafter-model",
        "drafter-revision-a",
    )
    .selection_identity_hash(&prompt_token_ids);

    for changed_model_identity_contract in [
        selection_contract_for_model_identity(
            "different-target",
            "target-revision-a",
            "drafter-model",
            "drafter-revision-a",
        ),
        selection_contract_for_model_identity(
            "target-model",
            "target-revision-b",
            "drafter-model",
            "drafter-revision-a",
        ),
        selection_contract_for_model_identity(
            "target-model",
            "target-revision-a",
            "different-drafter",
            "drafter-revision-a",
        ),
        selection_contract_for_model_identity(
            "target-model",
            "target-revision-a",
            "drafter-model",
            "drafter-revision-b",
        ),
    ] {
        assert_ne!(
            baseline_identity,
            changed_model_identity_contract.selection_identity_hash(&prompt_token_ids),
        );
    }
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
        "target-model".to_owned(),
        "target-revision".to_owned(),
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
