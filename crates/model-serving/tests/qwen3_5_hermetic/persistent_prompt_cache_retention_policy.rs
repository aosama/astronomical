use astronomical_model_serving::{
    PersistentPromptCacheBlockSaveAdmission, persistent_prompt_cache_save_admission,
};

#[test]
fn should_retain_parent_boundary_when_the_child_fits_without_pressure() {
    let save_admission =
        persistent_prompt_cache_save_admission(200, 100, 300, 300, 1_000, false, false);

    assert_eq!(
        save_admission,
        PersistentPromptCacheBlockSaveAdmission::SaveWithoutEviction
    );
    assert!(!save_admission.should_reclaim_parent_boundary());
}

#[test]
fn should_reclaim_the_parent_boundary_only_when_its_bytes_are_required() {
    let save_admission =
        persistent_prompt_cache_save_admission(800, 100, 300, 300, 1_000, false, false);

    assert_eq!(
        save_admission,
        PersistentPromptCacheBlockSaveAdmission::SaveAndReclaimParentBoundary
    );
    assert!(save_admission.should_reclaim_parent_boundary());
}

#[test]
fn should_reclaim_the_parent_then_evict_unrelated_files_under_greater_pressure() {
    let save_admission =
        persistent_prompt_cache_save_admission(980, 200, 300, 100, 1_000, false, false);

    assert_eq!(
        save_admission,
        PersistentPromptCacheBlockSaveAdmission::SaveReclaimParentAndEvictOldBlocksToFit
    );
}

#[test]
fn should_skip_one_capture_that_cannot_fit_an_empty_quota() {
    let save_admission =
        persistent_prompt_cache_save_admission(0, 600, 500, 0, 1_000, false, false);

    assert_eq!(
        save_admission,
        PersistentPromptCacheBlockSaveAdmission::SkipBecauseCacheIsFull
    );
}

#[test]
fn should_charge_only_state_kinds_that_are_not_already_tracked() {
    let save_admission =
        persistent_prompt_cache_save_admission(950, 400, 40, 0, 1_000, true, false);

    assert_eq!(
        save_admission,
        PersistentPromptCacheBlockSaveAdmission::SaveWithoutEviction
    );
}
