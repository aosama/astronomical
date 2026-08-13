use astronomical_model_serving::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthGuard, InferenceEngineError,
    combined_target_and_additional_persistent_growth_bytes,
    context_memory_admission_projected_active_memory_bytes,
    persistent_prompt_cache_restore_temporary_workspace_bytes,
};

#[test]
fn should_charge_target_and_mtp_persistent_growth_in_one_admission_projection() {
    let target_persistent_state_growth_bytes = 10_485_760;
    let mtp_full_attention_growth_bytes = 262_144;

    let combined_persistent_growth_bytes = combined_target_and_additional_persistent_growth_bytes(
        target_persistent_state_growth_bytes,
        mtp_full_attention_growth_bytes,
    );

    assert_eq!(
        combined_persistent_growth_bytes
            .expect("the target and MTP growth should fit the platform byte range"),
        10_747_904
    );
}

#[test]
fn should_require_reclamation_only_when_mtp_growth_is_added_to_fitting_target_growth() {
    let adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(1_000)
        .expect("a positive active-memory limit should create a guard");
    let target_persistent_state_growth_bytes = 300;

    let target_only_projection = adaptive_ram_growth_guard
        .project_growth_for_context(
            AdaptiveRamGrowthContext::decode(1, false, false),
            700,
            target_persistent_state_growth_bytes,
            0,
            0,
        )
        .expect("the target-only projection should not overflow");
    let combined_growth_bytes = combined_target_and_additional_persistent_growth_bytes(
        target_persistent_state_growth_bytes,
        1,
    )
    .expect("the combined growth should not overflow");
    let combined_projection = adaptive_ram_growth_guard
        .project_growth_for_context(
            AdaptiveRamGrowthContext::decode(1, false, false),
            700,
            combined_growth_bytes,
            0,
            0,
        )
        .expect("the combined projection should not overflow");

    assert!(target_only_projection.fits_stable_and_peak_limits());
    assert_eq!(
        combined_projection.operation_reclamation_required_bytes(),
        1
    );
    assert!(!combined_projection.fits_stable_and_peak_limits());
}

#[test]
fn should_reject_target_and_mtp_growth_overflow_with_a_typed_invalid_request() {
    let growth_rejection = combined_target_and_additional_persistent_growth_bytes(usize::MAX, 1)
        .expect_err("overflowing target and MTP growth must be rejected");

    assert!(matches!(
        growth_rejection,
        InferenceEngineError::InvalidRequest { reason }
            if reason == "target and additional persistent growth overflowed"
    ));
}

#[test]
fn should_project_exact_context_growth_without_cross_context_transient_memory() {
    let active_memory_bytes_after_reclamation = 35_972_348_166;
    let context_reservation_bytes = 3_815_485_440;

    let projected_active_memory_bytes = context_memory_admission_projected_active_memory_bytes(
        active_memory_bytes_after_reclamation,
        context_reservation_bytes,
        0,
    );

    assert_eq!(projected_active_memory_bytes, Some(39_787_833_606));
}

#[test]
fn should_reserve_the_loaded_kv_prefix_while_reconstructing_persistent_prompt_cache_state() {
    let active_memory_bytes_after_reclamation = 23_137_777_924;
    let context_memory_reservation_bytes_per_token = 20_480;
    let total_context_tokens = 92_681;
    let restored_persistent_prompt_cache_token_count = 71_680;
    let system_gpu_memory_limit_bytes = 25_769_803_776;

    let persistent_prompt_cache_restore_temporary_workspace_bytes =
        persistent_prompt_cache_restore_temporary_workspace_bytes(
            context_memory_reservation_bytes_per_token,
            restored_persistent_prompt_cache_token_count,
        );

    assert_eq!(
        persistent_prompt_cache_restore_temporary_workspace_bytes,
        Some(1_468_006_400)
    );
    let context_reservation_bytes = context_memory_reservation_bytes_per_token
        .checked_mul(total_context_tokens)
        .expect("the reproduced context reservation should fit usize");
    let projection_without_restore_workspace =
        context_memory_admission_projected_active_memory_bytes(
            active_memory_bytes_after_reclamation,
            context_reservation_bytes,
            0,
        )
        .expect("the reproduced pre-restore projection should fit usize");
    let projection_with_restore_workspace = projection_without_restore_workspace
        .checked_add(
            persistent_prompt_cache_restore_temporary_workspace_bytes
                .expect("the reproduced restore workspace should fit usize"),
        )
        .expect("the reproduced complete projection should fit usize");

    assert!(projection_without_restore_workspace <= system_gpu_memory_limit_bytes);
    assert!(projection_with_restore_workspace > system_gpu_memory_limit_bytes);
    assert_eq!(projection_with_restore_workspace, 26_503_891_204);

    let active_memory_after_restored_state_replaces_loaded_blocks =
        active_memory_bytes_after_reclamation
            .checked_add(
                persistent_prompt_cache_restore_temporary_workspace_bytes
                    .expect("the restored state payload should fit usize"),
            )
            .expect("the post-restore active memory should fit usize");
    let remaining_context_token_count = total_context_tokens
        .checked_sub(restored_persistent_prompt_cache_token_count)
        .expect("the restored prefix should fit the total context");
    let remaining_context_reservation_bytes = context_memory_reservation_bytes_per_token
        .checked_mul(remaining_context_token_count)
        .expect("the remaining context reservation should fit usize");
    let post_restore_projection = context_memory_admission_projected_active_memory_bytes(
        active_memory_after_restored_state_replaces_loaded_blocks,
        remaining_context_reservation_bytes,
        0,
    )
    .expect("the post-restore projection should fit usize");

    assert_eq!(
        post_restore_projection,
        projection_without_restore_workspace
    );
    assert!(post_restore_projection <= system_gpu_memory_limit_bytes);
}
