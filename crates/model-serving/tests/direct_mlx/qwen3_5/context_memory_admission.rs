use astronomical_model_serving::{
    AdaptiveRamGrowthContext, AdaptiveRamGrowthGuard, ContextAdmissionRequirements,
    combined_persistent_growth_bytes, persistent_context_restore_workspace_bytes,
};

#[test]
fn should_charge_all_persistent_growth_in_one_admission_projection() {
    let target_persistent_state_growth_bytes = 10_485_760;
    let additional_full_attention_growth_bytes = 262_144;

    let combined_persistent_growth_bytes = combined_persistent_growth_bytes(
        target_persistent_state_growth_bytes,
        additional_full_attention_growth_bytes,
    );

    assert_eq!(
        combined_persistent_growth_bytes
            .expect("the combined growth should fit the platform byte range"),
        10_747_904
    );
}

#[test]
fn should_require_reclamation_only_when_additional_growth_is_added_to_fitting_target_growth() {
    let adaptive_ram_growth_guard = AdaptiveRamGrowthGuard::new(1_000)
        .expect("a positive active-memory limit should create a guard");
    let target_persistent_state_growth_bytes = 300;

    let target_only_projection = adaptive_ram_growth_guard
        .project_growth_for_context(
            AdaptiveRamGrowthContext::decode(
                1,
                astronomical_model_serving::AdaptiveRamGrowthExecutionProfile::Resident,
            ),
            700,
            target_persistent_state_growth_bytes,
            0,
            0,
        )
        .expect("the target-only projection should not overflow");
    let combined_growth_bytes =
        combined_persistent_growth_bytes(target_persistent_state_growth_bytes, 1)
            .expect("the combined growth should not overflow");
    let combined_projection = adaptive_ram_growth_guard
        .project_growth_for_context(
            AdaptiveRamGrowthContext::decode(
                1,
                astronomical_model_serving::AdaptiveRamGrowthExecutionProfile::Resident,
            ),
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
fn should_fail_closed_when_combined_growth_overflows() {
    assert_eq!(combined_persistent_growth_bytes(usize::MAX, 1), None);
}

#[test]
fn should_project_exact_context_growth_without_cross_context_transient_memory() {
    let active_memory_bytes_after_reclamation = 35_972_348_166;
    let context_reservation_bytes = 3_815_485_440;

    let projected_active_memory_bytes = ContextAdmissionRequirements {
        current_active_memory_bytes: active_memory_bytes_after_reclamation,
        context_growth_bytes: context_reservation_bytes,
        expert_page_reservation_bytes: 0,
        temporary_workspace_bytes: 0,
        retained_expert_payload_bytes: 0,
        active_memory_ceiling_bytes: usize::MAX,
        complete_experts_are_resident: false,
    }
    .projected_active_memory_bytes();

    assert_eq!(projected_active_memory_bytes, Some(39_787_833_606));
}

#[test]
fn should_size_the_restore_workspace_by_one_block_for_incremental_restore() {
    // The restore loop absorbs one block at a time into a preallocated
    // destination, so the temporary workspace is a single source block, not
    // the whole restored prefix. The destination is already inside
    // context_growth; charging the full prefix would stack a bulk-restore
    // paper peak and reclaim expert pages the incremental restore never held.
    let active_memory_bytes_after_reclamation = 23_137_777_924;
    let context_memory_reservation_bytes_per_token = 20_480;
    let total_context_tokens = 92_681;
    let restore_block_token_count = 128;
    let system_gpu_memory_limit_bytes = 25_769_803_776;

    let restore_workspace_bytes = persistent_context_restore_workspace_bytes(
        context_memory_reservation_bytes_per_token,
        restore_block_token_count,
    )
    .expect("the one-block restore workspace should fit usize");

    // One block is far smaller than the full restored prefix (71_680 tokens
    // would be 1_468_006_400 bytes); the incremental restore only ever holds
    // one source block beside the destination.
    assert_eq!(restore_workspace_bytes, 2_621_440);

    let context_reservation_bytes = context_memory_reservation_bytes_per_token
        .checked_mul(total_context_tokens)
        .expect("the context reservation should fit usize");
    let projection_with_restore_workspace = ContextAdmissionRequirements {
        current_active_memory_bytes: active_memory_bytes_after_reclamation,
        context_growth_bytes: context_reservation_bytes,
        expert_page_reservation_bytes: 0,
        temporary_workspace_bytes: restore_workspace_bytes,
        retained_expert_payload_bytes: 0,
        active_memory_ceiling_bytes: system_gpu_memory_limit_bytes,
        complete_experts_are_resident: false,
    }
    .projected_active_memory_bytes()
    .expect("the projection with the one-block restore workspace should fit usize");

    // The one-block workspace keeps the projection under the ceiling, so no
    // expert reclamation is required (the old full-prefix workspace pushed it
    // to 26_503_891_204, over the 25_769_803_776 limit).
    assert!(projection_with_restore_workspace <= system_gpu_memory_limit_bytes);
}
