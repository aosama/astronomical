use astronomical_model_serving::{
    BOOTSTRAP_CONTEXT_WINDOW_RESERVE_BYTES, MlxRamBudget, MlxRamBudgetMeasurement,
    MlxRamBudgetModelGeometry, MlxRamBudgetPhase, measured_non_expert_forward_growth_bytes,
};

fn fable_class_geometry() -> MlxRamBudgetModelGeometry {
    MlxRamBudgetModelGeometry {
        model_core_payload_bytes: 2_360_000_000,
        complete_expert_payload_bytes: 36_238_786_560,
        largest_complete_expert_layer_bytes: 905_969_664,
        largest_routed_expert_page_bytes: 28_311_552,
    }
}

#[test]
fn should_bootstrap_context_window_reserve_at_one_gigabyte_before_measurements() {
    let mlx_ram_budget = MlxRamBudget::new(39_000_000_000, fable_class_geometry())
        .expect("positive ceiling should construct");

    assert_eq!(BOOTSTRAP_CONTEXT_WINDOW_RESERVE_BYTES, 1_000_000_000);
    assert_eq!(
        mlx_ram_budget.context_window_reserve_bytes(0),
        1_000_000_000
    );
    assert_eq!(
        mlx_ram_budget.context_window_reserve_bytes(4_096),
        1_000_000_000
    );
    assert!(!mlx_ram_budget.has_context_window_measurement());
    assert_eq!(
        mlx_ram_budget.activation_headroom_bytes(MlxRamBudgetPhase::Prefill),
        3_623_878_656,
    );
}

#[test]
fn should_compose_retained_expert_budget_from_ceiling_minus_fixed_owners() {
    let mlx_ram_budget = MlxRamBudget::new(39_000_000_000, fable_class_geometry())
        .expect("positive ceiling should construct");

    let planned_budget = mlx_ram_budget.plan(MlxRamBudgetPhase::Prefill, 4_096, 0);

    // retained_expert_budget =
    //   mlx_active_memory_ceiling
    //   - model_core
    //   - context_window_reserve
    //   - activation_headroom
    //   - complete_layer_stream_slot
    assert_eq!(planned_budget.context_window_reserve_bytes, 1_000_000_000);
    assert_eq!(planned_budget.activation_headroom_bytes, 3_623_878_656);
    assert_eq!(planned_budget.complete_layer_stream_slot_bytes, 905_969_664);
    assert_eq!(
        planned_budget.retained_expert_budget_bytes,
        39_000_000_000 - 2_360_000_000 - 1_000_000_000 - 3_623_878_656 - 905_969_664
    );
}

#[test]
fn should_raise_context_window_reserve_from_measurements_and_never_under_shoot() {
    let mut mlx_ram_budget = MlxRamBudget::new(39_000_000_000, fable_class_geometry())
        .expect("positive ceiling should construct");

    mlx_ram_budget.record_measurement(MlxRamBudgetMeasurement {
        phase: MlxRamBudgetPhase::Prefill,
        context_token_count: 2_048,
        measured_context_and_activation_bytes: 1_500_000_000,
        observed_activation_headroom_bytes: 400_000_000,
        exact_temporary_workspace_bytes: 0,
    });
    mlx_ram_budget.record_measurement(MlxRamBudgetMeasurement {
        phase: MlxRamBudgetPhase::Prefill,
        context_token_count: 4_096,
        measured_context_and_activation_bytes: 2_200_000_000,
        observed_activation_headroom_bytes: 700_000_000,
        exact_temporary_workspace_bytes: 0,
    });

    assert!(mlx_ram_budget.has_context_window_measurement());
    let context_window_reserve_for_2048 = mlx_ram_budget.context_window_reserve_bytes(2_048);
    let context_window_reserve_for_4096 = mlx_ram_budget.context_window_reserve_bytes(4_096);
    assert!(context_window_reserve_for_2048 >= 1_100_000_000);
    assert!(context_window_reserve_for_4096 >= context_window_reserve_for_2048);
    assert!(context_window_reserve_for_4096 >= 1_500_000_000);

    let planned_budget = mlx_ram_budget.plan(MlxRamBudgetPhase::Prefill, 4_096, 0);
    assert_eq!(planned_budget.activation_headroom_bytes, 3_623_878_656);
    // Experts must not be budgeted into the learned context-window / activation reserve.
    let fixed_non_expert_bytes = planned_budget.model_core_payload_bytes
        + planned_budget.context_window_reserve_bytes
        + planned_budget.activation_headroom_bytes
        + planned_budget.complete_layer_stream_slot_bytes;
    assert_eq!(
        planned_budget.retained_expert_budget_bytes,
        planned_budget
            .mlx_active_memory_ceiling_bytes
            .saturating_sub(fixed_non_expert_bytes)
    );
}

#[test]
fn should_not_charge_transient_workspace_as_both_context_and_activation() {
    let mut mlx_ram_budget = MlxRamBudget::new(23_000_000_000, fable_class_geometry())
        .expect("positive ceiling should construct");
    mlx_ram_budget.record_measurement(MlxRamBudgetMeasurement {
        phase: MlxRamBudgetPhase::Prefill,
        context_token_count: 7_000,
        measured_context_and_activation_bytes: 2_100_000_000,
        observed_activation_headroom_bytes: 2_000_000_000,
        exact_temporary_workspace_bytes: 100_000_000,
    });

    let planned_budget = mlx_ram_budget.plan(MlxRamBudgetPhase::Prefill, 7_000, 0);

    assert_eq!(planned_budget.context_window_reserve_bytes, 1_000_000_000);
    assert_eq!(planned_budget.activation_headroom_bytes, 3_623_878_656);
}

#[test]
fn should_not_charge_exact_temporary_workspace_as_persistent_context() {
    let mut mlx_ram_budget = MlxRamBudget::new(23_000_000_000, fable_class_geometry())
        .expect("positive ceiling should construct");
    mlx_ram_budget.record_measurement(MlxRamBudgetMeasurement {
        phase: MlxRamBudgetPhase::Prefill,
        context_token_count: 7_000,
        measured_context_and_activation_bytes: 3_000_000_000,
        observed_activation_headroom_bytes: 500_000_000,
        exact_temporary_workspace_bytes: 500_000_000,
    });

    assert_eq!(
        mlx_ram_budget.context_window_reserve_bytes(7_000),
        2_064_000_000
    );
}

#[test]
fn should_protect_the_first_decode_with_prefill_activation_evidence() {
    let mut mlx_ram_budget = MlxRamBudget::new(23_000_000_000, fable_class_geometry())
        .expect("positive ceiling should construct");
    mlx_ram_budget.record_measurement(MlxRamBudgetMeasurement {
        phase: MlxRamBudgetPhase::Prefill,
        context_token_count: 7_000,
        measured_context_and_activation_bytes: 2_000_000_000,
        observed_activation_headroom_bytes: 1_900_000_000,
        exact_temporary_workspace_bytes: 0,
    });

    let first_decode_plan = mlx_ram_budget.plan(MlxRamBudgetPhase::Decode, 7_000, 0);

    // Before decode has its own evidence, retaining experts into prefill-proven
    // transient space would force immediate reclamation on the first token.
    assert_eq!(first_decode_plan.activation_headroom_bytes, 3_623_878_656);
}

#[test]
fn should_use_decode_activation_evidence_after_decode_is_observed() {
    let mut mlx_ram_budget = MlxRamBudget::new(23_000_000_000, fable_class_geometry())
        .expect("positive ceiling should construct");
    mlx_ram_budget.record_measurement(MlxRamBudgetMeasurement {
        phase: MlxRamBudgetPhase::Prefill,
        context_token_count: 7_000,
        measured_context_and_activation_bytes: 2_000_000_000,
        observed_activation_headroom_bytes: 1_900_000_000,
        exact_temporary_workspace_bytes: 0,
    });
    mlx_ram_budget.record_measurement(MlxRamBudgetMeasurement {
        phase: MlxRamBudgetPhase::Decode,
        context_token_count: 7_000,
        measured_context_and_activation_bytes: 500_000_000,
        observed_activation_headroom_bytes: 400_000_000,
        exact_temporary_workspace_bytes: 0,
    });

    let learned_decode_plan = mlx_ram_budget.plan(MlxRamBudgetPhase::Decode, 7_000, 0);

    assert_eq!(learned_decode_plan.activation_headroom_bytes, 400_000_000);
}

#[test]
fn should_limit_retained_experts_to_leave_the_exact_admitted_prefill_reserve() {
    let mlx_ram_budget = MlxRamBudget::new(38_000_000_000, fable_class_geometry())
        .expect("positive ceiling should construct");
    let current_active_memory_bytes = 35_879_800_070;
    let current_retained_expert_payload_bytes = 33_520_877_568;
    // Production evidence showed 1.554 GB of context/activation growth followed
    // by one exact 905,969,664-byte complete-layer allocation.
    let admitted_forward_reserve_bytes = 2_460_246_024;

    let retained_expert_budget_bytes = mlx_ram_budget.retained_expert_budget_for_admitted_forward(
        current_active_memory_bytes,
        current_retained_expert_payload_bytes,
        admitted_forward_reserve_bytes,
    );

    assert_eq!(retained_expert_budget_bytes, 33_180_831_474);
    assert!(retained_expert_budget_bytes < current_retained_expert_payload_bytes);
    assert_eq!(
        current_active_memory_bytes
            .saturating_sub(current_retained_expert_payload_bytes)
            .saturating_add(retained_expert_budget_bytes)
            .saturating_add(admitted_forward_reserve_bytes),
        38_000_000_000,
    );
}

#[test]
fn should_exclude_newly_retained_experts_from_context_and_activation_learning() {
    assert_eq!(
        measured_non_expert_forward_growth_bytes(2_358_922_508, 36_867_500_000, 0, 33_520_877_568,),
        987_699_924,
    );
    assert_eq!(
        measured_non_expert_forward_growth_bytes(3_000, 4_500, 2_000, 2_000),
        1_500,
    );
}
