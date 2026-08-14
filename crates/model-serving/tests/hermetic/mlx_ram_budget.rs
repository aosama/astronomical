use astronomical_model_serving::{
    BOOTSTRAP_CONTEXT_WINDOW_RESERVE_BYTES, MlxRamBudget, MlxRamBudgetMeasurement,
    MlxRamBudgetModelGeometry, MlxRamBudgetPhase,
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
}

#[test]
fn should_compose_retained_expert_budget_from_ceiling_minus_fixed_owners() {
    let mlx_ram_budget = MlxRamBudget::new(39_000_000_000, fable_class_geometry())
        .expect("positive ceiling should construct");

    let planned_budget = mlx_ram_budget.plan(MlxRamBudgetPhase::Prefill, 4_096, 0, true);

    // retained_expert_budget =
    //   mlx_active_memory_ceiling
    //   - model_core
    //   - context_window_reserve
    //   - activation_headroom
    //   - complete_layer_stream_slot
    assert_eq!(planned_budget.context_window_reserve_bytes, 1_000_000_000);
    assert_eq!(planned_budget.complete_layer_stream_slot_bytes, 905_969_664);
    assert_eq!(
        planned_budget.retained_expert_budget_bytes,
        39_000_000_000 - 2_360_000_000 - 1_000_000_000 - 905_969_664
    );
    assert!(planned_budget.must_stream_operation_local);
    assert!(!planned_budget.may_grow_retained_expert_layers);
    assert!(!planned_budget.complete_residency_fits);
}

#[test]
fn should_require_operation_local_streaming_for_multi_token_prefill() {
    let mlx_ram_budget = MlxRamBudget::new(39_000_000_000, fable_class_geometry())
        .expect("positive ceiling should construct");

    let multi_token_prefill = mlx_ram_budget.plan(MlxRamBudgetPhase::Prefill, 2_048, 0, true);
    let decode = mlx_ram_budget.plan(MlxRamBudgetPhase::Decode, 2_048, 0, false);

    assert!(multi_token_prefill.must_stream_operation_local);
    assert!(!multi_token_prefill.may_grow_retained_expert_layers);
    // Decode may retain only when leftover retained-expert budget covers a complete layer.
    assert_eq!(
        decode.may_grow_retained_expert_layers,
        decode.retained_expert_budget_bytes
            >= fable_class_geometry().largest_complete_expert_layer_bytes
            && !decode.must_stream_operation_local
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

    let planned_budget = mlx_ram_budget.plan(MlxRamBudgetPhase::Prefill, 4_096, 0, true);
    assert_eq!(planned_budget.activation_headroom_bytes, 700_000_000);
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

    let planned_budget = mlx_ram_budget.plan(MlxRamBudgetPhase::Prefill, 7_000, 0, true);

    assert_eq!(planned_budget.context_window_reserve_bytes, 1_000_000_000);
    assert_eq!(planned_budget.activation_headroom_bytes, 2_000_000_000);
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

    let first_decode_plan = mlx_ram_budget.plan(MlxRamBudgetPhase::Decode, 7_000, 0, false);

    // Before decode has its own evidence, retaining experts into prefill-proven
    // transient space would force immediate reclamation on the first token.
    assert_eq!(first_decode_plan.activation_headroom_bytes, 1_900_000_000);
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

    let learned_decode_plan = mlx_ram_budget.plan(MlxRamBudgetPhase::Decode, 7_000, 0, false);

    assert_eq!(learned_decode_plan.activation_headroom_bytes, 400_000_000);
}

#[test]
fn should_compute_reclamation_when_retained_experts_exceed_budget() {
    let mlx_ram_budget = MlxRamBudget::new(39_000_000_000, fable_class_geometry())
        .expect("positive ceiling should construct");
    let planned_budget = mlx_ram_budget.plan(MlxRamBudgetPhase::Prefill, 4_096, 0, true);
    let retained_expert_payload_bytes = planned_budget.retained_expert_budget_bytes + 3_000_000_000;

    assert_eq!(
        mlx_ram_budget.expert_reclamation_target_bytes(
            MlxRamBudgetPhase::Prefill,
            4_096,
            0,
            true,
            retained_expert_payload_bytes,
        ),
        3_000_000_000
    );
}

#[test]
fn should_allow_complete_residency_only_when_core_experts_and_headroom_fit() {
    let small_geometry = MlxRamBudgetModelGeometry {
        model_core_payload_bytes: 500_000_000,
        complete_expert_payload_bytes: 2_000_000_000,
        largest_complete_expert_layer_bytes: 200_000_000,
        largest_routed_expert_page_bytes: 20_000_000,
    };
    let mlx_ram_budget = MlxRamBudget::new(8_000_000_000, small_geometry)
        .expect("positive ceiling should construct");

    let planned_budget = mlx_ram_budget.plan(MlxRamBudgetPhase::Idle, 0, 0, false);
    assert!(planned_budget.complete_residency_fits);

    let tight_budget = MlxRamBudget::new(2_400_000_000, small_geometry)
        .expect("positive ceiling should construct");
    assert!(
        !tight_budget
            .plan(MlxRamBudgetPhase::Idle, 0, 0, false)
            .complete_residency_fits
    );
}
