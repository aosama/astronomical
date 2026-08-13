use astronomical_model_serving::{
    BOOTSTRAP_CONTEXT_WINDOW_RESERVE_BYTES, MlxRamBudget, MlxRamBudgetMeasurement,
    MlxRamBudgetModelGeometry, MlxRamBudgetPhase,
};

fn fable_class_geometry() -> MlxRamBudgetModelGeometry {
    MlxRamBudgetModelGeometry {
        model_core_payload_bytes: 2_360_000_000,
        complete_expert_payload_bytes: 36_238_786_560,
        largest_complete_expert_layer_bytes: 905_969_664,
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
    });
    mlx_ram_budget.record_measurement(MlxRamBudgetMeasurement {
        phase: MlxRamBudgetPhase::Prefill,
        context_token_count: 4_096,
        measured_context_and_activation_bytes: 2_200_000_000,
        observed_activation_headroom_bytes: 700_000_000,
    });

    assert!(mlx_ram_budget.has_context_window_measurement());
    let context_window_reserve_for_2048 = mlx_ram_budget.context_window_reserve_bytes(2_048);
    let context_window_reserve_for_4096 = mlx_ram_budget.context_window_reserve_bytes(4_096);
    assert!(context_window_reserve_for_2048 >= 1_500_000_000);
    assert!(context_window_reserve_for_4096 >= context_window_reserve_for_2048);
    assert!(context_window_reserve_for_4096 >= 2_200_000_000);

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
fn should_count_how_many_complete_layers_fit_the_retained_expert_budget() {
    assert_eq!(
        MlxRamBudget::maximum_retained_complete_layer_count(0, 900_000_000),
        0
    );
    assert_eq!(
        MlxRamBudget::maximum_retained_complete_layer_count(2_700_000_000, 900_000_000),
        3
    );
    assert_eq!(
        MlxRamBudget::maximum_retained_complete_layer_count(2_699_999_999, 900_000_000),
        2
    );
    assert_eq!(
        MlxRamBudget::maximum_retained_complete_layer_count(10_000_000_000, 0),
        0
    );

    let mlx_ram_budget = MlxRamBudget::new(39_000_000_000, fable_class_geometry())
        .expect("positive ceiling should construct");
    let decode_plan = mlx_ram_budget.plan(MlxRamBudgetPhase::Decode, 4_096, 0, false);
    assert!(decode_plan.may_grow_retained_expert_layers);
    assert!(
        MlxRamBudget::maximum_retained_complete_layer_count(
            decode_plan.retained_expert_budget_bytes,
            fable_class_geometry().largest_complete_expert_layer_bytes,
        ) > 0
    );

    let multi_token_prefill = mlx_ram_budget.plan(MlxRamBudgetPhase::Prefill, 4_096, 0, true);
    assert!(!multi_token_prefill.may_grow_retained_expert_layers);
    // Prefill may not grow warm layers, but it must still publish a positive
    // keep-budget so existing warm layers are not zeroed without pressure.
    assert!(multi_token_prefill.retained_expert_budget_bytes > 8_000_000_000);

    // Decode-warm fill must use the composed budget rather than a fixed
    // machine-independent cap. The budget already reserves model core,
    // context, activations, and one complete-layer loading slot.
    let decode_warm_payload_budget_bytes = decode_plan.retained_expert_budget_bytes;
    let decode_warm_layer_count = MlxRamBudget::maximum_retained_complete_layer_count(
        decode_warm_payload_budget_bytes,
        fable_class_geometry().largest_complete_expert_layer_bytes,
    );
    assert!(decode_warm_layer_count > 0);
    assert_eq!(decode_warm_layer_count, 38);
    assert!(decode_warm_layer_count < 40);
}

#[test]
fn should_allow_complete_residency_only_when_core_experts_and_headroom_fit() {
    let small_geometry = MlxRamBudgetModelGeometry {
        model_core_payload_bytes: 500_000_000,
        complete_expert_payload_bytes: 2_000_000_000,
        largest_complete_expert_layer_bytes: 200_000_000,
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

#[test]
fn should_scale_progressive_expert_retention_by_completed_prompt_work() {
    assert_eq!(
        MlxRamBudget::progressive_retained_expert_payload_target_bytes(24_000_000_000, 0, 10_000,),
        0
    );
    assert_eq!(
        MlxRamBudget::progressive_retained_expert_payload_target_bytes(
            24_000_000_000,
            2_500,
            10_000,
        ),
        6_000_000_000
    );
    assert_eq!(
        MlxRamBudget::progressive_retained_expert_payload_target_bytes(
            24_000_000_000,
            10_000,
            10_000,
        ),
        24_000_000_000
    );
    assert_eq!(
        MlxRamBudget::progressive_retained_expert_payload_target_bytes(
            24_000_000_000,
            12_000,
            10_000,
        ),
        24_000_000_000
    );
    assert_eq!(
        MlxRamBudget::progressive_retained_expert_payload_target_bytes(24_000_000_000, 1, 0,),
        0
    );
}
