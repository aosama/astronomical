use astronomical_model_serving::Qwen3_5MoEPagedPrefillExecutionMode;

#[test]
fn should_resolve_multi_token_routes_before_executing_each_paged_layer() {
    let production_mode = Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault;

    for prefill_token_count in [2, 512, 1_024, 2_048, 4_096] {
        assert!(
            !production_mode.should_defer_host_route_materialization(prefill_token_count),
            "multi-token prefill must not execute with a holey expert snapshot"
        );
    }
}

#[test]
fn should_defer_only_the_one_token_production_decode_route() {
    assert!(
        Qwen3_5MoEPagedPrefillExecutionMode::ProductionDefault
            .should_defer_host_route_materialization(1)
    );
    assert!(
        !Qwen3_5MoEPagedPrefillExecutionMode::TargetVerificationWindow
            .should_defer_host_route_materialization(1)
    );
    assert!(
        !Qwen3_5MoEPagedPrefillExecutionMode::CompactPromptDiagnostic
            .should_defer_host_route_materialization(1)
    );
    assert!(
        !Qwen3_5MoEPagedPrefillExecutionMode::TokenLocalDiagnostic
            .should_defer_host_route_materialization(1)
    );
}
