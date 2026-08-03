use astronomical_model_serving::{
    Qwen3_5MoEMtpArtifactCapability, Qwen3_5MoEMtpRuntimeState,
    qwen3_5_moe_depth_one_mtp_window_fits, qwen3_5_moe_mtp_runtime_state_after_load,
    qwen3_5_moe_mtp_verification_may_cross_thinking_budget,
};

#[test]
fn should_report_bounded_unavailability_after_optional_mtp_initialization_fails() {
    let (mtp_runtime_state, mtp_unavailable_reason) = qwen3_5_moe_mtp_runtime_state_after_load(
        true,
        &Qwen3_5MoEMtpArtifactCapability::MtpCapable {
            discovered_mtp_layer_count: 1,
            supported_mtp_draft_depth: 1,
            mtp_tensor_count: 42,
        },
        false,
    );

    assert_eq!(mtp_runtime_state, Qwen3_5MoEMtpRuntimeState::Unavailable);
    assert_eq!(
        mtp_unavailable_reason.as_deref(),
        Some("no compatible MTP head")
    );
}

#[test]
fn should_skip_depth_one_mtp_when_the_accepted_draft_could_reach_the_thinking_budget() {
    assert!(qwen3_5_moe_mtp_verification_may_cross_thinking_budget(
        true,
        8,
        Some(10),
        2,
    ));
}

#[test]
fn should_require_two_remaining_outputs_and_context_positions_for_depth_one_mtp() {
    assert!(!qwen3_5_moe_depth_one_mtp_window_fits(9, 10, 99, 100));
    assert!(!qwen3_5_moe_depth_one_mtp_window_fits(8, 10, 99, 100));
    assert!(!qwen3_5_moe_depth_one_mtp_window_fits(9, 10, 98, 100));
    assert!(qwen3_5_moe_depth_one_mtp_window_fits(8, 10, 98, 100));
}

#[test]
fn should_allow_depth_one_mtp_when_both_possible_emissions_remain_below_the_thinking_budget() {
    assert!(!qwen3_5_moe_mtp_verification_may_cross_thinking_budget(
        true,
        7,
        Some(10),
        2,
    ));
}

#[test]
fn should_allow_mtp_outside_thinking_or_without_a_budget() {
    assert!(!qwen3_5_moe_mtp_verification_may_cross_thinking_budget(
        false,
        9,
        Some(10),
        2,
    ));
    assert!(!qwen3_5_moe_mtp_verification_may_cross_thinking_budget(
        true, 9, None, 2,
    ));
}
