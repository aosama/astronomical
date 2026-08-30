use astronomical_model_serving::{
    MtpDraftDepth, Qwen3_5MtpArtifactCapability, Qwen3_5MtpRuntimeState,
    qwen3_5_depth_one_mtp_window_fits, qwen3_5_mtp_runtime_configuration_after_load,
    qwen3_5_mtp_runtime_state_after_load,
};

#[test]
fn should_report_bounded_unavailability_after_optional_mtp_initialization_fails() {
    let (mtp_runtime_state, mtp_unavailable_reason) = qwen3_5_mtp_runtime_state_after_load(
        true,
        &Qwen3_5MtpArtifactCapability::MtpCapable {
            stored_mtp_layer_count: 1,
            artifact_maximum_draft_depth: Some(MtpDraftDepth::DEPTH_ONE),
            artifact_default_draft_depth: None,
            mtp_tensor_count: 42,
        },
        false,
    );

    assert_eq!(mtp_runtime_state, Qwen3_5MtpRuntimeState::Unavailable);
    assert_eq!(
        mtp_unavailable_reason.as_deref(),
        Some("no compatible MTP head")
    );
}

#[test]
fn should_resolve_explicit_depth_as_the_effective_execution_depth() {
    let artifact_maximum_depth = MtpDraftDepth::new(3).expect("depth three should be valid");
    let capability = Qwen3_5MtpArtifactCapability::MtpCapable {
        stored_mtp_layer_count: 1,
        artifact_maximum_draft_depth: Some(artifact_maximum_depth),
        artifact_default_draft_depth: Some(
            MtpDraftDepth::new(2).expect("depth two should be valid"),
        ),
        mtp_tensor_count: 42,
    };

    let (runtime_state, reason, depth_status) = qwen3_5_mtp_runtime_configuration_after_load(
        true,
        Some(artifact_maximum_depth),
        &capability,
        true,
    );

    assert_eq!(runtime_state, Qwen3_5MtpRuntimeState::Active);
    assert_eq!(reason, None);
    assert_eq!(depth_status.resolved_requested_draft_depth, Some(3));
    assert_eq!(depth_status.effective_execution_draft_depth, Some(3));
}

#[test]
fn should_use_an_explicit_artifact_default_for_automatic_depth() {
    let maximum_depth = MtpDraftDepth::new(3).expect("depth three should be valid");
    let default_depth = MtpDraftDepth::new(2).expect("depth two should be valid");
    let capability = Qwen3_5MtpArtifactCapability::MtpCapable {
        stored_mtp_layer_count: 1,
        artifact_maximum_draft_depth: Some(maximum_depth),
        artifact_default_draft_depth: Some(default_depth),
        mtp_tensor_count: 42,
    };

    let (_, _, depth_status) =
        qwen3_5_mtp_runtime_configuration_after_load(true, None, &capability, true);

    assert_eq!(depth_status.resolved_requested_draft_depth, Some(2));
}

#[test]
fn should_use_production_depth_one_when_the_artifact_has_no_default() {
    let capability = Qwen3_5MtpArtifactCapability::MtpCapable {
        stored_mtp_layer_count: 1,
        artifact_maximum_draft_depth: Some(
            MtpDraftDepth::new(3).expect("depth three should be valid"),
        ),
        artifact_default_draft_depth: None,
        mtp_tensor_count: 42,
    };

    let (_, _, depth_status) =
        qwen3_5_mtp_runtime_configuration_after_load(true, None, &capability, true);

    assert_eq!(depth_status.resolved_requested_draft_depth, Some(1));
}

#[test]
fn should_clamp_an_over_requested_depth_to_a_declared_artifact_maximum_without_disabling() {
    let capability = Qwen3_5MtpArtifactCapability::MtpCapable {
        stored_mtp_layer_count: 1,
        artifact_maximum_draft_depth: Some(MtpDraftDepth::DEPTH_ONE),
        artifact_default_draft_depth: None,
        mtp_tensor_count: 42,
    };
    let (runtime_state, reason, depth_status) = qwen3_5_mtp_runtime_configuration_after_load(
        true,
        Some(MtpDraftDepth::new(3).expect("depth three should be valid")),
        &capability,
        true,
    );

    assert_eq!(runtime_state, Qwen3_5MtpRuntimeState::Active);
    assert_eq!(reason, None);
    assert_eq!(depth_status.configured_draft_depth, Some(3));
    assert_eq!(depth_status.artifact_maximum_draft_depth, Some(1));
    assert_eq!(depth_status.resolved_requested_draft_depth, Some(3));
    assert_eq!(depth_status.capped_draft_depth, Some(1));
    assert_eq!(depth_status.effective_execution_draft_depth, Some(1));
    assert_eq!(
        depth_status.resolution_reason,
        Some(astronomical_ipc_protocol::MtpDepthResolutionReason::ConfiguredDepthClampedToArtifactMaximum)
    );
}

#[test]
fn should_honor_an_explicit_depth_when_the_artifact_declares_no_maximum() {
    let capability = Qwen3_5MtpArtifactCapability::MtpCapable {
        stored_mtp_layer_count: 1,
        artifact_maximum_draft_depth: None,
        artifact_default_draft_depth: None,
        mtp_tensor_count: 42,
    };
    let (runtime_state, reason, depth_status) = qwen3_5_mtp_runtime_configuration_after_load(
        true,
        Some(MtpDraftDepth::new(3).expect("depth three should be valid")),
        &capability,
        true,
    );

    assert_eq!(runtime_state, Qwen3_5MtpRuntimeState::Active);
    assert_eq!(reason, None);
    assert_eq!(depth_status.configured_draft_depth, Some(3));
    assert_eq!(depth_status.artifact_maximum_draft_depth, None);
    assert_eq!(depth_status.resolved_requested_draft_depth, Some(3));
    assert_eq!(depth_status.capped_draft_depth, Some(3));
    assert_eq!(depth_status.effective_execution_draft_depth, Some(3));
    assert_eq!(
        depth_status.resolution_reason,
        Some(astronomical_ipc_protocol::MtpDepthResolutionReason::ConfiguredDepthExceedsAutomaticGuidance)
    );
}

#[test]
fn should_keep_depth_one_as_the_automatic_default_on_a_silent_artifact() {
    let capability = Qwen3_5MtpArtifactCapability::MtpCapable {
        stored_mtp_layer_count: 1,
        artifact_maximum_draft_depth: None,
        artifact_default_draft_depth: None,
        mtp_tensor_count: 42,
    };

    let (runtime_state, reason, depth_status) =
        qwen3_5_mtp_runtime_configuration_after_load(true, None, &capability, true);

    assert_eq!(runtime_state, Qwen3_5MtpRuntimeState::Active);
    assert_eq!(reason, None);
    assert_eq!(depth_status.resolved_requested_draft_depth, Some(1));
    assert_eq!(depth_status.capped_draft_depth, Some(1));
    assert_eq!(depth_status.effective_execution_draft_depth, Some(1));
    assert_eq!(depth_status.resolution_reason, None);
}

#[test]
fn should_require_two_remaining_outputs_and_context_positions_for_depth_one_mtp() {
    assert!(!qwen3_5_depth_one_mtp_window_fits(9, 10, 99, 100));
    assert!(!qwen3_5_depth_one_mtp_window_fits(8, 10, 99, 100));
    assert!(!qwen3_5_depth_one_mtp_window_fits(9, 10, 98, 100));
    assert!(qwen3_5_depth_one_mtp_window_fits(8, 10, 98, 100));
}
