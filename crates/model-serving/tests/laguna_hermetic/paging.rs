use astronomical_model_serving::{
    ExpertResidencyPhase, LagunaArtifactValidator, LagunaCanonicalTensorAssemblyKind,
    LagunaExpertPagingPlan, LagunaExpertProjection, LagunaFeedForwardDescriptor,
    LagunaLayerTensorRole, LagunaPagingError, LagunaTargetNormalizer, LagunaTensorComponent,
    LagunaTensorId, PerformanceAttribution, PerformanceOperation,
    laguna_sliding_prefill_transient_token_count, rotating_prefill_transient_token_count,
};

use serde_json::json;

use super::artifact_support::SyntheticLagunaArtifact;
use super::support::{LagunaQualificationSize, qualification_config_value};

fn dense_then_sparse_fixture() -> SyntheticLagunaArtifact {
    let mut config = SyntheticLagunaArtifact::dense("").config;
    config["num_hidden_layers"] = json!(2);
    config["mlp_layer_types"] = json!(["dense", "sparse"]);
    config["num_experts"] = json!(2);
    config["num_experts_per_tok"] = json!(1);
    config["moe_intermediate_size"] = json!(3);
    config["shared_expert_intermediate_size"] = json!(0);
    let tensor_shapes = [
        ("model.embed_tokens.weight", vec![8, 4]),
        ("model.norm.weight", vec![4]),
        ("lm_head.weight", vec![8, 4]),
        ("model.layers.0.input_layernorm.weight", vec![4]),
        ("model.layers.0.post_attention_layernorm.weight", vec![4]),
        ("model.layers.0.self_attn.q_proj.weight", vec![4, 4]),
        ("model.layers.0.self_attn.k_proj.weight", vec![2, 4]),
        ("model.layers.0.self_attn.v_proj.weight", vec![2, 4]),
        ("model.layers.0.self_attn.o_proj.weight", vec![4, 4]),
        ("model.layers.0.self_attn.q_norm.weight", vec![2]),
        ("model.layers.0.self_attn.k_norm.weight", vec![2]),
        ("model.layers.0.mlp.gate_proj.weight", vec![6, 4]),
        ("model.layers.0.mlp.up_proj.weight", vec![6, 4]),
        ("model.layers.0.mlp.down_proj.weight", vec![4, 6]),
        ("model.layers.1.input_layernorm.weight", vec![4]),
        ("model.layers.1.post_attention_layernorm.weight", vec![4]),
        ("model.layers.1.self_attn.q_proj.weight", vec![4, 4]),
        ("model.layers.1.self_attn.k_proj.weight", vec![2, 4]),
        ("model.layers.1.self_attn.v_proj.weight", vec![2, 4]),
        ("model.layers.1.self_attn.o_proj.weight", vec![4, 4]),
        ("model.layers.1.self_attn.q_norm.weight", vec![2]),
        ("model.layers.1.self_attn.k_norm.weight", vec![2]),
        ("model.layers.1.mlp.gate.weight", vec![2, 4]),
        ("model.layers.1.mlp.experts.gate_proj.weight", vec![2, 3, 4]),
        ("model.layers.1.mlp.experts.up_proj.weight", vec![2, 3, 4]),
        ("model.layers.1.mlp.experts.down_proj.weight", vec![2, 4, 3]),
    ];
    SyntheticLagunaArtifact::from_tensor_shapes(config, "", &tensor_shapes)
}

fn validate_fixture(
    fixture: SyntheticLagunaArtifact,
) -> (
    tempfile::TempDir,
    astronomical_model_serving::ValidatedLagunaArtifact,
) {
    let model_directory =
        tempfile::tempdir().expect("the paging test should create a model directory");
    fixture.write(model_directory.path());
    let artifact = LagunaArtifactValidator::new()
        .validate(model_directory.path())
        .expect("the paging fixture should validate");
    (model_directory, artifact)
}

fn paging_plan(
    artifact: &astronomical_model_serving::ValidatedLagunaArtifact,
    model_directory: &std::path::Path,
) -> LagunaExpertPagingPlan {
    let mut performance_attribution = PerformanceAttribution::enabled();
    let plan = LagunaExpertPagingPlan::from_validated_artifact(
        artifact,
        model_directory,
        &mut performance_attribution,
    )
    .expect("canonical Laguna contracts should build a paging plan");
    assert!(
        performance_attribution
            .operation_measurement(PerformanceOperation::ExpertPagerPlanConstruction)
            .is_some()
    );
    plan
}

fn routed_weight_id(layer_index: usize, projection: LagunaExpertProjection) -> LagunaTensorId {
    LagunaTensorId::Layer {
        layer_index,
        role: LagunaLayerTensorRole::RoutedExpert(projection),
        component: LagunaTensorComponent::Weight,
    }
}

#[test]
fn should_page_only_routed_experts_and_keep_shared_and_router_resident() {
    let (model_directory, artifact) = validate_fixture(SyntheticLagunaArtifact::sparse_stacked());
    let plan = paging_plan(&artifact, model_directory.path());
    assert_eq!(plan.sparse_layers().len(), 1);
    let sparse_layer = &plan.sparse_layers()[0];
    assert_eq!(sparse_layer.decoder_layer_index(), 0);
    assert_eq!(sparse_layer.paging_slot_index(), 0);
    assert_eq!(sparse_layer.expert_capacity(), 2);
    assert_eq!(sparse_layer.experts_per_token(), 1);

    for projection in [
        LagunaExpertProjection::Gate,
        LagunaExpertProjection::Up,
        LagunaExpertProjection::Down,
    ] {
        assert!(
            artifact
                .tensor_contract()
                .descriptor(&routed_weight_id(0, projection))
                .is_some()
        );
    }
    assert!(
        artifact
            .tensor_contract()
            .descriptor(&LagunaTensorId::Layer {
                layer_index: 0,
                role: LagunaLayerTensorRole::Router,
                component: LagunaTensorComponent::Weight,
            })
            .is_some()
    );
}

#[test]
fn should_distinguish_complete_layer_prefill_pages_from_routed_decode_pages() {
    let (model_directory, artifact) = validate_fixture(SyntheticLagunaArtifact::sparse_stacked());
    let plan = paging_plan(&artifact, model_directory.path());
    let sparse_layer = &plan.sparse_layers()[0];
    let complete_page = sparse_layer
        .complete_layer_page()
        .expect("a complete-layer page should build");
    let routed_page = sparse_layer
        .routed_page(&[0])
        .expect("a routed decode page should build");

    assert!(complete_page.contains_all_experts());
    assert!(!routed_page.contains_all_experts());
    assert_eq!(complete_page.expert_ids, vec![0, 1]);
    assert_eq!(routed_page.expert_ids, vec![0]);
    assert_eq!(
        complete_page.payload_byte_count,
        sparse_layer
            .complete_layer_payload_byte_count()
            .expect("complete payload should be exact")
    );
    assert_eq!(
        routed_page.payload_byte_count,
        sparse_layer
            .routed_page_payload_byte_count()
            .expect("routed payload should be one top-K page")
    );
    assert_eq!(
        complete_page.payload_byte_count,
        routed_page.payload_byte_count * 2
    );
}

#[test]
fn should_produce_equivalent_payloads_for_stacked_and_per_expert_affine_layouts() {
    let (stacked_directory, stacked_artifact) =
        validate_fixture(SyntheticLagunaArtifact::direct_affine_sparse_stacked(4, 32));
    let (per_expert_directory, per_expert_artifact) = validate_fixture(
        SyntheticLagunaArtifact::direct_affine_sparse_per_expert(4, 32),
    );
    let stacked_plan = paging_plan(&stacked_artifact, stacked_directory.path());
    let per_expert_plan = paging_plan(&per_expert_artifact, per_expert_directory.path());
    let stacked_layer = &stacked_plan.sparse_layers()[0];
    let per_expert_layer = &per_expert_plan.sparse_layers()[0];

    assert_eq!(
        stacked_artifact
            .tensor_contract()
            .descriptor(&routed_weight_id(0, LagunaExpertProjection::Gate))
            .expect("stacked gate should exist")
            .assembly_kind(),
        LagunaCanonicalTensorAssemblyKind::StackedSource
    );
    assert_eq!(
        per_expert_artifact
            .tensor_contract()
            .descriptor(&routed_weight_id(0, LagunaExpertProjection::Gate))
            .expect("per-expert gate should exist")
            .assembly_kind(),
        LagunaCanonicalTensorAssemblyKind::PerExpertStack
    );
    assert_eq!(
        stacked_layer
            .complete_layer_payload_byte_count()
            .expect("stacked complete payload"),
        per_expert_layer
            .complete_layer_payload_byte_count()
            .expect("per-expert complete payload")
    );
    assert_eq!(
        stacked_layer
            .routed_page_payload_byte_count()
            .expect("stacked routed payload"),
        per_expert_layer
            .routed_page_payload_byte_count()
            .expect("per-expert routed payload")
    );
}

#[test]
fn should_produce_equivalent_payloads_for_fused_and_split_affine_layouts() {
    let (split_directory, split_artifact) =
        validate_fixture(SyntheticLagunaArtifact::direct_affine_sparse_stacked(4, 32));
    let (fused_directory, fused_artifact) = validate_fixture(
        SyntheticLagunaArtifact::direct_affine_sparse_fused_stacked(4, 32),
    );
    let split_plan = paging_plan(&split_artifact, split_directory.path());
    let fused_plan = paging_plan(&fused_artifact, fused_directory.path());
    assert_eq!(
        split_plan.sparse_layers()[0]
            .complete_layer_payload_byte_count()
            .expect("split complete payload"),
        fused_plan.sparse_layers()[0]
            .complete_layer_payload_byte_count()
            .expect("fused complete payload")
    );
}

#[test]
fn should_size_named_xs_and_s_routed_pages_from_canonical_router_geometry() {
    for (row_name, fixture_size, expected_experts_per_token) in [
        ("xs", LagunaQualificationSize::ExtraSmall, 8_u32),
        ("s", LagunaQualificationSize::Small, 10),
    ] {
        let contract = LagunaTargetNormalizer::normalize(
            &serde_json::to_vec(&qualification_config_value(fixture_size))
                .expect("qualification config should serialize"),
        )
        .unwrap_or_else(|_| panic!("{row_name} should normalize"));
        let sparse_layer = contract
            .layers()
            .iter()
            .find_map(|layer| match layer.feed_forward() {
                LagunaFeedForwardDescriptor::Moe(moe) => Some(moe),
                LagunaFeedForwardDescriptor::Dense(_) => None,
            })
            .unwrap_or_else(|| panic!("{row_name} should have a sparse layer"));
        assert_eq!(
            sparse_layer.experts_per_token(),
            expected_experts_per_token,
            "{row_name}"
        );
        assert_eq!(sparse_layer.expert_count(), 256, "{row_name}");
    }
}

#[test]
fn should_translate_laguna_geometry_into_phase_aware_residency_and_sliding_transients() {
    let (model_directory, artifact) = validate_fixture(SyntheticLagunaArtifact::sparse_stacked());
    let plan = paging_plan(&artifact, model_directory.path());
    let requirements = plan
        .request_memory_requirements(512, 128)
        .expect("request charges should be exact");
    assert_eq!(
        requirements.sliding_prefill_transient_token_count(),
        rotating_prefill_transient_token_count(512, 128).expect("neutral helper")
    );
    assert_eq!(
        laguna_sliding_prefill_transient_token_count(512, 128).expect("Laguna wrapper"),
        512 + 128 - 1
    );
    assert_eq!(
        requirements.complete_expert_payload_bytes(),
        plan.sparse_layers()[0]
            .complete_layer_payload_byte_count()
            .expect("complete bytes")
    );
    assert_eq!(
        requirements.complete_prefill_page_bytes(),
        plan.sparse_layers()[0]
            .complete_layer_payload_byte_count()
            .expect("complete bytes")
    );
    assert_eq!(
        requirements.routed_decode_page_bytes(),
        plan.sparse_layers()[0]
            .routed_page_payload_byte_count()
            .expect("routed bytes")
    );

    let residency_plan = plan
        .plan_phase_aware_residency(
            ExpertResidencyPhase::Prefill,
            requirements.complete_expert_payload_bytes(),
            &[],
        )
        .expect("a fitting complete payload should plan complete residency");
    assert_eq!(residency_plan.complete_layer_targets, vec![0]);
    assert!(matches!(
        plan.complete_residency_decision(
            0,
            0,
            requirements.complete_expert_payload_bytes()
                + requirements.complete_expert_payload_bytes() / 10
                + 1,
            0,
        )
        .expect("centralized residency decision"),
        astronomical_model_serving::CompleteResidencyDecision::Admit { .. }
    ));
}

#[test]
fn should_leave_dense_models_without_pageable_expert_layers() {
    let (model_directory, artifact) = validate_fixture(SyntheticLagunaArtifact::dense(""));
    let plan = paging_plan(&artifact, model_directory.path());
    assert!(plan.sparse_layers().is_empty());
}

#[test]
fn should_reject_out_of_range_routed_expert_ids() {
    let (model_directory, artifact) = validate_fixture(SyntheticLagunaArtifact::sparse_stacked());
    let plan = paging_plan(&artifact, model_directory.path());
    let rejection = plan.sparse_layers()[0].routed_page(&[0, 2]);
    assert!(matches!(
        rejection,
        Err(LagunaPagingError::Manifest(
            astronomical_model_serving::ExpertManifestError::ExpertIdExceedsCapacity { .. }
        ))
    ));
}

#[test]
fn should_remap_a_trailing_sparse_layer_onto_paging_slot_zero() {
    let (model_directory, artifact) = validate_fixture(dense_then_sparse_fixture());
    let plan = paging_plan(&artifact, model_directory.path());
    assert_eq!(plan.sparse_layers().len(), 1);
    let sparse_layer = &plan.sparse_layers()[0];
    assert_eq!(sparse_layer.decoder_layer_index(), 1);
    assert_eq!(sparse_layer.paging_slot_index(), 0);
    let geometry = plan
        .layer_geometries()
        .expect("dense-then-sparse geometry should be planner-ready");
    assert_eq!(geometry[0].layer_index, 0);
    let residency_plan = plan
        .plan_phase_aware_residency(
            ExpertResidencyPhase::Prefill,
            sparse_layer
                .complete_layer_payload_byte_count()
                .expect("complete payload"),
            &[],
        )
        .expect("slot zero should plan as a complete foundation");
    assert_eq!(residency_plan.complete_layer_targets, vec![0]);
}

#[test]
fn should_reject_a_zero_sliding_window_without_rederiving_the_formula() {
    assert!(matches!(
        laguna_sliding_prefill_transient_token_count(0, 8),
        Err(LagunaPagingError::InvalidSlidingTransient)
    ));
}
