use std::collections::BTreeSet;

use astronomical_model_serving::{
    LagunaAttentionProjection, LagunaExpertGateUpLayout, LagunaExpertProjection,
    LagunaGlobalTensorRole, LagunaLayerTensorRole, LagunaRawTensorNameRecord, LagunaTensorAssembly,
    LagunaTensorComponent, LagunaTensorId, LagunaTensorNameContract,
    LagunaTensorNameNormalizationError, LagunaTensorNameNormalizer,
};

#[test]
fn should_canonicalize_the_complete_startup_inventory_without_leaking_namespace_wrappers() {
    let bare_names = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.layers.4.input_layernorm.weight",
        "model.layers.4.self_attn.q_proj.weight",
        "model.layers.4.self_attn.k_norm.weight",
        "model.layers.4.post_attention_layernorm.weight",
    ];
    let wrapped_names = bare_names.map(|name| format!("language_model.{name}"));

    let bare_contract = normalize(9, 0, bare_names);
    let wrapped_contract = normalize(9, 0, wrapped_names);

    // Namespace spelling remains provenance only; downstream keys are structured IDs.
    assert_eq!(
        bare_contract.assemblies().keys().collect::<BTreeSet<_>>(),
        wrapped_contract
            .assemblies()
            .keys()
            .collect::<BTreeSet<_>>()
    );
    assert!(
        bare_contract
            .assemblies()
            .contains_key(&LagunaTensorId::Global {
                role: LagunaGlobalTensorRole::TokenEmbedding,
                component: LagunaTensorComponent::Weight,
            })
    );
    assert!(matches!(
        assembly(
            &bare_contract,
            &LagunaTensorId::Global {
                role: LagunaGlobalTensorRole::FinalNormalization,
                component: LagunaTensorComponent::Weight,
            },
        ),
        LagunaTensorAssembly::DirectAlias { .. }
    ));
    assert!(bare_contract.assemblies().contains_key(&layer_id(
        4,
        LagunaLayerTensorRole::Attention(LagunaAttentionProjection::Query),
    )));
    assert_eq!(
        source_names(assembly(
            &wrapped_contract,
            &LagunaTensorId::Global {
                role: LagunaGlobalTensorRole::OutputHead,
                component: LagunaTensorComponent::Weight,
            },
        )),
        vec!["language_model.lm_head.weight"]
    );
}

#[test]
fn should_reject_repeated_unknown_and_mixed_namespaces() {
    assert!(matches!(
        normalize_error(
            3,
            0,
            ["language_model.language_model.model.embed_tokens.weight"],
        ),
        LagunaTensorNameNormalizationError::RepeatedLanguageModelWrapper { .. }
    ));
    assert!(matches!(
        normalize_error(3, 0, ["vision_tower.patch_embed.weight"]),
        LagunaTensorNameNormalizationError::UnknownTensorRoot { .. }
    ));
    assert!(matches!(
        normalize_error(3, 0, ["model.layers.1.self_attn.future_proj.weight"]),
        LagunaTensorNameNormalizationError::UnknownTensorName { .. }
    ));
    assert!(matches!(
        normalize_error(
            3,
            0,
            ["model.embed_tokens.weight", "language_model.lm_head.weight",],
        ),
        LagunaTensorNameNormalizationError::MixedTensorNamespaces
    ));
}

#[test]
fn should_normalize_router_and_correction_bias_aliases_and_reject_alias_collisions() {
    for router_name in [
        "model.layers.2.mlp.gate.weight",
        "model.layers.2.mlp.gate.proj.weight",
    ] {
        let contract = normalize(6, 11, [router_name]);
        assert!(
            contract
                .assemblies()
                .contains_key(&layer_id(2, LagunaLayerTensorRole::Router,))
        );
    }

    for correction_bias_name in [
        "model.layers.2.mlp.gate.e_score_correction_bias",
        "model.layers.2.mlp.e_score_correction_bias",
        "model.layers.2.mlp.experts.e_score_correction_bias",
        "model.layers.2.mlp.switch_mlp.e_score_correction_bias",
    ] {
        let contract = normalize(6, 11, [correction_bias_name]);
        assert!(
            contract
                .assemblies()
                .contains_key(&layer_id(2, LagunaLayerTensorRole::RouterCorrectionBias,))
        );
    }

    assert!(matches!(
        normalize_error(
            6,
            11,
            [
                "model.layers.2.mlp.gate.weight",
                "model.layers.2.mlp.gate.proj.weight",
            ],
        ),
        LagunaTensorNameNormalizationError::CanonicalCollision { .. }
    ));
}

#[test]
fn should_canonicalize_dense_shared_and_shared_gate_projections() {
    let contract = normalize(
        8,
        13,
        [
            "model.layers.5.mlp.gate_proj.weight",
            "model.layers.5.mlp.up_proj.weight",
            "model.layers.5.mlp.down_proj.weight",
            "model.layers.6.mlp.shared_expert.gate_proj.weight",
            "model.layers.6.mlp.shared_expert.up_proj.weight",
            "model.layers.6.mlp.shared_expert.down_proj.weight",
            "model.layers.6.mlp.shared_expert_gate.weight",
        ],
    );

    for projection in [
        LagunaExpertProjection::Gate,
        LagunaExpertProjection::Up,
        LagunaExpertProjection::Down,
    ] {
        assert!(contract.assemblies().contains_key(&layer_id(
            5,
            LagunaLayerTensorRole::DenseFeedForward(projection),
        )));
        assert!(contract.assemblies().contains_key(&layer_id(
            6,
            LagunaLayerTensorRole::SharedExpert(projection),
        )));
    }
    assert!(
        contract
            .assemblies()
            .contains_key(&layer_id(6, LagunaLayerTensorRole::SharedExpertGate,))
    );
}

#[test]
fn should_classify_already_stacked_switch_mlp_as_split_sources() {
    let contract = normalize(10, 17, split_stacked_expert_names(7, "switch_mlp"));

    assert_eq!(
        contract.expert_gate_up_layout(7),
        Some(LagunaExpertGateUpLayout::Split)
    );
    for projection in [
        LagunaExpertProjection::Gate,
        LagunaExpertProjection::Up,
        LagunaExpertProjection::Down,
    ] {
        assert!(matches!(
            assembly(
                &contract,
                &layer_id(7, LagunaLayerTensorRole::RoutedExpert(projection)),
            ),
            LagunaTensorAssembly::StackedSource { .. }
        ));
    }
}

#[test]
fn should_stack_complete_per_expert_sources_in_expert_index_order() {
    let expert_count = 7;
    let mut reverse_ordered_names = split_per_expert_names(9, expert_count);
    reverse_ordered_names.reverse();
    let contract = normalize(12, expert_count, reverse_ordered_names);
    let gate_assembly = assembly(
        &contract,
        &layer_id(
            9,
            LagunaLayerTensorRole::RoutedExpert(LagunaExpertProjection::Gate),
        ),
    );

    assert_eq!(
        contract.expert_gate_up_layout(9),
        Some(LagunaExpertGateUpLayout::Split)
    );
    assert!(matches!(
        gate_assembly,
        LagunaTensorAssembly::PerExpertStack { .. }
    ));
    assert_eq!(
        source_names(gate_assembly).first(),
        Some(&"model.layers.9.mlp.experts.0.gate_proj.weight")
    );
    assert_eq!(
        source_names(gate_assembly).last(),
        Some(&"model.layers.9.mlp.experts.6.gate_proj.weight")
    );
}

#[test]
fn should_classify_stacked_and_per_expert_fused_gate_up_sources() {
    let stacked_contract = normalize(
        6,
        19,
        [
            "model.layers.3.mlp.experts.gate_up_proj.weight".to_owned(),
            "model.layers.3.mlp.experts.down_proj.weight".to_owned(),
        ],
    );
    assert_fused_gate_up_assemblies(&stacked_contract, 3, false);

    let expert_count = 5;
    let mut per_expert_names = Vec::new();
    for expert_index in 0..expert_count {
        per_expert_names.push(format!(
            "model.layers.4.mlp.experts.{expert_index}.gate_up_proj.weight"
        ));
        per_expert_names.push(format!(
            "model.layers.4.mlp.experts.{expert_index}.down_proj.weight"
        ));
    }
    let per_expert_contract = normalize(6, expert_count, per_expert_names);
    assert_fused_gate_up_assemblies(&per_expert_contract, 4, true);
}

#[test]
fn should_reject_invalid_layer_and_expert_indexes() {
    assert!(matches!(
        normalize_error(7, 3, ["model.layers.7.self_attn.q_proj.weight"]),
        LagunaTensorNameNormalizationError::InvalidLayerIndex {
            layer_index: 7,
            layer_count: 7,
            ..
        }
    ));
    assert!(matches!(
        normalize_error(7, 3, ["model.layers.2.mlp.experts.3.gate_proj.weight"],),
        LagunaTensorNameNormalizationError::InvalidExpertIndex {
            expert_index: 3,
            expert_count: 3,
            ..
        }
    ));
}

#[test]
fn should_reject_duplicate_canonical_ids_partial_expert_sets_and_mixed_packaging() {
    assert!(matches!(
        normalize_error(
            4,
            0,
            [
                "model.layers.1.self_attn.q_proj.weight",
                "model.layers.1.self_attn.q_proj.weight",
            ],
        ),
        LagunaTensorNameNormalizationError::CanonicalCollision { .. }
    ));

    let mut partial_names = split_per_expert_names(2, 5);
    partial_names.retain(|name| !name.contains("experts.3.gate_proj"));
    assert!(matches!(
        normalize_error(4, 5, partial_names),
        LagunaTensorNameNormalizationError::IncompleteExpertSet {
            layer_index: 2,
            projection: LagunaExpertProjection::Gate,
            expected_expert_count: 5,
            actual_expert_count: 4,
            ..
        }
    ));

    let mut mixed_names = vec!["model.layers.2.mlp.switch_mlp.gate_proj.weight".to_owned()];
    for expert_index in 0..5 {
        for projection_name in ["up_proj", "down_proj"] {
            mixed_names.push(format!(
                "model.layers.2.mlp.experts.{expert_index}.{projection_name}.weight"
            ));
        }
    }
    assert!(matches!(
        normalize_error(4, 5, mixed_names),
        LagunaTensorNameNormalizationError::MixedExpertPackaging { layer_index: 2 }
    ));
}

#[test]
fn should_support_synthetic_non_pinned_layer_and_expert_counts() {
    let layer_count = 11;
    let expert_count = 23;
    let contract = normalize(
        layer_count,
        expert_count,
        split_per_expert_names(layer_count - 1, expert_count),
    );

    assert_eq!(contract.assemblies().len(), 3);
    assert_eq!(
        source_names(assembly(
            &contract,
            &layer_id(
                layer_count - 1,
                LagunaLayerTensorRole::RoutedExpert(LagunaExpertProjection::Down),
            ),
        ))
        .len(),
        expert_count
    );
}

fn normalize<I, S>(
    layer_count: usize,
    expert_count: usize,
    raw_names: I,
) -> LagunaTensorNameContract
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    LagunaTensorNameNormalizer::new(layer_count, expert_count)
        .normalize(&records(raw_names))
        .expect("the synthetic Laguna tensor inventory should normalize")
}

fn normalize_error<I, S>(
    layer_count: usize,
    expert_count: usize,
    raw_names: I,
) -> LagunaTensorNameNormalizationError
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    LagunaTensorNameNormalizer::new(layer_count, expert_count)
        .normalize(&records(raw_names))
        .expect_err("the malformed Laguna tensor inventory should fail")
}

fn records<I, S>(raw_names: I) -> Vec<LagunaRawTensorNameRecord>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    raw_names
        .into_iter()
        .map(LagunaRawTensorNameRecord::new)
        .collect()
}

fn layer_id(layer_index: usize, role: LagunaLayerTensorRole) -> LagunaTensorId {
    LagunaTensorId::Layer {
        layer_index,
        role,
        component: LagunaTensorComponent::Weight,
    }
}

fn assembly<'a>(
    contract: &'a LagunaTensorNameContract,
    tensor_id: &LagunaTensorId,
) -> &'a LagunaTensorAssembly {
    contract
        .assemblies()
        .get(tensor_id)
        .expect("the canonical tensor assembly should exist")
}

fn source_names(assembly: &LagunaTensorAssembly) -> Vec<&str> {
    assembly
        .sources()
        .iter()
        .map(|source| source.raw_name())
        .collect()
}

fn split_stacked_expert_names(layer_index: usize, owner: &str) -> Vec<String> {
    ["gate_proj", "up_proj", "down_proj"]
        .into_iter()
        .map(|projection_name| {
            format!("model.layers.{layer_index}.mlp.{owner}.{projection_name}.weight")
        })
        .collect()
}

fn split_per_expert_names(layer_index: usize, expert_count: usize) -> Vec<String> {
    let mut raw_names = Vec::new();
    for expert_index in 0..expert_count {
        for projection_name in ["gate_proj", "up_proj", "down_proj"] {
            raw_names.push(format!(
                "model.layers.{layer_index}.mlp.experts.{expert_index}.{projection_name}.weight"
            ));
        }
    }
    raw_names
}

fn assert_fused_gate_up_assemblies(
    contract: &LagunaTensorNameContract,
    layer_index: usize,
    is_per_expert: bool,
) {
    assert_eq!(
        contract.expert_gate_up_layout(layer_index),
        Some(LagunaExpertGateUpLayout::Fused)
    );
    for projection in [LagunaExpertProjection::Gate, LagunaExpertProjection::Up] {
        let routed_assembly = assembly(
            contract,
            &layer_id(layer_index, LagunaLayerTensorRole::RoutedExpert(projection)),
        );
        assert!(matches!(
            (is_per_expert, routed_assembly),
            (false, LagunaTensorAssembly::FusedGateUpSource { .. })
                | (true, LagunaTensorAssembly::FusedPerExpertGateUp { .. })
        ));
    }
}
