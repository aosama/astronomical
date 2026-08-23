use astronomical_model_serving::{
    LagunaAttentionKind, LagunaCacheDescriptor, LagunaExecutionDtype, LagunaFeedForwardDescriptor,
    LagunaGatingKind, LagunaNormalizationError, LagunaRouterKind, LagunaTargetNormalizer,
};
use serde_json::{Value, json};

use super::support::{config_bytes, config_value, normalize};

#[test]
fn should_normalize_root_and_text_config_envelopes_into_the_same_contract() {
    let root_config = config_value(4);
    let mut nested_config = root_config.clone();
    let nested_object = nested_config
        .as_object_mut()
        .expect("the fixture should be an object");
    let model_type = nested_object
        .remove("model_type")
        .expect("the fixture should declare model_type");
    let architectures = nested_object
        .remove("architectures")
        .expect("the fixture should declare architectures");
    let storage_dtype = nested_object
        .remove("torch_dtype")
        .expect("the fixture should declare torch_dtype");
    let wrapped_config = json!({
        "model_type": model_type,
        "architectures": architectures,
        "torch_dtype": storage_dtype,
        "text_config": nested_config,
    });

    let root_contract = normalize(root_config);
    let nested_contract = normalize(wrapped_config);

    assert_eq!(root_contract, nested_contract);
    assert_eq!(nested_contract.model().layer_count(), 4);
    assert_eq!(nested_contract.layers().len(), 4);
}

#[test]
fn should_accept_equivalent_duplicates_and_reject_conflicting_envelope_fields() {
    let mut nested_geometry = config_value(3);
    nested_geometry["gating"] = json!("per_head");
    let equivalent = json!({
        "model_type": "laguna",
        "hidden_size": 1_536,
        "gating": "per-head",
        "text_config": nested_geometry.clone()
    });
    normalize(equivalent);

    let mut conflicting = json!({"hidden_size": 2_048, "text_config": nested_geometry});
    conflicting["model_type"] = json!("laguna");
    assert!(matches!(
        LagunaTargetNormalizer::normalize(&config_bytes(&conflicting)),
        Err(LagunaNormalizationError::ConflictingEnvelopeField {
            field_name,
        }) if field_name == "hidden_size"
    ));
}

#[test]
fn should_normalize_explicit_and_default_attention_schedules() {
    let schedule_rows = [
        (None, vec![LagunaAttentionKind::Full; 4]),
        (
            Some(json!(["full_attention", "full", "full_attention", "full"])),
            vec![LagunaAttentionKind::Full; 4],
        ),
        (
            Some(json!([
                "sliding_attention",
                "sliding",
                "sliding_attention",
                "sliding"
            ])),
            vec![LagunaAttentionKind::Sliding; 4],
        ),
        (
            Some(json!([
                "sliding",
                "full",
                "full_attention",
                "sliding_attention"
            ])),
            vec![
                LagunaAttentionKind::Sliding,
                LagunaAttentionKind::Full,
                LagunaAttentionKind::Full,
                LagunaAttentionKind::Sliding,
            ],
        ),
    ];

    for (explicit_schedule, expected_kinds) in schedule_rows {
        let mut config = config_value(4);
        if let Some(layer_types) = explicit_schedule {
            config["layer_types"] = layer_types;
            config["sliding_window"] = json!(768);
        }
        let contract = normalize(config);
        let actual_kinds = contract
            .layers()
            .iter()
            .map(|layer| layer.attention().kind())
            .collect::<Vec<_>>();
        assert_eq!(actual_kinds, expected_kinds);
        for layer in contract.layers() {
            let expected_cache = match layer.attention().kind() {
                LagunaAttentionKind::Full => LagunaCacheDescriptor::AppendOnly,
                LagunaAttentionKind::Sliding => {
                    LagunaCacheDescriptor::Rotating { window_size: 768 }
                }
            };
            assert_eq!(layer.attention().cache(), &expected_cache);
        }
    }
}

#[test]
fn should_apply_per_layer_query_heads_before_the_global_head_count() {
    let global_contract = normalize(config_value(3));
    assert!(
        global_contract
            .layers()
            .iter()
            .all(|layer| layer.attention().query_head_count() == 12)
    );

    let mut heterogeneous = config_value(3);
    heterogeneous["num_attention_heads"] = json!(10);
    heterogeneous["num_attention_heads_per_layer"] = json!([8, 12, 16]);
    let heterogeneous_contract = normalize(heterogeneous);
    assert_eq!(
        heterogeneous_contract
            .layers()
            .iter()
            .map(|layer| layer.attention().query_head_count())
            .collect::<Vec<_>>(),
        vec![8, 12, 16]
    );
}

#[test]
fn should_normalize_explicit_legacy_and_dense_only_feed_forward_schedules() {
    let dense_contract = normalize(config_value(4));
    assert!(
        dense_contract
            .layers()
            .iter()
            .all(|layer| matches!(layer.feed_forward(), LagunaFeedForwardDescriptor::Dense(_)))
    );

    let mut explicit = config_value(4);
    add_sparse_geometry(&mut explicit);
    explicit["mlp_layer_types"] = json!(["sparse", "dense", "moe", "dense"]);
    explicit["mlp_only_layers"] = json!([0]);
    explicit["decoder_sparse_step"] = json!(1);
    let explicit_contract = normalize(explicit);
    assert_eq!(sparse_layer_indexes(&explicit_contract), vec![0, 2]);
    let LagunaFeedForwardDescriptor::Moe(first_layer_moe) =
        explicit_contract.layers()[0].feed_forward()
    else {
        panic!("the explicit sparse layer should expose its MoE descriptor");
    };
    assert_eq!(first_layer_moe.router_kind(), LagunaRouterKind::SigmoidTopK);
    assert_eq!(first_layer_moe.expert_count(), 16);
    assert_eq!(first_layer_moe.experts_per_token(), 4);

    let mut legacy = config_value(5);
    add_sparse_geometry(&mut legacy);
    legacy["mlp_only_layers"] = json!([2]);
    legacy["decoder_sparse_step"] = json!(2);
    let legacy_contract = normalize(legacy);
    assert_eq!(sparse_layer_indexes(&legacy_contract), vec![1, 3]);
}

#[test]
fn should_normalize_none_global_and_per_layer_gating_aliases() {
    let mut disabled = config_value(3);
    disabled["gating"] = json!(false);
    let disabled_contract = normalize(disabled);
    assert!(
        disabled_contract
            .layers()
            .iter()
            .all(|layer| { layer.attention().gating_kind() == LagunaGatingKind::None })
    );

    let mut global = config_value(3);
    global["gating"] = json!("per-head");
    let global_contract = normalize(global);
    assert!(
        global_contract
            .layers()
            .iter()
            .all(|layer| { layer.attention().gating_kind() == LagunaGatingKind::PerHead })
    );

    let mut per_layer = config_value(3);
    per_layer["gating"] = json!(true);
    per_layer["gating_types"] = json!(["none", "per-element", "per_element"]);
    let per_layer_contract = normalize(per_layer);
    assert_eq!(
        per_layer_contract
            .layers()
            .iter()
            .map(|layer| layer.attention().gating_kind())
            .collect::<Vec<_>>(),
        vec![
            LagunaGatingKind::None,
            LagunaGatingKind::PerElement,
            LagunaGatingKind::PerElement,
        ]
    );
}

#[test]
fn should_canonicalize_omitted_zero_and_positive_router_softcaps() {
    for (softcap, expected_softcap) in [
        (None, 0.0),
        (Some(json!(0.0)), 0.0),
        (Some(json!(30.5)), 30.5),
    ] {
        let mut config = config_value(2);
        if let Some(softcap_value) = softcap {
            config["moe_router_logit_softcapping"] = softcap_value;
        }
        assert_eq!(
            normalize(config).model().router_logit_softcap(),
            expected_softcap
        );
    }
}

#[test]
fn should_reject_malformed_oversized_unsafe_and_inconsistent_declarations() {
    assert!(matches!(
        LagunaTargetNormalizer::normalize(b"{not-json"),
        Err(LagunaNormalizationError::MalformedJson(_))
    ));
    let oversized_json = vec![b' '; 32 * 1024 * 1024 + 1];
    assert!(matches!(
        LagunaTargetNormalizer::normalize(&oversized_json),
        Err(LagunaNormalizationError::ConfigTooLarge { .. })
    ));
    let mut non_finite_softcap = config_bytes(&config_value(1));
    assert_eq!(non_finite_softcap.pop(), Some(b'}'));
    non_finite_softcap.extend_from_slice(b",\"moe_router_logit_softcapping\":1e400}");
    assert!(matches!(
        LagunaTargetNormalizer::normalize(&non_finite_softcap),
        Err(LagunaNormalizationError::MalformedJson(_))
    ));

    let invalid_rows = [
        ("hidden_size", json!(0)),
        ("num_hidden_layers", json!(4_294_967_296_u64)),
        ("moe_router_logit_softcapping", json!(-1.0)),
    ];
    for (field_name, invalid_value) in invalid_rows {
        let mut config = config_value(3);
        config[field_name] = invalid_value;
        assert!(matches!(
            LagunaTargetNormalizer::normalize(&config_bytes(&config)),
            Err(LagunaNormalizationError::InvalidNumericValue { .. })
        ));
    }

    for field_name in [
        "layer_types",
        "num_attention_heads_per_layer",
        "gating_types",
        "mlp_layer_types",
    ] {
        let mut config = config_value(3);
        config[field_name] = match field_name {
            "layer_types" => json!(["full", "full"]),
            "num_attention_heads_per_layer" => json!([12, 12]),
            "gating_types" => json!(["none", "none"]),
            _ => json!(["dense", "dense"]),
        };
        assert!(matches!(
            LagunaTargetNormalizer::normalize(&config_bytes(&config)),
            Err(LagunaNormalizationError::LayerArrayLengthMismatch { .. })
        ));
    }

    let mut indivisible_heads = config_value(2);
    indivisible_heads["num_attention_heads_per_layer"] = json!([12, 10]);
    assert!(matches!(
        LagunaTargetNormalizer::normalize(&config_bytes(&indivisible_heads)),
        Err(LagunaNormalizationError::InvalidHeadDivisibility { layer_index: 1, .. })
    ));

    let mut invalid_top_k = config_value(2);
    add_sparse_geometry(&mut invalid_top_k);
    invalid_top_k["num_experts_per_tok"] = json!(17);
    assert!(matches!(
        LagunaTargetNormalizer::normalize(&config_bytes(&invalid_top_k)),
        Err(LagunaNormalizationError::TopKExceedsExpertCount { .. })
    ));
}

#[test]
fn should_reject_unsupported_attention_mlp_gating_and_model_values() {
    let mutations = [
        ("layer_types", json!(["linear", "full"])),
        ("mlp_layer_types", json!(["hybrid", "dense"])),
        ("gating_types", json!(["channel", "none"])),
        ("torch_dtype", json!("float64")),
    ];
    for (field_name, unsupported_value) in mutations {
        let mut config = config_value(2);
        config[field_name] = unsupported_value;
        assert!(matches!(
            LagunaTargetNormalizer::normalize(&config_bytes(&config)),
            Err(LagunaNormalizationError::UnsupportedValue { .. })
        ));
    }

    let mut ambiguous_boolean = config_value(2);
    ambiguous_boolean["gating"] = json!(true);
    assert!(matches!(
        LagunaTargetNormalizer::normalize(&config_bytes(&ambiguous_boolean)),
        Err(LagunaNormalizationError::AmbiguousGatingBoolean)
    ));
}

#[test]
fn should_normalize_a_synthetic_non_pinned_geometry() {
    let mut config = config_value(7);
    config["vocab_size"] = json!(65_537);
    config["hidden_size"] = json!(1_792);
    config["intermediate_size"] = json!(5_376);
    config["num_attention_heads"] = json!(14);
    config["num_key_value_heads"] = json!(2);
    config["head_dim"] = json!(96);
    config["num_attention_heads_per_layer"] = json!([14, 16, 18, 20, 22, 24, 26]);
    config["layer_types"] = json!([
        "full", "sliding", "sliding", "full", "sliding", "full", "full"
    ]);
    config["sliding_window"] = json!(640);
    config["rope_parameters"]["partial_rotary_factor"] = json!(0.5);

    let contract = normalize(config);

    assert_eq!(contract.model().vocabulary_size(), 65_537);
    assert_eq!(contract.layers().len(), 7);
    assert_eq!(contract.layers()[6].attention().query_head_count(), 26);
    assert_eq!(
        contract.layers()[1].attention().rope().rotary_dimension(),
        48
    );
    assert_eq!(
        contract.model().execution_dtype(),
        LagunaExecutionDtype::Bfloat16
    );
}

fn add_sparse_geometry(config: &mut Value) {
    // Sparse schedule tests declare complete router geometry so failures identify scheduling.
    config["num_experts"] = json!(16);
    config["num_experts_per_tok"] = json!(4);
    config["moe_intermediate_size"] = json!(768);
    config["shared_expert_intermediate_size"] = json!(512);
    config["norm_topk_prob"] = json!(true);
    config["moe_routed_scaling_factor"] = json!(1.5);
    config["moe_apply_router_weight_on_input"] = json!(false);
}

fn sparse_layer_indexes(contract: &astronomical_model_serving::LagunaTargetContract) -> Vec<usize> {
    contract
        .layers()
        .iter()
        .filter_map(|layer| {
            matches!(layer.feed_forward(), LagunaFeedForwardDescriptor::Moe(_))
                .then_some(layer.layer_index())
        })
        .collect()
}
