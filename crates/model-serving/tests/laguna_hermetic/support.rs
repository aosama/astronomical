use astronomical_model_serving::{LagunaTargetContract, LagunaTargetNormalizer};
use serde_json::{Map, Value, json};

#[derive(Clone, Copy)]
pub(super) enum LagunaAcceptanceSize {
    ExtraSmall,
    Small,
}

/// Builds a small generalized config whose layer count can vary independently.
pub(super) fn config_value(layer_count: usize) -> Value {
    json!({
        "architectures": ["LagunaForCausalLM"],
        "model_type": "laguna",
        "vocab_size": 32_128,
        "hidden_size": 1_536,
        "intermediate_size": 4_096,
        "num_hidden_layers": layer_count,
        "num_attention_heads": 12,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "max_position_embeddings": 32_768,
        "attention_bias": false,
        "attention_dropout": 0.0,
        "rms_norm_eps": 0.00001,
        "tie_word_embeddings": false,
        "use_cache": true,
        "torch_dtype": "bfloat16",
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 0.5
        }
    })
}

/// Retains observed XS/S shapes as acceptance rows rather than allowed-domain constants.
pub(super) fn acceptance_config_value(fixture_size: LagunaAcceptanceSize) -> Value {
    let (
        hidden_size,
        intermediate_size,
        layer_count,
        sliding_query_head_count,
        maximum_position_count,
        experts_per_token,
        expert_intermediate_size,
        yarn_factor,
        yarn_beta_fast,
        yarn_attention_factor,
        quantization_group_size,
        uses_wrapped_namespace,
    ) = match fixture_size {
        LagunaAcceptanceSize::ExtraSmall => (
            2_048,
            8_192,
            40,
            64,
            262_144,
            8,
            512,
            32.0,
            64.0,
            1.346_573_590_279_972_7,
            64,
            false,
        ),
        LagunaAcceptanceSize::Small => (
            3_072,
            12_288,
            48,
            72,
            1_048_576,
            10,
            1_024,
            128.0,
            32.0,
            1.485_203_026_391_961_8,
            128,
            true,
        ),
    };
    let layer_types = (0..layer_count)
        .map(|layer_index| {
            if layer_index % 4 == 0 {
                "full"
            } else {
                "sliding"
            }
        })
        .collect::<Vec<_>>();
    let query_head_counts = layer_types
        .iter()
        .map(|layer_type| {
            if *layer_type == "full" {
                48
            } else {
                sliding_query_head_count
            }
        })
        .collect::<Vec<_>>();
    let mlp_layer_types = (0..layer_count)
        .map(|layer_index| if layer_index == 0 { "dense" } else { "sparse" })
        .collect::<Vec<_>>();
    let override_prefix = if uses_wrapped_namespace {
        "language_model."
    } else {
        ""
    };
    let mut quantization_fields = Map::new();
    assert!(
        quantization_fields
            .insert("bits".to_owned(), json!(2))
            .is_none()
    );
    assert!(
        quantization_fields
            .insert("group_size".to_owned(), json!(quantization_group_size))
            .is_none()
    );
    assert!(
        quantization_fields
            .insert("mode".to_owned(), json!("affine"))
            .is_none()
    );
    assert!(
        quantization_fields
            .insert(
                format!("{override_prefix}lm_head"),
                json!({"bits": 8, "group_size": 64, "mode": "affine"}),
            )
            .is_none()
    );
    let quantization = Value::Object(quantization_fields);
    let mut config = config_value(layer_count);
    config["vocab_size"] = json!(100_352);
    config["hidden_size"] = json!(hidden_size);
    config["intermediate_size"] = json!(intermediate_size);
    config["max_position_embeddings"] = json!(maximum_position_count);
    config["rms_norm_eps"] = json!(0.000001);
    config["sliding_window"] = json!(512);
    config["layer_types"] = json!(layer_types);
    config["num_attention_heads"] = json!(48);
    config["num_key_value_heads"] = json!(8);
    config["num_attention_heads_per_layer"] = json!(query_head_counts);
    config["gating_types"] = json!(vec!["per_head"; layer_count]);
    config["mlp_layer_types"] = json!(mlp_layer_types);
    config["num_experts"] = json!(256);
    config["num_experts_per_tok"] = json!(experts_per_token);
    config["moe_intermediate_size"] = json!(expert_intermediate_size);
    config["shared_expert_intermediate_size"] = json!(expert_intermediate_size);
    config["norm_topk_prob"] = json!(true);
    config["moe_routed_scaling_factor"] = json!(2.5);
    config["moe_apply_router_weight_on_input"] = json!(false);
    config["rope_parameters"] = json!({
        "full_attention": {
            "rope_type": "yarn",
            "rope_theta": 500_000.0,
            "factor": yarn_factor,
            "original_max_position_embeddings": 8_192,
            "beta_slow": 1.0,
            "beta_fast": yarn_beta_fast,
            "attention_factor": yarn_attention_factor,
            "partial_rotary_factor": 0.5
        },
        "sliding_attention": {
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 1.0
        }
    });
    config["quantization"] = quantization.clone();
    config["quantization_config"] = quantization;
    if matches!(fixture_size, LagunaAcceptanceSize::Small) {
        config["moe_router_logit_softcapping"] = json!(0.0);
    }
    config
}

pub(super) fn normalize(config: Value) -> LagunaTargetContract {
    LagunaTargetNormalizer::normalize(&config_bytes(&config))
        .expect("the synthetic generalized Laguna config should normalize")
}

pub(super) fn config_bytes(config: &Value) -> Vec<u8> {
    serde_json::to_vec(config).expect("the synthetic Laguna config should serialize")
}
