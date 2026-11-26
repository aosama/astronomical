//! The frozen Ornith 1.0 configuration documents: the config, OptiQ
//! metadata, vision config and mixed-precision config bytes one Qwen 3.5
//! model contract is pinned to. They hold only JSON, no model types, so the
//! Qwen 3.5 unit tests, the serving integration tests and the acceptance GPU
//! contracts all build from this one copy and cannot drift apart; each keeps
//! its own typed wrappers beside its tests.

use serde_json::{Value, json};

pub fn frozen_ornith_1_0_config_bytes() -> Vec<u8> {
    let config_bytes = br#"
    {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "torch_dtype": "bfloat16",
        "eos_token_id": [248046, 248044],
        "pad_token_id": 248044,
        "tie_word_embeddings": false,
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_act": "silu",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "rms_norm_eps": 0.000001,
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "attention_bias": false,
            "mlp_bias": false,
            "norm_topk_prob": true,
            "output_router_logits": false,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "full_attention_interval": 4,
            "linear_conv_kernel_dim": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512
        }
    }
    "#;
    let mut config_value = serde_json::from_slice::<Value>(config_bytes)
        .expect("the frozen test config should decode as JSON");
    let mut quantization = json!({
        "group_size": 64,
        "bits": 6,
        "mode": "affine",
    });
    for decoder_layer_index in 0..40 {
        for gate_name in ["gate", "shared_expert_gate"] {
            quantization
                [format!("language_model.model.layers.{decoder_layer_index}.mlp.{gate_name}")] =
                json!({"group_size": 64, "bits": 8});
        }
    }
    // Embedding and lm_head must be 8-bit even with a 6-bit default.
    quantization["language_model.model.embed_tokens"] = json!({"group_size": 64, "bits": 8});
    quantization["language_model.lm_head"] = json!({"group_size": 64, "bits": 8});
    config_value["quantization"] = quantization.clone();
    config_value["quantization_config"] = quantization;
    config_value["text_config"]["layer_types"] = json!(expected_layer_types());
    serde_json::to_vec(&config_value).expect("the frozen test config should serialize as JSON")
}

pub fn frozen_ornith_1_0_optiq_config_bytes() -> Vec<u8> {
    let config_bytes = br#"
    {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "dtype": "bfloat16",
        "eos_token_id": [248046, 248044],
        "pad_token_id": 248044,
        "tie_word_embeddings": false,
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_act": "silu",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "rms_norm_eps": 0.000001,
            "rope_parameters": {
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000.0,
                "type": "default"
            },
            "partial_rotary_factor": 0.25,
            "attention_bias": false,
            "mlp_bias": false,
            "norm_topk_prob": true,
            "output_router_logits": false,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "full_attention_interval": 4,
            "linear_conv_kernel_dim": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512
        }
    }
    "#;
    let mut config_value = serde_json::from_slice::<Value>(config_bytes)
        .expect("the frozen OptiQ test config should decode as JSON");
    let mut quantization = json!({
        "group_size": 64,
        "bits": 4,
        "mode": "affine",
    });
    let quantized_module_names = expected_quantized_module_names();
    for (quantized_module_index, quantized_module_name) in quantized_module_names.iter().enumerate()
    {
        let quantization_bits = if quantized_module_index < 113 { 4 } else { 8 };
        quantization[quantized_module_name] = json!({"group_size": 64, "bits": quantization_bits});
    }
    config_value["quantization"] = quantization.clone();
    config_value["quantization_config"] = quantization;
    config_value["text_config"]["layer_types"] = json!(expected_layer_types());
    serde_json::to_vec(&config_value)
        .expect("the frozen OptiQ test config should serialize as JSON")
}

pub fn frozen_optiq_metadata_bytes() -> Vec<u8> {
    let config_value = serde_json::from_slice::<Value>(&frozen_ornith_1_0_optiq_config_bytes())
        .expect("the frozen OptiQ test config should decode as JSON");
    let mut measured_module_bits = config_value["quantization"]
        .as_object()
        .expect("the quantization map should be an object")
        .clone();
    for non_module_field_name in [
        "group_size",
        "bits",
        "mode",
        "language_model.model.embed_tokens",
        "language_model.lm_head",
    ] {
        measured_module_bits.remove(non_module_field_name);
    }
    serde_json::to_vec(&json!({
        "method": "optiq_mixed_precision_transferred",
        "base_model": "deepreinforce-ai/Ornith-1.0-35B",
        "reference": "bit map transferred from mlx-community/Qwen3.5-35B-A3B-OptiQ-4bit",
        "sensitivity_measured_on": "Qwen/Qwen3.5-35B-A3B",
        "target_bpw": 4.5,
        "achieved_bpw": 4.5131342941951385,
        "n_high_bits": 397,
        "n_low_bits": 113,
        "threshold": 0.0,
        "per_layer": measured_module_bits,
    }))
    .expect("the frozen OptiQ metadata should serialize as JSON")
}

/// The model document of the sparse mixed-precision shape before its
/// quantization map and layer types are added.
const SPARSE_MIXED_PRECISION_BASE_CONFIG_JSON: &[u8] = br#"
    {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "dtype": "bfloat16",
        "eos_token_id": [248046, 248044],
        "tie_word_embeddings": false,
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_act": "silu",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "rms_norm_eps": 0.000001,
            "rope_parameters": {
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000.0,
                "type": "default"
            },
            "partial_rotary_factor": 0.25,
            "attention_bias": false,
            "mlp_bias": false,
            "norm_topk_prob": true,
            "output_router_logits": false,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "full_attention_interval": 4,
            "linear_conv_kernel_dim": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512
        }
    }
    "#;

/// Freezes a sparse mixed-precision quantization configuration shape:
/// a 6-bit affine default with a sparse override map whose unlisted expected
/// modules (router gates) are stored as native floating point and
/// resolved through the shard-index scan.
pub fn frozen_sparse_mixed_precision_config_bytes() -> Vec<u8> {
    let mut config_value = serde_json::from_slice::<Value>(SPARSE_MIXED_PRECISION_BASE_CONFIG_JSON)
        .expect("the frozen test config should decode as JSON");
    let mut quantization = json!({
        "group_size": 64,
        "bits": 6,
        "mode": "affine",
    });
    lift_shared_experts_to_eight_bit(&mut quantization);
    widen_linear_attention_output_projections(&mut quantization);
    lift_sparse_attention_input_projections_to_eight_bit(&mut quantization);
    quantization["language_model.model.embed_tokens"] = json!({"group_size": 64, "bits": 8});
    quantization["language_model.lm_head"] = json!({"group_size": 64, "bits": 8});
    config_value["quantization"] = quantization.clone();
    config_value["quantization_config"] = quantization;
    config_value["text_config"]["layer_types"] = json!(expected_layer_types());
    serde_json::to_vec(&config_value).expect("the frozen test config should serialize as JSON")
}

/// Every layer lifts the shared expert to 8-bit; the gate stays at group 64
/// while its projections widen to group 128.
fn lift_shared_experts_to_eight_bit(quantization: &mut Value) {
    for decoder_layer_index in 0..40 {
        let layer_prefix = format!("language_model.model.layers.{decoder_layer_index}");
        quantization[format!("{layer_prefix}.mlp.shared_expert_gate")] =
            json!({"group_size": 64, "bits": 8});
        for projection_name in ["gate_proj", "up_proj", "down_proj"] {
            quantization[format!("{layer_prefix}.mlp.shared_expert.{projection_name}")] =
                json!({"group_size": 128, "bits": 8});
        }
    }
}

/// Linear-attention output projections all widen to group 128 at the default
/// 6-bit width.
fn widen_linear_attention_output_projections(quantization: &mut Value) {
    for decoder_layer_index in 0..40 {
        if decoder_layer_index % 4 == 3 {
            continue;
        }
        quantization
            [format!("language_model.model.layers.{decoder_layer_index}.linear_attn.out_proj")] =
            json!({"group_size": 128, "bits": 6});
    }
}

/// Only a sparse subset of attention input projections is lifted to 8-bit;
/// the remaining full-attention and linear-attention layers keep the 6-bit
/// group-64 default.
fn lift_sparse_attention_input_projections_to_eight_bit(quantization: &mut Value) {
    for decoder_layer_index in [3, 7, 31, 35, 39] {
        for projection_name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            quantization[format!(
                "language_model.model.layers.{decoder_layer_index}.self_attn.{projection_name}"
            )] = json!({"group_size": 64, "bits": 8});
        }
    }
    for decoder_layer_index in [0, 10] {
        for projection_name in ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"] {
            quantization[format!(
                "language_model.model.layers.{decoder_layer_index}.linear_attn.{projection_name}"
            )] = json!({"group_size": 64, "bits": 8});
        }
    }
}

pub fn frozen_ornith_1_0_optiq_vision_config_bytes() -> Vec<u8> {
    let mut config_value = serde_json::from_slice::<Value>(&frozen_ornith_1_0_optiq_config_bytes())
        .expect("the frozen OptiQ test config should decode as JSON");
    config_value["vision_config"] = json!({
        "deepstack_visual_indexes": [],
        "depth": 27,
        "dtype": "bfloat16",
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4304,
        "model_type": "qwen3_5_moe_vision",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2048,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2
    });
    serde_json::to_vec(&config_value)
        .expect("the frozen OptiQ vision test config should serialize as JSON")
}

pub fn expected_quantized_module_names() -> Vec<String> {
    let mut quantized_module_names = Vec::with_capacity(512);
    for decoder_layer_index in 0..40 {
        let layer_prefix = format!("language_model.model.layers.{decoder_layer_index}");
        if decoder_layer_index % 4 == 3 {
            for projection_name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                quantized_module_names.push(format!("{layer_prefix}.self_attn.{projection_name}"));
            }
        } else {
            for projection_name in [
                "in_proj_qkv",
                "in_proj_z",
                "in_proj_b",
                "in_proj_a",
                "out_proj",
            ] {
                quantized_module_names
                    .push(format!("{layer_prefix}.linear_attn.{projection_name}"));
            }
        }
        for module_suffix in [
            "mlp.gate",
            "mlp.switch_mlp.gate_proj",
            "mlp.switch_mlp.up_proj",
            "mlp.switch_mlp.down_proj",
            "mlp.shared_expert.gate_proj",
            "mlp.shared_expert.up_proj",
            "mlp.shared_expert.down_proj",
            "mlp.shared_expert_gate",
        ] {
            quantized_module_names.push(format!("{layer_prefix}.{module_suffix}"));
        }
    }
    quantized_module_names.push("language_model.model.embed_tokens".to_owned());
    quantized_module_names.push("language_model.lm_head".to_owned());
    quantized_module_names
}

pub fn expected_layer_types() -> Vec<&'static str> {
    (0..40)
        .map(|decoder_layer_index| {
            if decoder_layer_index % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect()
}

#[cfg(test)]
#[path = "frozen_ornith_1_0_fixture_tests.rs"]
mod tests;
