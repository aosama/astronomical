use serde_json::{Value, json};

use super::{
    expected_layer_types, frozen_sparse_mixed_precision_config_bytes,
    lift_shared_experts_to_eight_bit, lift_sparse_attention_input_projections_to_eight_bit,
    widen_linear_attention_output_projections,
};

fn override_count(quantization: &Value) -> usize {
    quantization
        .as_object()
        .expect("the quantization map should be an object")
        .len()
}

#[test]
fn should_lift_the_shared_expert_of_every_layer_to_eight_bit() {
    let mut quantization = json!({});
    lift_shared_experts_to_eight_bit(&mut quantization);

    // Four shared-expert modules per layer across all forty layers.
    assert_eq!(override_count(&quantization), 160);
    assert_eq!(
        quantization["language_model.model.layers.0.mlp.shared_expert_gate"],
        json!({"group_size": 64, "bits": 8})
    );
    assert_eq!(
        quantization["language_model.model.layers.39.mlp.shared_expert.down_proj"],
        json!({"group_size": 128, "bits": 8})
    );
}

#[test]
fn should_leave_existing_default_entries_untouched_when_lifting_shared_experts() {
    let mut quantization = json!({"group_size": 64, "bits": 6, "mode": "affine"});
    lift_shared_experts_to_eight_bit(&mut quantization);

    assert_eq!(quantization["bits"], json!(6));
    assert_eq!(quantization["mode"], json!("affine"));
    assert!(quantization["language_model.model.layers.0.mlp.gate"].is_null());
}

#[test]
fn should_widen_the_output_projection_of_linear_attention_layers_only() {
    let mut quantization = json!({});
    widen_linear_attention_output_projections(&mut quantization);

    // Every fourth layer is full attention, leaving thirty linear layers.
    assert_eq!(override_count(&quantization), 30);
    assert_eq!(
        quantization["language_model.model.layers.0.linear_attn.out_proj"],
        json!({"group_size": 128, "bits": 6})
    );
    assert_eq!(
        quantization["language_model.model.layers.38.linear_attn.out_proj"],
        json!({"group_size": 128, "bits": 6})
    );
    assert!(quantization["language_model.model.layers.3.linear_attn.out_proj"].is_null());
    assert!(quantization["language_model.model.layers.39.linear_attn.out_proj"].is_null());
}

#[test]
fn should_lift_only_the_listed_attention_input_projections() {
    let mut quantization = json!({});
    lift_sparse_attention_input_projections_to_eight_bit(&mut quantization);

    // Five full-attention layers with four projections each, plus two
    // linear-attention layers with four projections each.
    assert_eq!(override_count(&quantization), 28);
    assert_eq!(
        quantization["language_model.model.layers.39.self_attn.o_proj"],
        json!({"group_size": 64, "bits": 8})
    );
    assert_eq!(
        quantization["language_model.model.layers.10.linear_attn.in_proj_a"],
        json!({"group_size": 64, "bits": 8})
    );
    assert!(quantization["language_model.model.layers.11.self_attn.q_proj"].is_null());
    assert!(quantization["language_model.model.layers.1.linear_attn.in_proj_qkv"].is_null());
}

#[test]
fn should_freeze_the_sparse_config_with_a_six_bit_default_and_native_router_gates() {
    let config_value: Value = serde_json::from_slice(&frozen_sparse_mixed_precision_config_bytes())
        .expect("the frozen sparse config should decode as JSON");
    let quantization = &config_value["quantization"];

    assert_eq!(quantization["bits"], json!(6));
    assert_eq!(quantization["group_size"], json!(64));
    assert_eq!(
        quantization["language_model.lm_head"],
        json!({"group_size": 64, "bits": 8})
    );
    assert_eq!(config_value["quantization_config"], *quantization);
    assert_eq!(
        config_value["text_config"]["layer_types"],
        json!(expected_layer_types())
    );
    // Router gates carry no override: they resolve through the shard-index
    // scan as native floating point.
    assert!(quantization["language_model.model.layers.0.mlp.gate"].is_null());
}
