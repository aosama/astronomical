//! Descriptor matrices for dense and resident-MoE complete-model references.

use astronomical_runtime_integration::MlxDtype;
use serde_json::{Map, Value, json};

pub(super) struct ReferenceRow {
    pub(super) row_name: &'static str,
    pub(super) target_config: Value,
    pub(super) activation_dtype: MlxDtype,
    pub(super) prefill_token_ids: Vec<u32>,
    pub(super) decode_token_ids: Vec<u32>,
    pub(super) tolerance: f32,
    pub(super) has_attention_gate: bool,
    pub(super) has_sliding_attention: bool,
    pub(super) has_sparse_feed_forward: bool,
    pub(super) has_shared_expert: bool,
    pub(super) has_correction_bias: bool,
    pub(super) expects_assignment_sort: bool,
    pub(super) expected_affine_profiles: Vec<(i32, i32)>,
}

pub(super) fn generic_rows() -> Vec<ReferenceRow> {
    vec![
        dense_row(
            "generic_full",
            &["full"],
            &["none"],
            &[4],
            "float32",
            true,
            4,
        ),
        dense_row(
            "generic_sliding",
            &["sliding"],
            &["per_element"],
            &[4],
            "float16",
            false,
            2,
        ),
        dense_row(
            "generic_mixed",
            &["sliding", "full", "sliding"],
            &["none", "per_head", "per_element"],
            &[4, 6, 4],
            "bfloat16",
            false,
            2,
        ),
    ]
}

pub(super) fn generic_moe_rows() -> Vec<ReferenceRow> {
    vec![
        moe_row(
            "native_sparse",
            &["full"],
            &["sparse"],
            "float32",
            3,
            2,
            0,
            false,
            false,
            0.0,
            1.5,
            false,
            vec![1, 2],
        ),
        moe_row(
            "native_mixed_shared",
            &["sliding", "full", "sliding"],
            &["dense", "sparse", "dense"],
            "float16",
            4,
            2,
            8,
            true,
            false,
            2.0,
            2.5,
            true,
            vec![1, 2, 3],
        ),
        moe_row(
            "native_input_weighted",
            &["full"],
            &["sparse"],
            "bfloat16",
            4,
            2,
            8,
            false,
            true,
            0.0,
            1.25,
            true,
            vec![2, 3],
        ),
        moe_row(
            "native_sorted_threshold",
            &["sliding"],
            &["sparse"],
            "float32",
            10,
            8,
            0,
            true,
            false,
            0.0,
            1.0,
            false,
            vec![0, 1, 2, 3, 4, 5, 6, 7],
        ),
    ]
}

pub(super) fn named_rows() -> Vec<ReferenceRow> {
    vec![
        named_row(
            "xs_compact",
            40,
            64,
            8,
            8,
            "bfloat16",
            64,
            &[2, 3, 4, 8],
            false,
            32.0,
            64.0,
            1.346_573_6,
        ),
        named_row(
            "s_compact",
            48,
            72,
            10,
            10,
            "float16",
            128,
            &[2, 3, 4, 6, 8],
            true,
            128.0,
            32.0,
            1.485_203,
        ),
    ]
}

#[allow(clippy::too_many_arguments)]
fn dense_row(
    row_name: &'static str,
    layer_types: &[&str],
    gating_types: &[&str],
    query_head_counts: &[u32],
    dtype_name: &str,
    has_tied_embeddings: bool,
    sliding_window: u32,
) -> ReferenceRow {
    let layer_count = layer_types.len();
    ReferenceRow {
        row_name,
        target_config: base_config(
            layer_types,
            gating_types,
            query_head_counts,
            &vec!["dense"; layer_count],
            dtype_name,
            has_tied_embeddings,
            sliding_window,
            2,
            16,
        ),
        activation_dtype: dtype(dtype_name),
        prefill_token_ids: vec![1, 2],
        decode_token_ids: vec![3],
        tolerance: if dtype_name == "float32" { 5e-4 } else { 2e-2 },
        has_attention_gate: gating_types.iter().any(|gating| *gating != "none"),
        has_sliding_attention: layer_types.contains(&"sliding"),
        has_sparse_feed_forward: false,
        has_shared_expert: false,
        has_correction_bias: false,
        expects_assignment_sort: false,
        expected_affine_profiles: Vec::new(),
    }
}

#[allow(clippy::too_many_arguments)]
fn moe_row(
    row_name: &'static str,
    layer_types: &[&str],
    feed_forward_types: &[&str],
    dtype_name: &str,
    expert_count: u32,
    experts_per_token: u32,
    shared_intermediate_size: u32,
    normalizes_top_k: bool,
    weights_router_input: bool,
    router_softcap: f64,
    routed_scale: f64,
    has_correction_bias: bool,
    prefill_token_ids: Vec<u32>,
) -> ReferenceRow {
    let layer_count = layer_types.len();
    let mut config = base_config(
        layer_types,
        &vec!["none"; layer_count],
        &vec![4; layer_count],
        feed_forward_types,
        dtype_name,
        false,
        4,
        2,
        8,
    );
    add_moe_config(
        &mut config,
        expert_count,
        experts_per_token,
        8,
        shared_intermediate_size,
        normalizes_top_k,
        weights_router_input,
        router_softcap,
        routed_scale,
    );
    let assignment_count = prefill_token_ids.len() as u32 * experts_per_token;
    ReferenceRow {
        row_name,
        target_config: config,
        activation_dtype: dtype(dtype_name),
        prefill_token_ids,
        decode_token_ids: vec![5],
        tolerance: if dtype_name == "float32" { 8e-4 } else { 3e-2 },
        has_attention_gate: false,
        has_sliding_attention: layer_types.contains(&"sliding"),
        has_sparse_feed_forward: true,
        has_shared_expert: shared_intermediate_size > 0,
        has_correction_bias,
        expects_assignment_sort: assignment_count >= 64 && !weights_router_input,
        expected_affine_profiles: Vec::new(),
    }
}

#[allow(clippy::too_many_arguments)]
fn named_row(
    row_name: &'static str,
    layer_count: usize,
    sliding_query_heads: u32,
    experts_per_token: u32,
    expert_count: u32,
    dtype_name: &str,
    group_size: i32,
    affine_bits: &[i32],
    uses_wrapped_overrides: bool,
    yarn_factor: f64,
    yarn_beta_fast: f64,
    yarn_attention_factor: f64,
) -> ReferenceRow {
    let layer_types = (0..layer_count)
        .map(|layer_index| {
            if layer_index % 4 == 0 {
                "full"
            } else {
                "sliding"
            }
        })
        .collect::<Vec<_>>();
    let query_heads = layer_types
        .iter()
        .map(|layer_type| {
            if *layer_type == "full" {
                48
            } else {
                sliding_query_heads
            }
        })
        .collect::<Vec<_>>();
    let feed_forward_types = (0..layer_count)
        .map(|layer_index| if layer_index == 0 { "dense" } else { "sparse" })
        .collect::<Vec<_>>();
    let mut config = base_config(
        &layer_types,
        &vec!["per_head"; layer_count],
        &query_heads,
        &feed_forward_types,
        dtype_name,
        false,
        512,
        16,
        group_size,
    );
    config["num_key_value_heads"] = json!(8);
    add_moe_config(
        &mut config,
        expert_count,
        experts_per_token,
        group_size as u32,
        group_size as u32,
        true,
        false,
        0.0,
        2.5,
    );
    config["rope_parameters"] = json!({
        "full_attention": {
            "rope_type": "yarn", "rope_theta": 500000.0, "factor": yarn_factor,
            "original_max_position_embeddings": 8192, "beta_slow": 1.0,
            "beta_fast": yarn_beta_fast, "attention_factor": yarn_attention_factor,
            "partial_rotary_factor": 0.5
        },
        "sliding_attention": {
            "rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0
        }
    });
    let quantization = affine_quantization(group_size, affine_bits, uses_wrapped_overrides);
    config["quantization"] = quantization.clone();
    config["quantization_config"] = quantization;
    ReferenceRow {
        row_name,
        target_config: config,
        activation_dtype: dtype(dtype_name),
        prefill_token_ids: vec![1],
        decode_token_ids: vec![2],
        // Packed gathered kernels and the dequantized dense oracle accumulate
        // differently across 40/48 layers; the established named-row bound is 3%.
        tolerance: 3e-2,
        has_attention_gate: true,
        has_sliding_attention: true,
        has_sparse_feed_forward: true,
        has_shared_expert: true,
        has_correction_bias: false,
        expects_assignment_sort: false,
        expected_affine_profiles: affine_bits.iter().map(|bits| (*bits, group_size)).collect(),
    }
}

fn affine_quantization(
    group_size: i32,
    affine_bits: &[i32],
    uses_wrapped_overrides: bool,
) -> Value {
    let namespace_prefix = if uses_wrapped_overrides {
        "language_model."
    } else {
        ""
    };
    let mut quantization = Map::new();
    quantization.insert("bits".to_owned(), json!(affine_bits[0]));
    quantization.insert("group_size".to_owned(), json!(group_size));
    quantization.insert("mode".to_owned(), json!("affine"));
    for (profile_index, bits) in affine_bits.iter().copied().enumerate().skip(1) {
        let projection_name = match profile_index % 4 {
            1 => "gate_proj",
            2 => "up_proj",
            3 => "down_proj",
            _ => "gate_proj",
        };
        let layer_index = 1 + profile_index / 4;
        quantization.insert(
            format!(
                "{namespace_prefix}model.layers.{layer_index}.mlp.switch_mlp.{projection_name}"
            ),
            json!({"bits": bits, "group_size": group_size, "mode": "affine"}),
        );
    }
    Value::Object(quantization)
}

#[allow(clippy::too_many_arguments)]
fn add_moe_config(
    config: &mut Value,
    expert_count: u32,
    experts_per_token: u32,
    expert_intermediate_size: u32,
    shared_intermediate_size: u32,
    normalizes_top_k: bool,
    weights_router_input: bool,
    router_softcap: f64,
    routed_scale: f64,
) {
    config["num_experts"] = json!(expert_count);
    config["num_experts_per_tok"] = json!(experts_per_token);
    config["moe_intermediate_size"] = json!(expert_intermediate_size);
    config["shared_expert_intermediate_size"] = json!(shared_intermediate_size);
    config["norm_topk_prob"] = json!(normalizes_top_k);
    config["moe_routed_scaling_factor"] = json!(routed_scale);
    config["moe_apply_router_weight_on_input"] = json!(weights_router_input);
    if router_softcap > 0.0 {
        config["moe_router_logit_softcapping"] = json!(router_softcap);
    }
}

#[allow(clippy::too_many_arguments)]
fn base_config(
    layer_types: &[&str],
    gating_types: &[&str],
    query_head_counts: &[u32],
    feed_forward_types: &[&str],
    dtype_name: &str,
    tied: bool,
    sliding_window: u32,
    head_dimension: u32,
    hidden_size: i32,
) -> Value {
    json!({
        "architectures": ["LagunaForCausalLM"], "model_type": "laguna",
        "vocab_size": 12, "hidden_size": hidden_size, "intermediate_size": hidden_size,
        "num_hidden_layers": layer_types.len(), "num_attention_heads": query_head_counts[0],
        "num_attention_heads_per_layer": query_head_counts,
        "num_key_value_heads": 2, "head_dim": head_dimension,
        "max_position_embeddings": 32768, "rms_norm_eps": 0.00001,
        "tie_word_embeddings": tied, "torch_dtype": dtype_name,
        "layer_types": layer_types, "sliding_window": sliding_window,
        "mlp_layer_types": feed_forward_types, "gating_types": gating_types,
        "rope_parameters": {
            "rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0
        }
    })
}

fn dtype(dtype_name: &str) -> MlxDtype {
    match dtype_name {
        "float32" => MlxDtype::Float32,
        "float16" => MlxDtype::Float16,
        "bfloat16" => MlxDtype::BFloat16,
        _ => unreachable!("reference dtype is fixed by the row"),
    }
}
