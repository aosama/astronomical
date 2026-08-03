use std::collections::BTreeSet;

use crate::{TensorDtype, TensorProfile};

use super::Qwen3_5MoEConfig;
const MTP_ROOT_TENSOR_NAMES: [&str; 4] = [
    "fc.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
    "norm.weight",
];
const MTP_NORMALIZATION_TENSOR_NAMES: [&str; 4] = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
];
const MTP_ATTENTION_AFFINE_PROJECTION_NAMES: [&str; 4] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
];
const MTP_SPARSE_AFFINE_PROJECTION_NAMES: [&str; 4] = [
    "mlp.switch_mlp.gate_proj",
    "mlp.switch_mlp.up_proj",
    "mlp.switch_mlp.down_proj",
    "mlp.shared_expert_gate",
];
const MTP_SHARED_EXPERT_PROJECTION_NAMES: [&str; 3] = [
    "mlp.shared_expert.gate_proj",
    "mlp.shared_expert.up_proj",
    "mlp.shared_expert.down_proj",
];
const MTP_DENSE_PROJECTION_NAMES: [&str; 3] = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"];

/// Returns the stored one-layer affine Qwen MTP namespace used by oQ4e artifacts.
#[must_use]
pub fn qwen3_5_moe_quantized_mtp_tensor_names(
    qwen3_5_moe_config: &Qwen3_5MoEConfig,
) -> BTreeSet<String> {
    let mut expected_tensor_names = BTreeSet::new();
    for root_tensor_name in MTP_ROOT_TENSOR_NAMES {
        expected_tensor_names.insert(format!("language_model.mtp.{root_tensor_name}"));
    }
    for normalization_tensor_name in MTP_NORMALIZATION_TENSOR_NAMES {
        expected_tensor_names.insert(format!(
            "language_model.mtp.layers.0.{normalization_tensor_name}"
        ));
    }
    let affine_projection_names: Vec<&str> = if qwen3_5_moe_config.is_dense_model() {
        MTP_ATTENTION_AFFINE_PROJECTION_NAMES
            .iter()
            .copied()
            .chain(MTP_DENSE_PROJECTION_NAMES)
            .collect()
    } else {
        expected_tensor_names.insert("language_model.mtp.layers.0.mlp.gate.weight".to_owned());
        MTP_ATTENTION_AFFINE_PROJECTION_NAMES
            .into_iter()
            .chain(MTP_SPARSE_AFFINE_PROJECTION_NAMES)
            .chain(MTP_SHARED_EXPERT_PROJECTION_NAMES)
            .collect()
    };
    for affine_projection_name in affine_projection_names {
        for affine_component_name in ["weight", "scales", "biases"] {
            expected_tensor_names.insert(format!(
                "language_model.mtp.layers.0.{affine_projection_name}.{affine_component_name}"
            ));
        }
    }
    expected_tensor_names
}

/// Returns exact tensor profiles for Qwen's stored one-layer affine MTP head.
#[must_use]
pub fn qwen3_5_moe_mtp_tensor_profiles(
    qwen3_5_moe_config: &Qwen3_5MoEConfig,
) -> Vec<TensorProfile> {
    if qwen3_5_moe_config.mtp_layer_count() == 0 {
        return Vec::new();
    }

    let hidden_size = qwen3_5_moe_config.hidden_size() as usize;
    let head_dimension = qwen3_5_moe_config.head_dimension() as usize;
    let query_head_count = qwen3_5_moe_config.query_head_count() as usize;
    let key_value_head_count = qwen3_5_moe_config.key_value_head_count() as usize;
    let mut mtp_tensor_profiles = Vec::new();
    let mtp_prefix = "language_model.mtp";

    for normalization_name in [
        "pre_fc_norm_embedding.weight",
        "pre_fc_norm_hidden.weight",
        "norm.weight",
    ] {
        mtp_tensor_profiles.push(tensor_profile(
            format!("{mtp_prefix}.{normalization_name}"),
            TensorDtype::BFloat16,
            vec![hidden_size],
        ));
    }
    mtp_tensor_profiles.push(tensor_profile(
        format!("{mtp_prefix}.fc.weight"),
        TensorDtype::BFloat16,
        vec![hidden_size, hidden_size * 2],
    ));

    let mtp_layer_prefix = format!("{mtp_prefix}.layers.0");
    mtp_tensor_profiles.push(tensor_profile(
        format!("{mtp_layer_prefix}.input_layernorm.weight"),
        TensorDtype::BFloat16,
        vec![hidden_size],
    ));
    for (projection_name, output_dimension, input_dimension) in [
        ("q_proj", query_head_count * head_dimension * 2, hidden_size),
        ("k_proj", key_value_head_count * head_dimension, hidden_size),
        ("v_proj", key_value_head_count * head_dimension, hidden_size),
        ("o_proj", hidden_size, query_head_count * head_dimension),
    ] {
        push_affine_tensor_profiles(
            &mut mtp_tensor_profiles,
            &format!("{mtp_layer_prefix}.self_attn.{projection_name}"),
            &[output_dimension],
            input_dimension,
            qwen3_5_moe_config,
        );
    }
    for normalization_name in ["q_norm.weight", "k_norm.weight"] {
        mtp_tensor_profiles.push(tensor_profile(
            format!("{mtp_layer_prefix}.self_attn.{normalization_name}"),
            TensorDtype::BFloat16,
            vec![head_dimension],
        ));
    }
    mtp_tensor_profiles.push(tensor_profile(
        format!("{mtp_layer_prefix}.post_attention_layernorm.weight"),
        TensorDtype::BFloat16,
        vec![hidden_size],
    ));
    if qwen3_5_moe_config.is_dense_model() {
        let dense_intermediate_size = qwen3_5_moe_config.dense_intermediate_size() as usize;
        for (projection_name, output_dimension, input_dimension) in [
            ("gate_proj", dense_intermediate_size, hidden_size),
            ("up_proj", dense_intermediate_size, hidden_size),
            ("down_proj", hidden_size, dense_intermediate_size),
        ] {
            push_affine_tensor_profiles(
                &mut mtp_tensor_profiles,
                &format!("{mtp_layer_prefix}.mlp.{projection_name}"),
                &[output_dimension],
                input_dimension,
                qwen3_5_moe_config,
            );
        }
    } else {
        let expert_count = qwen3_5_moe_config.expert_count() as usize;
        let expert_intermediate_size = qwen3_5_moe_config.expert_intermediate_size() as usize;
        let shared_expert_intermediate_size =
            qwen3_5_moe_config.shared_expert_intermediate_size() as usize;
        mtp_tensor_profiles.push(tensor_profile(
            format!("{mtp_layer_prefix}.mlp.gate.weight"),
            TensorDtype::BFloat16,
            vec![expert_count, hidden_size],
        ));
        for (projection_name, output_dimension, input_dimension) in [
            ("gate_proj", expert_intermediate_size, hidden_size),
            ("up_proj", expert_intermediate_size, hidden_size),
            ("down_proj", hidden_size, expert_intermediate_size),
        ] {
            push_affine_tensor_profiles(
                &mut mtp_tensor_profiles,
                &format!("{mtp_layer_prefix}.mlp.switch_mlp.{projection_name}"),
                &[expert_count, output_dimension],
                input_dimension,
                qwen3_5_moe_config,
            );
        }
        for (projection_name, output_dimension, input_dimension) in [
            ("gate_proj", shared_expert_intermediate_size, hidden_size),
            ("up_proj", shared_expert_intermediate_size, hidden_size),
            ("down_proj", hidden_size, shared_expert_intermediate_size),
        ] {
            push_affine_tensor_profiles(
                &mut mtp_tensor_profiles,
                &format!("{mtp_layer_prefix}.mlp.shared_expert.{projection_name}"),
                &[output_dimension],
                input_dimension,
                qwen3_5_moe_config,
            );
        }
        push_affine_tensor_profiles(
            &mut mtp_tensor_profiles,
            &format!("{mtp_layer_prefix}.mlp.shared_expert_gate"),
            &[1],
            hidden_size,
            qwen3_5_moe_config,
        );
    }
    mtp_tensor_profiles
}

fn push_affine_tensor_profiles(
    mtp_tensor_profiles: &mut Vec<TensorProfile>,
    module_name: &str,
    leading_dimensions: &[usize],
    input_dimension: usize,
    qwen3_5_moe_config: &Qwen3_5MoEConfig,
) {
    let quantization_profile = qwen3_5_moe_config.quantization_profile_for_module(module_name);
    let quantization_bits = quantization_profile.bits as usize;
    let quantization_group_size = quantization_profile.group_size as usize;
    let mut packed_weight_shape = leading_dimensions.to_vec();
    packed_weight_shape.push(input_dimension * quantization_bits / 32);
    let mut scale_shape = leading_dimensions.to_vec();
    scale_shape.push(input_dimension / quantization_group_size);
    mtp_tensor_profiles.push(tensor_profile(
        format!("{module_name}.weight"),
        TensorDtype::UInt32,
        packed_weight_shape,
    ));
    mtp_tensor_profiles.push(tensor_profile(
        format!("{module_name}.scales"),
        TensorDtype::BFloat16,
        scale_shape.clone(),
    ));
    mtp_tensor_profiles.push(tensor_profile(
        format!("{module_name}.biases"),
        TensorDtype::BFloat16,
        scale_shape,
    ));
}

fn tensor_profile(name: String, dtype: TensorDtype, shape: Vec<usize>) -> TensorProfile {
    TensorProfile { name, dtype, shape }
}
