use std::collections::BTreeSet;

use crate::qwen3_5::dense::mtp_tensor_spec::{
    append_qwen3_5_dense_mtp_tensor_names, append_qwen3_5_dense_mtp_tensor_profiles,
};
use crate::qwen3_5_moe::artifacts::mtp_tensor_spec::{
    append_qwen3_5_moe_mtp_tensor_names, append_qwen3_5_moe_mtp_tensor_profiles,
};
use crate::{TensorDtype, TensorProfile};

use super::tensor_spec::qwen3_5_tensor_profile;
use super::{Qwen3_5Config, Qwen3_5FeedForwardArchitecture};

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

/// Returns the stored one-layer affine Qwen MTP namespace used by oQ4e artifacts.
#[must_use]
pub fn qwen3_5_quantized_mtp_tensor_names(qwen3_5_config: &Qwen3_5Config) -> BTreeSet<String> {
    let mut expected_mtp_tensor_names = BTreeSet::new();
    for root_tensor_name in MTP_ROOT_TENSOR_NAMES {
        expected_mtp_tensor_names.insert(format!("language_model.mtp.{root_tensor_name}"));
    }
    let mtp_layer_prefix = "language_model.mtp.layers.0";
    for normalization_tensor_name in MTP_NORMALIZATION_TENSOR_NAMES {
        expected_mtp_tensor_names.insert(format!("{mtp_layer_prefix}.{normalization_tensor_name}"));
    }
    append_qwen3_5_mtp_quantized_affine_tensor_names(
        &mut expected_mtp_tensor_names,
        mtp_layer_prefix,
        &MTP_ATTENTION_AFFINE_PROJECTION_NAMES,
    );
    match qwen3_5_config.feed_forward_architecture() {
        Qwen3_5FeedForwardArchitecture::Dense => {
            append_qwen3_5_dense_mtp_tensor_names(&mut expected_mtp_tensor_names, mtp_layer_prefix)
        }
        Qwen3_5FeedForwardArchitecture::MixtureOfExperts => {
            append_qwen3_5_moe_mtp_tensor_names(&mut expected_mtp_tensor_names, mtp_layer_prefix);
        }
    }
    expected_mtp_tensor_names
}

/// Returns exact tensor profiles for Qwen's stored one-layer affine MTP head.
#[must_use]
pub fn qwen3_5_mtp_tensor_profiles(qwen3_5_config: &Qwen3_5Config) -> Vec<TensorProfile> {
    if qwen3_5_config.mtp_layer_count() == 0 {
        return Vec::new();
    }

    let hidden_size = qwen3_5_config.hidden_size() as usize;
    let head_dimension = qwen3_5_config.head_dimension() as usize;
    let query_head_count = qwen3_5_config.query_head_count() as usize;
    let key_value_head_count = qwen3_5_config.key_value_head_count() as usize;
    let mut mtp_tensor_profiles = Vec::new();
    let mtp_prefix = "language_model.mtp";

    for normalization_name in [
        "pre_fc_norm_embedding.weight",
        "pre_fc_norm_hidden.weight",
        "norm.weight",
    ] {
        mtp_tensor_profiles.push(qwen3_5_tensor_profile(
            format!("{mtp_prefix}.{normalization_name}"),
            TensorDtype::BFloat16,
            vec![hidden_size],
        ));
    }
    mtp_tensor_profiles.push(qwen3_5_tensor_profile(
        format!("{mtp_prefix}.fc.weight"),
        TensorDtype::BFloat16,
        vec![hidden_size, hidden_size * 2],
    ));

    let mtp_layer_prefix = format!("{mtp_prefix}.layers.0");
    mtp_tensor_profiles.push(qwen3_5_tensor_profile(
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
        append_qwen3_5_mtp_quantized_affine_tensor_profiles(
            &mut mtp_tensor_profiles,
            &format!("{mtp_layer_prefix}.self_attn.{projection_name}"),
            &[output_dimension],
            input_dimension,
            qwen3_5_config,
        );
    }
    for normalization_name in ["q_norm.weight", "k_norm.weight"] {
        mtp_tensor_profiles.push(qwen3_5_tensor_profile(
            format!("{mtp_layer_prefix}.self_attn.{normalization_name}"),
            TensorDtype::BFloat16,
            vec![head_dimension],
        ));
    }
    mtp_tensor_profiles.push(qwen3_5_tensor_profile(
        format!("{mtp_layer_prefix}.post_attention_layernorm.weight"),
        TensorDtype::BFloat16,
        vec![hidden_size],
    ));
    match qwen3_5_config.feed_forward_architecture() {
        Qwen3_5FeedForwardArchitecture::Dense => append_qwen3_5_dense_mtp_tensor_profiles(
            &mut mtp_tensor_profiles,
            &mtp_layer_prefix,
            hidden_size,
            qwen3_5_config,
        ),
        Qwen3_5FeedForwardArchitecture::MixtureOfExperts => {
            append_qwen3_5_moe_mtp_tensor_profiles(
                &mut mtp_tensor_profiles,
                &mtp_layer_prefix,
                hidden_size,
                qwen3_5_config,
            );
        }
    }
    mtp_tensor_profiles
}

pub(crate) fn append_qwen3_5_mtp_quantized_affine_tensor_names(
    expected_mtp_tensor_names: &mut BTreeSet<String>,
    mtp_layer_prefix: &str,
    affine_projection_names: &[&str],
) {
    for affine_projection_name in affine_projection_names {
        for affine_component_name in ["weight", "scales", "biases"] {
            expected_mtp_tensor_names.insert(format!(
                "{mtp_layer_prefix}.{affine_projection_name}.{affine_component_name}"
            ));
        }
    }
}

pub(crate) fn append_qwen3_5_mtp_quantized_affine_tensor_profiles(
    mtp_tensor_profiles: &mut Vec<TensorProfile>,
    module_name: &str,
    leading_dimensions: &[usize],
    input_dimension: usize,
    qwen3_5_config: &Qwen3_5Config,
) {
    let quantization_profile = qwen3_5_config.quantization_profile_for_module(module_name);
    let quantization_bits = quantization_profile.bits as usize;
    let quantization_group_size = quantization_profile.group_size as usize;
    let mut packed_weight_shape = leading_dimensions.to_vec();
    packed_weight_shape.push(input_dimension * quantization_bits / 32);
    let mut scale_shape = leading_dimensions.to_vec();
    scale_shape.push(input_dimension / quantization_group_size);
    mtp_tensor_profiles.push(qwen3_5_tensor_profile(
        format!("{module_name}.weight"),
        TensorDtype::UInt32,
        packed_weight_shape,
    ));
    mtp_tensor_profiles.push(qwen3_5_tensor_profile(
        format!("{module_name}.scales"),
        TensorDtype::BFloat16,
        scale_shape.clone(),
    ));
    mtp_tensor_profiles.push(qwen3_5_tensor_profile(
        format!("{module_name}.biases"),
        TensorDtype::BFloat16,
        scale_shape,
    ));
}
