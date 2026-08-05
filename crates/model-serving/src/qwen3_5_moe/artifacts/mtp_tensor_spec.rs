use std::collections::BTreeSet;

use crate::qwen3_5::Qwen3_5Config;
use crate::qwen3_5::artifacts::mtp_tensor_namespace::{
    append_qwen3_5_mtp_quantized_affine_tensor_names,
    append_qwen3_5_mtp_quantized_affine_tensor_profiles,
};
use crate::qwen3_5::artifacts::tensor_spec::qwen3_5_tensor_profile;
use crate::{TensorDtype, TensorProfile};

const SPARSE_MTP_AFFINE_PROJECTION_NAMES: [&str; 4] = [
    "mlp.switch_mlp.gate_proj",
    "mlp.switch_mlp.up_proj",
    "mlp.switch_mlp.down_proj",
    "mlp.shared_expert_gate",
];
const SHARED_EXPERT_MTP_AFFINE_PROJECTION_NAMES: [&str; 3] = [
    "mlp.shared_expert.gate_proj",
    "mlp.shared_expert.up_proj",
    "mlp.shared_expert.down_proj",
];

pub(crate) fn append_qwen3_5_moe_mtp_tensor_names(
    expected_mtp_tensor_names: &mut BTreeSet<String>,
    mtp_layer_prefix: &str,
) {
    expected_mtp_tensor_names.insert(format!("{mtp_layer_prefix}.mlp.gate.weight"));
    append_qwen3_5_mtp_quantized_affine_tensor_names(
        expected_mtp_tensor_names,
        mtp_layer_prefix,
        &SPARSE_MTP_AFFINE_PROJECTION_NAMES,
    );
    append_qwen3_5_mtp_quantized_affine_tensor_names(
        expected_mtp_tensor_names,
        mtp_layer_prefix,
        &SHARED_EXPERT_MTP_AFFINE_PROJECTION_NAMES,
    );
}

pub(crate) fn append_qwen3_5_moe_mtp_tensor_profiles(
    mtp_tensor_profiles: &mut Vec<TensorProfile>,
    mtp_layer_prefix: &str,
    hidden_size: usize,
    qwen3_5_config: &Qwen3_5Config,
) {
    let expert_count = qwen3_5_config.expert_count() as usize;
    let expert_intermediate_size = qwen3_5_config.expert_intermediate_size() as usize;
    let shared_expert_intermediate_size = qwen3_5_config.shared_expert_intermediate_size() as usize;
    mtp_tensor_profiles.push(qwen3_5_tensor_profile(
        format!("{mtp_layer_prefix}.mlp.gate.weight"),
        TensorDtype::ModelFloat,
        vec![expert_count, hidden_size],
    ));
    for (projection_name, output_dimension, input_dimension) in [
        ("gate_proj", expert_intermediate_size, hidden_size),
        ("up_proj", expert_intermediate_size, hidden_size),
        ("down_proj", hidden_size, expert_intermediate_size),
    ] {
        append_qwen3_5_mtp_quantized_affine_tensor_profiles(
            mtp_tensor_profiles,
            &format!("{mtp_layer_prefix}.mlp.switch_mlp.{projection_name}"),
            &[expert_count, output_dimension],
            input_dimension,
            qwen3_5_config,
        );
    }
    for (projection_name, output_dimension, input_dimension) in [
        ("gate_proj", shared_expert_intermediate_size, hidden_size),
        ("up_proj", shared_expert_intermediate_size, hidden_size),
        ("down_proj", hidden_size, shared_expert_intermediate_size),
    ] {
        append_qwen3_5_mtp_quantized_affine_tensor_profiles(
            mtp_tensor_profiles,
            &format!("{mtp_layer_prefix}.mlp.shared_expert.{projection_name}"),
            &[output_dimension],
            input_dimension,
            qwen3_5_config,
        );
    }
    append_qwen3_5_mtp_quantized_affine_tensor_profiles(
        mtp_tensor_profiles,
        &format!("{mtp_layer_prefix}.mlp.shared_expert_gate"),
        &[1],
        hidden_size,
        qwen3_5_config,
    );
}
