use std::collections::BTreeSet;

use crate::artifact_validation::TensorProfile;
use crate::qwen3_5::Qwen3_5Config;
use crate::qwen3_5::artifacts::mtp_tensor_namespace::{
    append_qwen3_5_mtp_quantized_affine_tensor_names,
    append_qwen3_5_mtp_quantized_affine_tensor_profiles,
};

const DENSE_MTP_AFFINE_PROJECTION_NAMES: [&str; 3] =
    ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"];

pub(crate) fn append_qwen3_5_dense_mtp_tensor_names(
    expected_mtp_tensor_names: &mut BTreeSet<String>,
    mtp_layer_prefix: &str,
) {
    append_qwen3_5_mtp_quantized_affine_tensor_names(
        expected_mtp_tensor_names,
        mtp_layer_prefix,
        &DENSE_MTP_AFFINE_PROJECTION_NAMES,
    );
}

pub(crate) fn append_qwen3_5_dense_mtp_tensor_profiles(
    mtp_tensor_profiles: &mut Vec<TensorProfile>,
    mtp_layer_prefix: &str,
    hidden_size: usize,
    qwen3_5_config: &Qwen3_5Config,
) {
    let dense_intermediate_size = qwen3_5_config.dense_intermediate_size() as usize;
    for (projection_name, output_dimension, input_dimension) in [
        ("gate_proj", dense_intermediate_size, hidden_size),
        ("up_proj", dense_intermediate_size, hidden_size),
        ("down_proj", hidden_size, dense_intermediate_size),
    ] {
        append_qwen3_5_mtp_quantized_affine_tensor_profiles(
            mtp_tensor_profiles,
            &format!("{mtp_layer_prefix}.mlp.{projection_name}"),
            &[output_dimension],
            input_dimension,
            qwen3_5_config,
        );
    }
}
