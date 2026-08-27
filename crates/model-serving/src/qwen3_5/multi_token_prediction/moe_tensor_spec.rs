//! MoE MTP tensor profiles: router, packed switch_mlp experts, and shared expert.

use crate::TensorProfile;
use crate::qwen3_5::Qwen3_5Config;
use crate::qwen3_5::artifacts::tensor_spec::append_qwen3_5_quantized_affine_tensor_profiles;
pub(crate) use crate::qwen3_5::artifacts::tensor_spec::qwen3_5_tensor_profile;

/// Appends quantized-affine tensor profiles for one MTP module.
/// Shared between MoE and dense paths to avoid circular module imports.
pub(crate) fn append_qwen3_5_mtp_affine_tensor_profiles(
    mtp_tensor_profiles: &mut Vec<TensorProfile>,
    module_name: &str,
    leading_dimensions: &[usize],
    input_dimension: usize,
    qwen3_5_config: &Qwen3_5Config,
) {
    let quantization_profile = qwen3_5_config.quantization_profile_for_module(module_name);
    append_qwen3_5_quantized_affine_tensor_profiles(
        mtp_tensor_profiles,
        module_name,
        leading_dimensions,
        input_dimension,
        quantization_profile,
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
    // The MoE gate may be quantized or unquantized depending on the
    // artifact's affine profile. It goes through the quantization-aware
    // profile generator so that packed U32 weights, scales, and biases
    // are included when quantized, or a single BF16 weight when native.
    append_qwen3_5_mtp_affine_tensor_profiles(
        mtp_tensor_profiles,
        &format!("{mtp_layer_prefix}.mlp.gate"),
        &[expert_count],
        hidden_size,
        qwen3_5_config,
    );
    // Packed switch_mlp is the resident MTP expert layout the executor already
    // knows. Sidecars that store per-expert 2D tensors omit these names; artifact
    // validation drops the packed profiles in that case. SSD streaming never
    // pages an MTP sparse layer.
    for (projection_name, output_dimension, input_dimension) in [
        ("gate_proj", expert_intermediate_size, hidden_size),
        ("up_proj", expert_intermediate_size, hidden_size),
        ("down_proj", hidden_size, expert_intermediate_size),
    ] {
        append_qwen3_5_mtp_affine_tensor_profiles(
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
        append_qwen3_5_mtp_affine_tensor_profiles(
            mtp_tensor_profiles,
            &format!("{mtp_layer_prefix}.mlp.shared_expert.{projection_name}"),
            &[output_dimension],
            input_dimension,
            qwen3_5_config,
        );
    }
    append_qwen3_5_mtp_affine_tensor_profiles(
        mtp_tensor_profiles,
        &format!("{mtp_layer_prefix}.mlp.shared_expert_gate"),
        &[1],
        hidden_size,
        qwen3_5_config,
    );
}
