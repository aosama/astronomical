use std::collections::BTreeSet;

use crate::{TensorDtype, TensorProfile};

use super::quantizations::optiq::OptiQQuantizationProfile;
use super::{Qwen3_5MoEArtifactError, Qwen3_5MoEConfig};

const PACKING_WORD_BITS: usize = 32;

/// Generates the executable language-tensor metadata for the pinned Qwen3.5-MoE MLX artifact.
///
/// This describes tensor names, safetensors dtypes, and packed quantized shapes only. It does not
/// allocate MLX arrays and intentionally excludes the 333 non-executable vision tensors.
///
/// Supports both uniform group-size models (oQ4, all gs=64) and mixed group-size models
/// (oQ6e, gs=64 and gs=128) by looking up per-module quantization profiles.
#[must_use]
pub fn qwen3_5_moe_language_tensor_profiles(
    qwen3_5_moe_config: &Qwen3_5MoEConfig,
) -> Vec<TensorProfile> {
    let hidden_size = qwen3_5_moe_config.hidden_size() as usize;
    let layer_count = qwen3_5_moe_config.layer_count() as usize;
    let vocabulary_size = qwen3_5_moe_config.vocabulary_size() as usize;
    let mut tensor_profiles = Vec::new();
    push_quantized_affine(
        &mut tensor_profiles,
        "language_model.model.embed_tokens",
        &[vocabulary_size],
        hidden_size,
        quantization_profile_for_module(qwen3_5_moe_config, "language_model.model.embed_tokens"),
    );
    for decoder_layer_index in 0..layer_count {
        push_decoder_layer_tensor_profiles(
            &mut tensor_profiles,
            decoder_layer_index,
            hidden_size,
            qwen3_5_moe_config,
        );
    }
    tensor_profiles.push(tensor_profile(
        "language_model.model.norm.weight".to_owned(),
        TensorDtype::BFloat16,
        vec![hidden_size],
    ));
    push_quantized_affine(
        &mut tensor_profiles,
        "language_model.lm_head",
        &[vocabulary_size],
        hidden_size,
        quantization_profile_for_module(qwen3_5_moe_config, "language_model.lm_head"),
    );
    tensor_profiles
}

/// Generates only the language tensors that should be resident in MLX memory.
///
/// Sparse selected-expert switch-MLP tensors are excluded so startup
/// materialization keeps router and shared-expert weights resident while the
/// pager owns selected experts.
#[must_use]
pub fn qwen3_5_moe_resident_language_tensor_profiles(
    qwen3_5_moe_config: &Qwen3_5MoEConfig,
) -> Vec<TensorProfile> {
    let complete_language_tensor_profiles =
        qwen3_5_moe_language_tensor_profiles(qwen3_5_moe_config);
    if qwen3_5_moe_config.is_dense_model() {
        return complete_language_tensor_profiles;
    }

    complete_language_tensor_profiles
        .into_iter()
        .filter(|tensor_profile| !is_sparse_selected_expert_tensor_name(&tensor_profile.name))
        .collect()
}

pub(in crate::qwen3_5_moe) fn is_sparse_selected_expert_tensor_name(tensor_name: &str) -> bool {
    tensor_name.contains(".mlp.switch_mlp.")
}

pub(super) fn validate_language_tensor_names(
    actual_language_tensor_names: &BTreeSet<&str>,
    language_tensor_profiles: &[TensorProfile],
) -> Result<(), Qwen3_5MoEArtifactError> {
    let expected_language_tensor_names = language_tensor_profiles
        .iter()
        .map(|tensor_profile| tensor_profile.name.as_str())
        .collect::<BTreeSet<_>>();

    if let Some(unexpected_tensor_name) = actual_language_tensor_names
        .difference(&expected_language_tensor_names)
        .next()
    {
        return Err(Qwen3_5MoEArtifactError::UnexpectedLanguageTensor {
            tensor_name: (*unexpected_tensor_name).to_owned(),
        });
    }
    if let Some(missing_tensor_name) = expected_language_tensor_names
        .difference(actual_language_tensor_names)
        .next()
    {
        return Err(Qwen3_5MoEArtifactError::MissingLanguageTensor {
            tensor_name: (*missing_tensor_name).to_owned(),
        });
    }
    Ok(())
}

fn push_decoder_layer_tensor_profiles(
    tensor_profiles: &mut Vec<TensorProfile>,
    decoder_layer_index: usize,
    hidden_size: usize,
    qwen3_5_moe_config: &Qwen3_5MoEConfig,
) {
    let query_head_count = qwen3_5_moe_config.query_head_count() as usize;
    let key_value_head_count = qwen3_5_moe_config.key_value_head_count() as usize;
    let head_dimension = qwen3_5_moe_config.head_dimension() as usize;
    let linear_key_head_count = qwen3_5_moe_config.linear_key_head_count() as usize;
    let linear_value_head_count = qwen3_5_moe_config.linear_value_head_count() as usize;
    let linear_key_head_dimension = qwen3_5_moe_config.linear_key_head_dimension() as usize;
    let linear_value_head_dimension = qwen3_5_moe_config.linear_value_head_dimension() as usize;
    let linear_convolution_kernel_dimension =
        qwen3_5_moe_config.linear_convolution_kernel_dimension() as usize;
    let layer_prefix = format!("language_model.model.layers.{decoder_layer_index}");
    tensor_profiles.push(tensor_profile(
        format!("{layer_prefix}.input_layernorm.weight"),
        TensorDtype::BFloat16,
        vec![hidden_size],
    ));
    if qwen3_5_moe_config.decoder_layer_is_full_attention(decoder_layer_index) {
        push_full_attention_tensor_profiles(
            tensor_profiles,
            &layer_prefix,
            hidden_size,
            query_head_count,
            key_value_head_count,
            head_dimension,
            qwen3_5_moe_config,
        );
    } else {
        push_linear_attention_tensor_profiles(
            tensor_profiles,
            &layer_prefix,
            hidden_size,
            linear_key_head_count,
            linear_value_head_count,
            linear_key_head_dimension,
            linear_value_head_dimension,
            linear_convolution_kernel_dimension,
            qwen3_5_moe_config,
        );
    }
    tensor_profiles.push(tensor_profile(
        format!("{layer_prefix}.post_attention_layernorm.weight"),
        TensorDtype::BFloat16,
        vec![hidden_size],
    ));
    if qwen3_5_moe_config.is_dense_model() {
        push_dense_mlp_tensor_profiles(
            tensor_profiles,
            &layer_prefix,
            hidden_size,
            qwen3_5_moe_config.dense_intermediate_size() as usize,
            qwen3_5_moe_config,
        );
    } else {
        push_sparse_moe_tensor_profiles(
            tensor_profiles,
            &layer_prefix,
            hidden_size,
            qwen3_5_moe_config.expert_count() as usize,
            qwen3_5_moe_config.expert_intermediate_size() as usize,
            qwen3_5_moe_config.shared_expert_intermediate_size() as usize,
            qwen3_5_moe_config,
        );
    }
}

fn push_dense_mlp_tensor_profiles(
    tensor_profiles: &mut Vec<TensorProfile>,
    layer_prefix: &str,
    hidden_size: usize,
    dense_intermediate_size: usize,
    qwen3_5_moe_config: &Qwen3_5MoEConfig,
) {
    let dense_mlp_prefix = format!("{layer_prefix}.mlp");
    for (projection_name, output_dimension, input_dimension) in [
        ("gate_proj", dense_intermediate_size, hidden_size),
        ("up_proj", dense_intermediate_size, hidden_size),
        ("down_proj", hidden_size, dense_intermediate_size),
    ] {
        push_quantized_affine(
            tensor_profiles,
            &format!("{dense_mlp_prefix}.{projection_name}"),
            &[output_dimension],
            input_dimension,
            quantization_profile_for_module(
                qwen3_5_moe_config,
                &format!("{dense_mlp_prefix}.{projection_name}"),
            ),
        );
    }
}

fn push_full_attention_tensor_profiles(
    tensor_profiles: &mut Vec<TensorProfile>,
    layer_prefix: &str,
    hidden_size: usize,
    query_head_count: usize,
    key_value_head_count: usize,
    head_dimension: usize,
    qwen3_5_moe_config: &Qwen3_5MoEConfig,
) {
    let query_projection_output_dimension = query_head_count * head_dimension * 2;
    let key_value_projection_output_dimension = key_value_head_count * head_dimension;
    let attention_output_input_dimension = query_head_count * head_dimension;
    for (projection_name, output_dimension, input_dimension) in [
        ("q_proj", query_projection_output_dimension, hidden_size),
        ("k_proj", key_value_projection_output_dimension, hidden_size),
        ("v_proj", key_value_projection_output_dimension, hidden_size),
        ("o_proj", hidden_size, attention_output_input_dimension),
    ] {
        push_quantized_affine(
            tensor_profiles,
            &format!("{layer_prefix}.self_attn.{projection_name}"),
            &[output_dimension],
            input_dimension,
            quantization_profile_for_module(
                qwen3_5_moe_config,
                &format!("{layer_prefix}.self_attn.{projection_name}"),
            ),
        );
    }
    for norm_name in ["q_norm", "k_norm"] {
        tensor_profiles.push(tensor_profile(
            format!("{layer_prefix}.self_attn.{norm_name}.weight"),
            TensorDtype::BFloat16,
            vec![head_dimension],
        ));
    }
}

#[allow(clippy::too_many_arguments)]
fn push_linear_attention_tensor_profiles(
    tensor_profiles: &mut Vec<TensorProfile>,
    layer_prefix: &str,
    hidden_size: usize,
    linear_key_head_count: usize,
    linear_value_head_count: usize,
    linear_key_head_dimension: usize,
    linear_value_head_dimension: usize,
    linear_convolution_kernel_dimension: usize,
    qwen3_5_moe_config: &Qwen3_5MoEConfig,
) {
    let linear_key_dimension = linear_key_head_count * linear_key_head_dimension;
    let linear_value_dimension = linear_value_head_count * linear_value_head_dimension;
    let convolution_dimension = linear_key_dimension * 2 + linear_value_dimension;
    let linear_prefix = format!("{layer_prefix}.linear_attn");

    tensor_profiles.push(tensor_profile(
        format!("{linear_prefix}.conv1d.weight"),
        TensorDtype::BFloat16,
        vec![
            convolution_dimension,
            linear_convolution_kernel_dimension,
            1,
        ],
    ));
    for (projection_name, output_dimension, input_dimension) in [
        (
            "in_proj_qkv",
            linear_key_dimension * 2 + linear_value_dimension,
            hidden_size,
        ),
        ("in_proj_z", linear_value_dimension, hidden_size),
        ("in_proj_b", linear_value_head_count, hidden_size),
        ("in_proj_a", linear_value_head_count, hidden_size),
        ("out_proj", hidden_size, linear_value_dimension),
    ] {
        push_quantized_affine(
            tensor_profiles,
            &format!("{linear_prefix}.{projection_name}"),
            &[output_dimension],
            input_dimension,
            quantization_profile_for_module(
                qwen3_5_moe_config,
                &format!("{linear_prefix}.{projection_name}"),
            ),
        );
    }
    tensor_profiles.push(tensor_profile(
        format!("{linear_prefix}.dt_bias"),
        TensorDtype::BFloat16,
        vec![linear_value_head_count],
    ));
    tensor_profiles.push(tensor_profile(
        format!("{linear_prefix}.A_log"),
        TensorDtype::BFloat16OrFloat32,
        vec![linear_value_head_count],
    ));
    tensor_profiles.push(tensor_profile(
        format!("{linear_prefix}.norm.weight"),
        TensorDtype::BFloat16,
        vec![linear_value_head_dimension],
    ));
}

fn push_sparse_moe_tensor_profiles(
    tensor_profiles: &mut Vec<TensorProfile>,
    layer_prefix: &str,
    hidden_size: usize,
    expert_count: usize,
    expert_intermediate_size: usize,
    shared_expert_intermediate_size: usize,
    qwen3_5_moe_config: &Qwen3_5MoEConfig,
) {
    let mlp_prefix = format!("{layer_prefix}.mlp");
    let gate_profile =
        quantization_profile_for_module(qwen3_5_moe_config, &format!("{mlp_prefix}.gate"));
    if gate_profile.is_unquantized() {
        // The MoE router gate is stored as a plain bfloat16 weight (no scales/biases).
        tensor_profiles.push(tensor_profile(
            format!("{mlp_prefix}.gate.weight"),
            TensorDtype::BFloat16,
            vec![expert_count, hidden_size],
        ));
    } else {
        push_quantized_affine(
            tensor_profiles,
            &format!("{mlp_prefix}.gate"),
            &[expert_count],
            hidden_size,
            gate_profile,
        );
    }
    for (projection_name, output_dimension, input_dimension) in [
        ("gate_proj", expert_intermediate_size, hidden_size),
        ("up_proj", expert_intermediate_size, hidden_size),
        ("down_proj", hidden_size, expert_intermediate_size),
    ] {
        push_quantized_affine(
            tensor_profiles,
            &format!("{mlp_prefix}.switch_mlp.{projection_name}"),
            &[expert_count, output_dimension],
            input_dimension,
            quantization_profile_for_module(
                qwen3_5_moe_config,
                &format!("{mlp_prefix}.switch_mlp.{projection_name}"),
            ),
        );
    }
    for (projection_name, output_dimension, input_dimension) in [
        ("gate_proj", shared_expert_intermediate_size, hidden_size),
        ("up_proj", shared_expert_intermediate_size, hidden_size),
        ("down_proj", hidden_size, shared_expert_intermediate_size),
    ] {
        push_quantized_affine(
            tensor_profiles,
            &format!("{mlp_prefix}.shared_expert.{projection_name}"),
            &[output_dimension],
            input_dimension,
            quantization_profile_for_module(
                qwen3_5_moe_config,
                &format!("{mlp_prefix}.shared_expert.{projection_name}"),
            ),
        );
    }
    push_quantized_affine(
        tensor_profiles,
        &format!("{mlp_prefix}.shared_expert_gate"),
        &[1],
        hidden_size,
        quantization_profile_for_module(
            qwen3_5_moe_config,
            &format!("{mlp_prefix}.shared_expert_gate"),
        ),
    );
}

fn quantization_profile_for_module(
    qwen3_5_moe_config: &Qwen3_5MoEConfig,
    module_name: &str,
) -> OptiQQuantizationProfile {
    qwen3_5_moe_config.quantization_profile_for_module(module_name)
}

fn push_quantized_affine(
    tensor_profiles: &mut Vec<TensorProfile>,
    tensor_prefix: &str,
    leading_dimensions: &[usize],
    input_dimension: usize,
    quantization_profile: OptiQQuantizationProfile,
) {
    if quantization_profile.is_unquantized() {
        let mut native_bfloat16_weight_shape = leading_dimensions.to_vec();
        native_bfloat16_weight_shape.push(input_dimension);
        tensor_profiles.push(tensor_profile(
            format!("{tensor_prefix}.weight"),
            TensorDtype::BFloat16,
            native_bfloat16_weight_shape,
        ));
        return;
    }
    let quantization_bits = quantization_profile.bits as usize;
    let quantization_group_size = quantization_profile.group_size as usize;
    let mut packed_weight_shape = leading_dimensions.to_vec();
    packed_weight_shape.push(input_dimension * quantization_bits / PACKING_WORD_BITS);
    let mut scale_shape = leading_dimensions.to_vec();
    scale_shape.push(input_dimension / quantization_group_size);
    tensor_profiles.push(tensor_profile(
        format!("{tensor_prefix}.weight"),
        TensorDtype::UInt32,
        packed_weight_shape,
    ));
    tensor_profiles.push(tensor_profile(
        format!("{tensor_prefix}.scales"),
        TensorDtype::BFloat16,
        scale_shape.clone(),
    ));
    tensor_profiles.push(tensor_profile(
        format!("{tensor_prefix}.biases"),
        TensorDtype::BFloat16,
        scale_shape,
    ));
}

fn tensor_profile(
    tensor_name: String,
    tensor_dtype: TensorDtype,
    tensor_shape: Vec<usize>,
) -> TensorProfile {
    TensorProfile {
        name: tensor_name,
        dtype: tensor_dtype,
        shape: tensor_shape,
    }
}
