use std::collections::BTreeSet;

use crate::qwen3_5::dense::tensor_spec::append_qwen3_5_dense_mlp_tensor_profiles;
use crate::qwen3_5_moe::artifacts::tensor_spec::{
    append_qwen3_5_moe_feed_forward_tensor_profiles, is_sparse_selected_expert_tensor_name,
};
use crate::{TensorDtype, TensorProfile};

use super::quantizations::optiq::OptiQQuantizationProfile;
use super::{Qwen3_5ArtifactError, Qwen3_5Config, Qwen3_5FeedForwardArchitecture};

const PACKING_WORD_BITS: usize = 32;

/// Generates the executable language-tensor metadata for a Qwen3.5 artifact.
#[must_use]
pub fn qwen3_5_language_tensor_profiles(qwen3_5_config: &Qwen3_5Config) -> Vec<TensorProfile> {
    let hidden_size = qwen3_5_config.hidden_size() as usize;
    let layer_count = qwen3_5_config.layer_count() as usize;
    let vocabulary_size = qwen3_5_config.vocabulary_size() as usize;
    let mut tensor_profiles = Vec::new();
    append_qwen3_5_quantized_affine_tensor_profiles(
        &mut tensor_profiles,
        "language_model.model.embed_tokens",
        &[vocabulary_size],
        hidden_size,
        qwen3_5_config.quantization_profile_for_module("language_model.model.embed_tokens"),
    );
    for decoder_layer_index in 0..layer_count {
        append_qwen3_5_decoder_layer_tensor_profiles(
            &mut tensor_profiles,
            decoder_layer_index,
            hidden_size,
            qwen3_5_config,
        );
    }
    tensor_profiles.push(qwen3_5_tensor_profile(
        "language_model.model.norm.weight".to_owned(),
        TensorDtype::BFloat16,
        vec![hidden_size],
    ));
    append_qwen3_5_quantized_affine_tensor_profiles(
        &mut tensor_profiles,
        "language_model.lm_head",
        &[vocabulary_size],
        hidden_size,
        qwen3_5_config.quantization_profile_for_module("language_model.lm_head"),
    );
    tensor_profiles
}

/// Returns the language tensors that remain resident after sparse expert paging.
#[must_use]
pub fn qwen3_5_resident_language_tensor_profiles(
    qwen3_5_config: &Qwen3_5Config,
) -> Vec<TensorProfile> {
    let complete_language_tensor_profiles = qwen3_5_language_tensor_profiles(qwen3_5_config);
    match qwen3_5_config.feed_forward_architecture() {
        Qwen3_5FeedForwardArchitecture::Dense => complete_language_tensor_profiles,
        Qwen3_5FeedForwardArchitecture::MixtureOfExperts => complete_language_tensor_profiles
            .into_iter()
            .filter(|tensor_profile| !is_sparse_selected_expert_tensor_name(&tensor_profile.name))
            .collect(),
    }
}

pub(super) fn validate_language_tensor_names(
    actual_language_tensor_names: &BTreeSet<&str>,
    language_tensor_profiles: &[TensorProfile],
) -> Result<(), Qwen3_5ArtifactError> {
    let expected_language_tensor_names = language_tensor_profiles
        .iter()
        .map(|tensor_profile| tensor_profile.name.as_str())
        .collect::<BTreeSet<_>>();
    if let Some(unexpected_tensor_name) = actual_language_tensor_names
        .difference(&expected_language_tensor_names)
        .next()
    {
        return Err(Qwen3_5ArtifactError::UnexpectedLanguageTensor {
            tensor_name: (*unexpected_tensor_name).to_owned(),
        });
    }
    if let Some(missing_tensor_name) = expected_language_tensor_names
        .difference(actual_language_tensor_names)
        .next()
    {
        return Err(Qwen3_5ArtifactError::MissingLanguageTensor {
            tensor_name: (*missing_tensor_name).to_owned(),
        });
    }
    Ok(())
}

fn append_qwen3_5_decoder_layer_tensor_profiles(
    tensor_profiles: &mut Vec<TensorProfile>,
    decoder_layer_index: usize,
    hidden_size: usize,
    qwen3_5_config: &Qwen3_5Config,
) {
    let query_head_count = qwen3_5_config.query_head_count() as usize;
    let key_value_head_count = qwen3_5_config.key_value_head_count() as usize;
    let head_dimension = qwen3_5_config.head_dimension() as usize;
    let layer_prefix = format!("language_model.model.layers.{decoder_layer_index}");
    tensor_profiles.push(qwen3_5_tensor_profile(
        format!("{layer_prefix}.input_layernorm.weight"),
        TensorDtype::BFloat16,
        vec![hidden_size],
    ));
    if qwen3_5_config.decoder_layer_is_full_attention(decoder_layer_index) {
        append_qwen3_5_full_attention_tensor_profiles(
            tensor_profiles,
            &layer_prefix,
            hidden_size,
            query_head_count,
            key_value_head_count,
            head_dimension,
            qwen3_5_config,
        );
    } else {
        append_qwen3_5_linear_attention_tensor_profiles(
            tensor_profiles,
            &layer_prefix,
            hidden_size,
            qwen3_5_config,
        );
    }
    tensor_profiles.push(qwen3_5_tensor_profile(
        format!("{layer_prefix}.post_attention_layernorm.weight"),
        TensorDtype::BFloat16,
        vec![hidden_size],
    ));
    match qwen3_5_config.feed_forward_architecture() {
        Qwen3_5FeedForwardArchitecture::Dense => append_qwen3_5_dense_mlp_tensor_profiles(
            tensor_profiles,
            &layer_prefix,
            hidden_size,
            qwen3_5_config,
        ),
        Qwen3_5FeedForwardArchitecture::MixtureOfExperts => {
            append_qwen3_5_moe_feed_forward_tensor_profiles(
                tensor_profiles,
                &layer_prefix,
                hidden_size,
                qwen3_5_config,
            );
        }
    }
}

fn append_qwen3_5_full_attention_tensor_profiles(
    tensor_profiles: &mut Vec<TensorProfile>,
    layer_prefix: &str,
    hidden_size: usize,
    query_head_count: usize,
    key_value_head_count: usize,
    head_dimension: usize,
    qwen3_5_config: &Qwen3_5Config,
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
        let projection_module_name = format!("{layer_prefix}.self_attn.{projection_name}");
        append_qwen3_5_quantized_affine_tensor_profiles(
            tensor_profiles,
            &projection_module_name,
            &[output_dimension],
            input_dimension,
            qwen3_5_config.quantization_profile_for_module(&projection_module_name),
        );
    }
    for normalization_name in ["q_norm", "k_norm"] {
        tensor_profiles.push(qwen3_5_tensor_profile(
            format!("{layer_prefix}.self_attn.{normalization_name}.weight"),
            TensorDtype::BFloat16,
            vec![head_dimension],
        ));
    }
}

fn append_qwen3_5_linear_attention_tensor_profiles(
    tensor_profiles: &mut Vec<TensorProfile>,
    layer_prefix: &str,
    hidden_size: usize,
    qwen3_5_config: &Qwen3_5Config,
) {
    let linear_key_head_count = qwen3_5_config.linear_key_head_count() as usize;
    let linear_value_head_count = qwen3_5_config.linear_value_head_count() as usize;
    let linear_key_head_dimension = qwen3_5_config.linear_key_head_dimension() as usize;
    let linear_value_head_dimension = qwen3_5_config.linear_value_head_dimension() as usize;
    let linear_key_dimension = linear_key_head_count * linear_key_head_dimension;
    let linear_value_dimension = linear_value_head_count * linear_value_head_dimension;
    let convolution_dimension = linear_key_dimension * 2 + linear_value_dimension;
    let linear_attention_prefix = format!("{layer_prefix}.linear_attn");
    tensor_profiles.push(qwen3_5_tensor_profile(
        format!("{linear_attention_prefix}.conv1d.weight"),
        TensorDtype::BFloat16,
        vec![
            convolution_dimension,
            qwen3_5_config.linear_convolution_kernel_dimension() as usize,
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
        let projection_module_name = format!("{linear_attention_prefix}.{projection_name}");
        append_qwen3_5_quantized_affine_tensor_profiles(
            tensor_profiles,
            &projection_module_name,
            &[output_dimension],
            input_dimension,
            qwen3_5_config.quantization_profile_for_module(&projection_module_name),
        );
    }
    tensor_profiles.push(qwen3_5_tensor_profile(
        format!("{linear_attention_prefix}.dt_bias"),
        TensorDtype::BFloat16,
        vec![linear_value_head_count],
    ));
    tensor_profiles.push(qwen3_5_tensor_profile(
        format!("{linear_attention_prefix}.A_log"),
        TensorDtype::BFloat16OrFloat32,
        vec![linear_value_head_count],
    ));
    tensor_profiles.push(qwen3_5_tensor_profile(
        format!("{linear_attention_prefix}.norm.weight"),
        TensorDtype::BFloat16,
        vec![linear_value_head_dimension],
    ));
}

pub(crate) fn append_qwen3_5_quantized_affine_tensor_profiles(
    tensor_profiles: &mut Vec<TensorProfile>,
    tensor_prefix: &str,
    leading_dimensions: &[usize],
    input_dimension: usize,
    quantization_profile: OptiQQuantizationProfile,
) {
    if quantization_profile.is_unquantized() {
        let mut native_bfloat16_weight_shape = leading_dimensions.to_vec();
        native_bfloat16_weight_shape.push(input_dimension);
        tensor_profiles.push(qwen3_5_tensor_profile(
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
    tensor_profiles.push(qwen3_5_tensor_profile(
        format!("{tensor_prefix}.weight"),
        TensorDtype::UInt32,
        packed_weight_shape,
    ));
    tensor_profiles.push(qwen3_5_tensor_profile(
        format!("{tensor_prefix}.scales"),
        TensorDtype::BFloat16,
        scale_shape.clone(),
    ));
    tensor_profiles.push(qwen3_5_tensor_profile(
        format!("{tensor_prefix}.biases"),
        TensorDtype::BFloat16,
        scale_shape,
    ));
}

pub(crate) fn qwen3_5_tensor_profile(
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
