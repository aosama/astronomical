use std::collections::BTreeSet;

use crate::qwen3_5_core::dense::tensor_spec;
use crate::{TensorDtype, TensorProfile};

use super::quantizations::optiq::OptiQQuantizationProfile;
use super::{Qwen3_5ArtifactError, Qwen3_5Config, Qwen3_5FeedForwardArchitecture};

#[cfg(test)]
mod tests;

const PACKING_WORD_BITS: usize = 32;

/// Generates the executable language-tensor metadata for a Qwen3.5 artifact.
///
/// Returns an error when a module's storage codec cannot be turned into an
/// honest tensor contract — notably a jangtq2 codebook module whose shapes
/// come from the family's contract rather than the affine grid.
pub fn qwen3_5_language_tensor_profiles(
    qwen3_5_config: &Qwen3_5Config,
) -> Result<Vec<TensorProfile>, Qwen3_5ArtifactError> {
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
    )?;
    for decoder_layer_index in 0..layer_count {
        append_qwen3_5_decoder_layer_tensor_profiles(
            &mut tensor_profiles,
            decoder_layer_index,
            hidden_size,
            qwen3_5_config,
        )?;
    }
    tensor_profiles.push(qwen3_5_tensor_profile(
        "language_model.model.norm.weight".to_owned(),
        TensorDtype::ModelFloat,
        vec![hidden_size],
    ));
    if !qwen3_5_config.has_tied_embeddings() {
        append_qwen3_5_quantized_affine_tensor_profiles(
            &mut tensor_profiles,
            "language_model.lm_head",
            &[vocabulary_size],
            hidden_size,
            qwen3_5_config.quantization_profile_for_module("language_model.lm_head"),
        )?;
    }
    Ok(tensor_profiles)
}

/// Returns the language tensors that remain resident after sparse expert paging.
/// Resident paging executes only in the direct-MLX engine world, so the fn and
/// its helper exist only there.
#[cfg(feature = "direct-mlx")]
pub(crate) fn qwen3_5_resident_language_tensor_profiles(
    qwen3_5_config: &Qwen3_5Config,
) -> Result<Vec<TensorProfile>, Qwen3_5ArtifactError> {
    let complete_language_tensor_profiles = qwen3_5_language_tensor_profiles(qwen3_5_config)?;
    match qwen3_5_config.feed_forward_architecture() {
        Qwen3_5FeedForwardArchitecture::Dense => Ok(complete_language_tensor_profiles),
        Qwen3_5FeedForwardArchitecture::MixtureOfExperts => Ok(complete_language_tensor_profiles
            .into_iter()
            .filter(|tensor_profile| !is_sparse_selected_expert_tensor_name(&tensor_profile.name))
            .collect()),
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
) -> Result<(), Qwen3_5ArtifactError> {
    let query_head_count = qwen3_5_config.query_head_count() as usize;
    let key_value_head_count = qwen3_5_config.key_value_head_count() as usize;
    let head_dimension = qwen3_5_config.head_dimension() as usize;
    let layer_prefix = format!("language_model.model.layers.{decoder_layer_index}");
    tensor_profiles.push(qwen3_5_tensor_profile(
        format!("{layer_prefix}.input_layernorm.weight"),
        TensorDtype::ModelFloat,
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
        )?;
    } else {
        append_qwen3_5_linear_attention_tensor_profiles(
            tensor_profiles,
            &layer_prefix,
            hidden_size,
            qwen3_5_config,
        )?;
    }
    tensor_profiles.push(qwen3_5_tensor_profile(
        format!("{layer_prefix}.post_attention_layernorm.weight"),
        TensorDtype::ModelFloat,
        vec![hidden_size],
    ));
    match qwen3_5_config.feed_forward_architecture() {
        Qwen3_5FeedForwardArchitecture::Dense => {
            tensor_spec::append_qwen3_5_dense_mlp_tensor_profiles(
                tensor_profiles,
                &layer_prefix,
                hidden_size,
                qwen3_5_config,
            )?
        }
        Qwen3_5FeedForwardArchitecture::MixtureOfExperts => {
            append_qwen3_5_moe_feed_forward_tensor_profiles(
                tensor_profiles,
                &layer_prefix,
                hidden_size,
                qwen3_5_config,
            )?;
        }
    }
    Ok(())
}

fn append_qwen3_5_moe_feed_forward_tensor_profiles(
    tensor_profiles: &mut Vec<TensorProfile>,
    decoder_layer_prefix: &str,
    hidden_size: usize,
    qwen3_5_config: &Qwen3_5Config,
) -> Result<(), Qwen3_5ArtifactError> {
    let mixture_of_experts_prefix = format!("{decoder_layer_prefix}.mlp");
    let router_module_name = format!("{mixture_of_experts_prefix}.gate");
    let router_quantization_profile =
        qwen3_5_config.quantization_profile_for_module(&router_module_name);
    if router_quantization_profile.is_unquantized() {
        tensor_profiles.push(qwen3_5_tensor_profile(
            format!("{router_module_name}.weight"),
            TensorDtype::ModelFloat,
            vec![qwen3_5_config.expert_count() as usize, hidden_size],
        ));
    } else {
        append_qwen3_5_quantized_affine_tensor_profiles(
            tensor_profiles,
            &router_module_name,
            &[qwen3_5_config.expert_count() as usize],
            hidden_size,
            router_quantization_profile,
        )?;
    }
    for (projection_name, output_dimension, input_dimension) in [
        (
            "gate_proj",
            qwen3_5_config.expert_intermediate_size() as usize,
            hidden_size,
        ),
        (
            "up_proj",
            qwen3_5_config.expert_intermediate_size() as usize,
            hidden_size,
        ),
        (
            "down_proj",
            hidden_size,
            qwen3_5_config.expert_intermediate_size() as usize,
        ),
    ] {
        let projection_module_name =
            format!("{mixture_of_experts_prefix}.switch_mlp.{projection_name}");
        append_qwen3_5_quantized_affine_tensor_profiles(
            tensor_profiles,
            &projection_module_name,
            &[qwen3_5_config.expert_count() as usize, output_dimension],
            input_dimension,
            qwen3_5_config.quantization_profile_for_module(&projection_module_name),
        )?;
    }
    for (projection_name, output_dimension, input_dimension) in [
        (
            "gate_proj",
            qwen3_5_config.shared_expert_intermediate_size() as usize,
            hidden_size,
        ),
        (
            "up_proj",
            qwen3_5_config.shared_expert_intermediate_size() as usize,
            hidden_size,
        ),
        (
            "down_proj",
            hidden_size,
            qwen3_5_config.shared_expert_intermediate_size() as usize,
        ),
    ] {
        let projection_module_name =
            format!("{mixture_of_experts_prefix}.shared_expert.{projection_name}");
        append_qwen3_5_quantized_affine_tensor_profiles(
            tensor_profiles,
            &projection_module_name,
            &[output_dimension],
            input_dimension,
            qwen3_5_config.quantization_profile_for_module(&projection_module_name),
        )?;
    }
    let shared_expert_gate_module_name = format!("{mixture_of_experts_prefix}.shared_expert_gate");
    append_qwen3_5_quantized_affine_tensor_profiles(
        tensor_profiles,
        &shared_expert_gate_module_name,
        &[1],
        hidden_size,
        qwen3_5_config.quantization_profile_for_module(&shared_expert_gate_module_name),
    )?;
    Ok(())
}

#[cfg(feature = "direct-mlx")]
fn is_sparse_selected_expert_tensor_name(tensor_name: &str) -> bool {
    tensor_name.contains(".mlp.switch_mlp.")
}

fn append_qwen3_5_full_attention_tensor_profiles(
    tensor_profiles: &mut Vec<TensorProfile>,
    layer_prefix: &str,
    hidden_size: usize,
    query_head_count: usize,
    key_value_head_count: usize,
    head_dimension: usize,
    qwen3_5_config: &Qwen3_5Config,
) -> Result<(), Qwen3_5ArtifactError> {
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
        )?;
    }
    for normalization_name in ["q_norm", "k_norm"] {
        tensor_profiles.push(qwen3_5_tensor_profile(
            format!("{layer_prefix}.self_attn.{normalization_name}.weight"),
            TensorDtype::ModelFloat,
            vec![head_dimension],
        ));
    }
    Ok(())
}

fn append_qwen3_5_linear_attention_tensor_profiles(
    tensor_profiles: &mut Vec<TensorProfile>,
    layer_prefix: &str,
    hidden_size: usize,
    qwen3_5_config: &Qwen3_5Config,
) -> Result<(), Qwen3_5ArtifactError> {
    let linear_key_head_count = qwen3_5_config.linear_key_head_count() as usize;
    let linear_value_head_count = qwen3_5_config.linear_value_head_count() as usize;
    let linear_key_head_dimension = qwen3_5_config.linear_key_head_dimension() as usize;
    let linear_value_head_dimension = qwen3_5_config.linear_value_head_dimension() as usize;
    let linear_key_dimension = linear_key_head_count * linear_key_head_dimension;
    let linear_value_dimension = linear_value_head_count * linear_value_head_dimension;
    let convolution_dimension = linear_key_dimension * 2 + linear_value_dimension;
    let convolution_kernel_dimension =
        qwen3_5_config.linear_convolution_kernel_dimension() as usize;
    let linear_attention_prefix = format!("{layer_prefix}.linear_attn");
    // MLX Conv1d consumes `[output, kernel, input]`. Published Qwen3.5
    // SafeTensors may keep PyTorch's `[output, input, kernel]` order; the
    // loader transposes that equivalent layout before either convolution path.
    tensor_profiles.push(TensorProfile {
        name: format!("{linear_attention_prefix}.conv1d.weight"),
        dtype: TensorDtype::ModelFloat,
        shape: vec![convolution_dimension, convolution_kernel_dimension, 1],
        equivalent_published_shapes: vec![vec![
            convolution_dimension,
            1,
            convolution_kernel_dimension,
        ]],
    });
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
        )?;
    }
    tensor_profiles.push(qwen3_5_tensor_profile(
        format!("{linear_attention_prefix}.dt_bias"),
        TensorDtype::ModelFloat,
        vec![linear_value_head_count],
    ));
    tensor_profiles.push(qwen3_5_tensor_profile(
        format!("{linear_attention_prefix}.A_log"),
        TensorDtype::ModelFloat,
        vec![linear_value_head_count],
    ));
    tensor_profiles.push(qwen3_5_tensor_profile(
        format!("{linear_attention_prefix}.norm.weight"),
        TensorDtype::ModelFloat,
        vec![linear_value_head_dimension],
    ));
    Ok(())
}

/// Appends the tensor profiles for one quantized module.
///
/// The codec decides the shape source: affine and native-bfloat16 profiles
/// derive from the MLX affine grid exactly as before, while a jangtq2
/// codebook profile delegates wholly to the jangtq2 family's tensor
/// contract — this function never derives codebook shapes itself.
pub(crate) fn append_qwen3_5_quantized_affine_tensor_profiles(
    tensor_profiles: &mut Vec<TensorProfile>,
    tensor_prefix: &str,
    leading_dimensions: &[usize],
    input_dimension: usize,
    quantization_profile: OptiQQuantizationProfile,
) -> Result<(), Qwen3_5ArtifactError> {
    if quantization_profile.is_jangtq2_codebook() {
        return append_jangtq2_codebook_tensor_profiles(
            tensor_profiles,
            tensor_prefix,
            leading_dimensions,
            input_dimension,
            quantization_profile,
        );
    }
    if quantization_profile.is_unquantized() {
        let mut native_bfloat16_weight_shape = leading_dimensions.to_vec();
        native_bfloat16_weight_shape.push(input_dimension);
        tensor_profiles.push(qwen3_5_tensor_profile(
            format!("{tensor_prefix}.weight"),
            TensorDtype::ModelFloat,
            native_bfloat16_weight_shape,
        ));
        return Ok(());
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
        TensorDtype::AffineQuantizationFloat,
        scale_shape.clone(),
    ));
    tensor_profiles.push(qwen3_5_tensor_profile(
        format!("{tensor_prefix}.biases"),
        TensorDtype::AffineQuantizationFloat,
        scale_shape,
    ));
    Ok(())
}

/// Appends the expert-stacked codebook tensors the jangtq2 family's contract
/// prescribes: `tq2_packed` `uint32` codes and `tq2_scales` `float16` row
/// scales, both stacked over every expert of the layer.
///
/// Leading dimensions distinguish the routed-expert projections (expert
/// count, output width) from any hypothetical non-expert codebook module
/// (output width alone, a single unstacked expert).
fn append_jangtq2_codebook_tensor_profiles(
    tensor_profiles: &mut Vec<TensorProfile>,
    tensor_prefix: &str,
    leading_dimensions: &[usize],
    input_dimension: usize,
    quantization_profile: OptiQQuantizationProfile,
) -> Result<(), Qwen3_5ArtifactError> {
    let (codebook_expert_count, codebook_output_dimension) = match leading_dimensions {
        [expert_count, output_dimension] => (*expert_count as u32, *output_dimension as u32),
        [output_dimension] => (1, *output_dimension as u32),
        _ => {
            return Err(Qwen3_5ArtifactError::Jangtq2TensorContract {
                module_name: tensor_prefix.to_owned(),
                source:
                    astronomical_jangtq2_qwen35::configuration::Jangtq2ConfigurationError::InputWidthNotPackingAligned {
                        actual_input_width: input_dimension as u32,
                        required_alignment_multiple: 1,
                    },
            });
        }
    };
    let crate::qwen3_5_core::quantizations::optiq::ModuleWeightCodec::Jangtq2CodebookQuantized {
        rotation,
    } = quantization_profile.weight_codec
    else {
        // The caller dispatches on the codec before delegating; this arm is
        // unreachable by construction.
        return Ok(());
    };
    let codebook_override_entry =
        astronomical_jangtq2_qwen35::configuration::Jangtq2WeightOverrideEntry::Codebook {
            mode: astronomical_jangtq2_qwen35::configuration::Jangtq2CodebookMode,
            bits: quantization_profile.bits,
            rotation,
        };
    // The affine group size is an affine-arm concern only; the codebook
    // contract never consults it.
    let codebook_module_geometry = astronomical_jangtq2_qwen35::artifacts::Jangtq2ModuleGeometry {
        input_width: input_dimension as u32,
        output_width: codebook_output_dimension,
        expert_count: codebook_expert_count,
        affine_group_size: 64,
    };
    let codebook_tensor_inventory =
        astronomical_jangtq2_qwen35::artifacts::tensor_inventory_for_module(
            tensor_prefix,
            &codebook_override_entry,
            &codebook_module_geometry,
        )
        .map_err(|source| Qwen3_5ArtifactError::Jangtq2TensorContract {
            module_name: tensor_prefix.to_owned(),
            source,
        })?;
    for codebook_tensor_expectation in codebook_tensor_inventory.expected_tensors {
        let contract_dtype = match codebook_tensor_expectation.element_kind {
            astronomical_jangtq2_qwen35::artifacts::ContractTensorElementKind::PackedUnsignedInteger32 => {
                TensorDtype::UInt32
            }
            astronomical_jangtq2_qwen35::artifacts::ContractTensorElementKind::CodebookScaleFloat16 => {
                TensorDtype::Float16
            }
            astronomical_jangtq2_qwen35::artifacts::ContractTensorElementKind::AffineCompanionBfloat16 => {
                TensorDtype::AffineQuantizationFloat
            }
            astronomical_jangtq2_qwen35::artifacts::ContractTensorElementKind::NativeWeightBfloat16 => {
                TensorDtype::ModelFloat
            }
        };
        tensor_profiles.push(qwen3_5_tensor_profile(
            codebook_tensor_expectation.tensor_name,
            contract_dtype,
            codebook_tensor_expectation.shape,
        ));
    }
    Ok(())
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
        equivalent_published_shapes: Vec::new(),
    }
}
