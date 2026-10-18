//! Packed-weight modules and family weight loading.
//!
//! Every rotated projection applies the Hadamard transform unconditionally:
//! a `PackedProjection` cannot be constructed without its sign vector and
//! block size, so there is no code path that could silently skip the
//! transform the pack's contract requires.

use astronomical_mlx_c_rust::{MlxArray, MlxCError};
use astronomical_runtime_integration::{MlxRuntime, MlxSafetensors};

use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_core::model_math::tensor_slicing;

use super::super::artifacts::ValidatedFoldedPack;
use super::PrismHadamardModelError;
use super::fwht::{apply_forward_transform, apply_inverse_transform};

/// The ternary quantization constants of the family's packs.
const QUANTIZATION_GROUP_SIZE: i32 = 128;
const QUANTIZATION_BITS: i32 = 2;

/// One rotated ternary projection: Hadamard forward transform, then grouped
/// affine quantized matmul against the transposed packed weights.
#[derive(Debug)]
pub(crate) struct PackedProjection {
    packed_weight: MlxArray,
    quantization_scales: MlxArray,
    quantization_biases: MlxArray,
    sign_vector: MlxArray,
    block_size: u32,
}

impl PackedProjection {
    pub(crate) fn forward(
        &self,
        runtime: &MlxRuntime,
        activations: &MlxArray,
    ) -> Result<MlxArray, MlxCError> {
        let transformed_activations =
            apply_forward_transform(runtime, activations, &self.sign_vector, self.block_size)?;
        runtime.quantized_matmul_affine(
            &transformed_activations,
            &self.packed_weight,
            &self.quantization_scales,
            &self.quantization_biases,
            true,
            QUANTIZATION_GROUP_SIZE,
            QUANTIZATION_BITS,
        )
    }
}

/// The packed token embedding: gather rows, dequantize, then the inverse
/// Hadamard transform restores the unrotated hidden basis.
#[derive(Debug)]
pub(crate) struct PackedEmbedding {
    packed_weight: MlxArray,
    quantization_scales: MlxArray,
    quantization_biases: MlxArray,
    sign_vector: MlxArray,
    block_size: u32,
}

impl PackedEmbedding {
    pub(crate) fn lookup(
        &self,
        runtime: &MlxRuntime,
        token_indices: &MlxArray,
    ) -> Result<MlxArray, MlxCError> {
        let selected_weights = runtime.take_axis(&self.packed_weight, token_indices, 0)?;
        let selected_scales = runtime.take_axis(&self.quantization_scales, token_indices, 0)?;
        let selected_biases = runtime.take_axis(&self.quantization_biases, token_indices, 0)?;
        let dequantized = runtime.dequantize_affine(
            &selected_weights,
            &selected_scales,
            &selected_biases,
            QUANTIZATION_GROUP_SIZE,
            QUANTIZATION_BITS,
        )?;
        apply_inverse_transform(runtime, &dequantized, &self.sign_vector, self.block_size)
    }
}

/// One unrotated float projection bound at the activation dtype.
#[derive(Debug)]
pub(crate) struct FloatProjection {
    weight: MlxArray,
}

impl FloatProjection {
    pub(crate) fn forward(
        &self,
        runtime: &MlxRuntime,
        activations: &MlxArray,
    ) -> Result<MlxArray, MlxCError> {
        let transposed_weight = runtime.transpose_axes(&self.weight, &[1, 0])?;
        runtime.matmul(activations, &transposed_weight)
    }
}

/// One decoder layer's weights: the attention half (either GDN
/// linear-attention or full attention) plus that layer's dense SwiGLU MLP.
#[derive(Debug)]
pub(crate) struct DecoderLayerWeights {
    pub(crate) attention: DecoderLayerAttentionWeights,
    pub(crate) mlp: DenseMlpWeights,
}

/// The attention half of one decoder layer.
#[derive(Debug)]
pub(crate) enum DecoderLayerAttentionWeights {
    Linear(LinearLayerWeights),
    Full(FullLayerWeights),
}

/// One dense SwiGLU MLP's three rotated projections.
#[derive(Debug)]
pub(crate) struct DenseMlpWeights {
    pub(crate) gate: PackedProjection,
    pub(crate) up: PackedProjection,
    pub(crate) down: PackedProjection,
}

#[derive(Debug)]
pub(crate) struct LinearLayerWeights {
    pub(crate) query_key_value: PackedProjection,
    pub(crate) output_gate: PackedProjection,
    pub(crate) output: PackedProjection,
    pub(crate) update_rate: FloatProjection,
    pub(crate) decay_interval: FloatProjection,
    pub(crate) convolution: MlxArray,
    pub(crate) decay_rate_logarithm: MlxArray,
    pub(crate) decay_interval_bias: MlxArray,
    pub(crate) normalization: MlxArray,
}

#[derive(Debug)]
pub(crate) struct FullLayerWeights {
    pub(crate) query: PackedProjection,
    pub(crate) key: PackedProjection,
    pub(crate) value: PackedProjection,
    pub(crate) output: PackedProjection,
    pub(crate) query_normalization: MlxArray,
    pub(crate) key_normalization: MlxArray,
}

/// The complete loaded family weight set.
#[derive(Debug)]
pub(crate) struct PrismHadamardWeights {
    pub(crate) embedding: PackedEmbedding,
    pub(crate) decoder_layers: Vec<DecoderLayerWeights>,
    pub(crate) input_normalization_weights: Vec<MlxArray>,
    pub(crate) post_attention_normalization_weights: Vec<MlxArray>,
    pub(crate) final_normalization: MlxArray,
    pub(crate) language_model_head: PackedProjection,
}

/// Loads every family tensor from the pack's retained weights file map.
pub(crate) fn load_family_weights(
    runtime: &MlxRuntime,
    pack: &ValidatedFoldedPack,
    loaded_tensors: &MlxSafetensors,
) -> Result<PrismHadamardWeights, PrismHadamardModelError> {
    let text = pack.config().text();
    let activation_dtype = super::fwht::activation_dtype();

    let take = |tensor_name: &str| -> Result<MlxArray, PrismHadamardModelError> {
        loaded_tensors
            .tensor(tensor_name)
            .map_err(|source| PrismHadamardModelError::TensorLookup {
                tensor_name: tensor_name.to_owned(),
                source,
            })
    };
    let take_cast = |tensor_name: &str| -> Result<MlxArray, PrismHadamardModelError> {
        let array = take(tensor_name)?;
        runtime
            .astype(&array, activation_dtype)
            .map_err(PrismHadamardModelError::WeightCasting)
    };

    let packed_projection =
        |module_path: &str| -> Result<PackedProjection, PrismHadamardModelError> {
            let stem = format!("language_model.{module_path}");
            let packed_weight = take(&format!("{stem}.weight"))?;
            let quantization_scales = take(&format!("{stem}.scales"))?;
            let quantization_biases = take(&format!("{stem}.biases"))?;
            let sign_vector = take(&format!("{stem}.signs"))?;
            Ok(PackedProjection {
                packed_weight,
                quantization_scales,
                quantization_biases,
                sign_vector,
                block_size: pack.hadamard_contract().block_size(),
            })
        };

    let embedding_stem = "language_model.model.embed_tokens";
    let embedding = PackedEmbedding {
        packed_weight: take(&format!("{embedding_stem}.weight"))?,
        quantization_scales: take(&format!("{embedding_stem}.scales"))?,
        quantization_biases: take(&format!("{embedding_stem}.biases"))?,
        sign_vector: take(&format!("{embedding_stem}.signs"))?,
        block_size: pack.hadamard_contract().block_size(),
    };

    let mut decoder_layers = Vec::with_capacity(text.num_hidden_layers as usize);
    let mut input_normalization_weights = Vec::with_capacity(text.num_hidden_layers as usize);
    let mut post_attention_normalization_weights =
        Vec::with_capacity(text.num_hidden_layers as usize);
    for (layer_index, layer_type) in text.layer_types.iter().enumerate() {
        let layer_prefix = format!("language_model.model.layers.{layer_index}");
        input_normalization_weights.push(take_cast(&format!(
            "{layer_prefix}.input_layernorm.weight"
        ))?);
        post_attention_normalization_weights.push(take_cast(&format!(
            "{layer_prefix}.post_attention_layernorm.weight"
        ))?);
        let decoder_layer = match layer_type.as_str() {
            "linear_attention" => DecoderLayerWeights {
                attention: DecoderLayerAttentionWeights::Linear(LinearLayerWeights {
                    query_key_value: packed_projection(&format!(
                        "model.layers.{layer_index}.linear_attn.in_proj_qkv"
                    ))?,
                    output_gate: packed_projection(&format!(
                        "model.layers.{layer_index}.linear_attn.in_proj_z"
                    ))?,
                    output: packed_projection(&format!(
                        "model.layers.{layer_index}.linear_attn.out_proj"
                    ))?,
                    update_rate: FloatProjection {
                        weight: take_cast(&format!("{layer_prefix}.linear_attn.in_proj_b.weight"))?,
                    },
                    decay_interval: FloatProjection {
                        weight: take_cast(&format!("{layer_prefix}.linear_attn.in_proj_a.weight"))?,
                    },
                    convolution: take_cast(&format!("{layer_prefix}.linear_attn.conv1d.weight"))?,
                    decay_rate_logarithm: take_cast(&format!("{layer_prefix}.linear_attn.A_log"))?,
                    decay_interval_bias: take_cast(&format!("{layer_prefix}.linear_attn.dt_bias"))?,
                    normalization: take_cast(&format!("{layer_prefix}.linear_attn.norm.weight"))?,
                }),
                mlp: DenseMlpWeights {
                    gate: packed_projection(&format!("model.layers.{layer_index}.mlp.gate_proj"))?,
                    up: packed_projection(&format!("model.layers.{layer_index}.mlp.up_proj"))?,
                    down: packed_projection(&format!("model.layers.{layer_index}.mlp.down_proj"))?,
                },
            },
            "full_attention" => DecoderLayerWeights {
                attention: DecoderLayerAttentionWeights::Full(FullLayerWeights {
                    query: packed_projection(&format!(
                        "model.layers.{layer_index}.self_attn.q_proj"
                    ))?,
                    key: packed_projection(&format!(
                        "model.layers.{layer_index}.self_attn.k_proj"
                    ))?,
                    value: packed_projection(&format!(
                        "model.layers.{layer_index}.self_attn.v_proj"
                    ))?,
                    output: packed_projection(&format!(
                        "model.layers.{layer_index}.self_attn.o_proj"
                    ))?,
                    query_normalization: take_cast(&format!(
                        "{layer_prefix}.self_attn.q_norm.weight"
                    ))?,
                    key_normalization: take_cast(&format!(
                        "{layer_prefix}.self_attn.k_norm.weight"
                    ))?,
                }),
                mlp: DenseMlpWeights {
                    gate: packed_projection(&format!("model.layers.{layer_index}.mlp.gate_proj"))?,
                    up: packed_projection(&format!("model.layers.{layer_index}.mlp.up_proj"))?,
                    down: packed_projection(&format!("model.layers.{layer_index}.mlp.down_proj"))?,
                },
            },
            other_layer_type => {
                return Err(PrismHadamardModelError::UnsupportedLayerType {
                    layer_type: other_layer_type.to_owned(),
                });
            }
        };
        decoder_layers.push(decoder_layer);
    }

    let final_normalization = take_cast("language_model.model.norm.weight")?;
    let language_model_head = packed_projection("lm_head")?;

    Ok(PrismHadamardWeights {
        embedding,
        decoder_layers,
        input_normalization_weights,
        post_attention_normalization_weights,
        final_normalization,
        language_model_head,
    })
}

/// The GDN query/key normalization scales folded into per-channel weights,
/// mirroring the core's construction: every channel carries `1/hd` for
/// queries and `1/sqrt(hd)` for keys through the activation dtype rounding.
pub(crate) fn build_gdn_normalization_scale_weights(
    runtime: &MlxRuntime,
    head_dimension: u32,
) -> Result<(MlxArray, MlxArray), PrismHadamardModelError> {
    let head_dimension_i32 = head_dimension as i32;
    let activation_dtype = super::fwht::activation_dtype();
    let query_scale_value = 1.0 / head_dimension as f32;
    let key_scale_value = 1.0 / (head_dimension as f32).sqrt();
    let query_scale_weight = runtime
        .array_from_f32(
            &vec![query_scale_value; head_dimension_i32 as usize],
            &[head_dimension_i32],
        )
        .map_err(PrismHadamardModelError::WeightCasting)
        .and_then(|float32_weight| {
            runtime
                .astype(&float32_weight, activation_dtype)
                .map_err(PrismHadamardModelError::WeightCasting)
        })?;
    let key_scale_weight = runtime
        .array_from_f32(
            &vec![key_scale_value; head_dimension_i32 as usize],
            &[head_dimension_i32],
        )
        .map_err(PrismHadamardModelError::WeightCasting)
        .and_then(|float32_weight| {
            runtime
                .astype(&float32_weight, activation_dtype)
                .map_err(PrismHadamardModelError::WeightCasting)
        })?;
    Ok((query_scale_weight, key_scale_weight))
}

/// The shared last-dimension slice helper re-exported for the forward pass.
pub(crate) fn slice_last_dimension(
    runtime: &MlxRuntime,
    array: &MlxArray,
    start: i32,
    end: i32,
) -> Result<MlxArray, PrismHadamardModelError> {
    tensor_slicing::slice_last_dimension(runtime, array, start, end).map_err(
        |execution_error: Qwen3_5ExecutionError| PrismHadamardModelError::WeightSlicing {
            description: execution_error.to_string(),
        },
    )
}
