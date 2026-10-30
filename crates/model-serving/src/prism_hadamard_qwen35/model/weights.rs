//! Packed-weight modules and family weight loading.
//!
//! Every rotated projection applies the Hadamard transform unconditionally:
//! a `PackedProjection` runs only on [`RotatedActivations`], which only the
//! transform produces, so there is no code path that could silently skip the
//! transform the pack's contract requires. The transform itself is shared:
//! one [`HadamardRotation`] per activation width, built from the validated
//! contract's sign table exactly as the upstream runtime indexes it.
//!
//! Sibling projections that read one rotated activation are fused at load
//! into one [`FusedPackedProjection`] (see `fused_projection.rs`), and every
//! fused projection of the pack is materialized in one evaluation at the end
//! of loading.

use std::collections::BTreeMap;

use astronomical_runtime_integration::{MlxRuntime, MlxSafetensors};
use mlx_c_rust::{MlxArray, MlxCError, MlxCompiledElementwiseGraphs, MlxMetalKernel};

use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;
use crate::qwen3_5_core::model_math::tensor_slicing;

use super::super::artifacts::ValidatedFoldedPack;
use super::PrismHadamardModelError;
use super::fused_projection::FusedPackedProjection;
use super::fwht::{HadamardRotation, RotatedActivations};

/// The ternary quantization constants of the family's packs.
const QUANTIZATION_GROUP_SIZE: i32 = 128;
const QUANTIZATION_BITS: i32 = 2;

/// One rotated ternary projection: grouped affine quantized matmul against
/// the transposed packed weights, over activations already carried into the
/// rotated basis at this projection's input width.
#[derive(Debug)]
pub(crate) struct PackedProjection {
    packed_weight: MlxArray,
    quantization_scales: MlxArray,
    quantization_biases: MlxArray,
    input_width: i32,
}

impl PackedProjection {
    pub(crate) fn forward(
        &self,
        runtime: &MlxRuntime,
        rotated_activations: &RotatedActivations,
    ) -> Result<MlxArray, MlxCError> {
        if rotated_activations.width() != self.input_width {
            return Err(MlxCError {
                operation: "apply a rotated ternary projection",
                description: format!(
                    "activations rotated at width {} cannot feed a projection of input width {}",
                    rotated_activations.width(),
                    self.input_width
                ),
            });
        }
        runtime.quantized_matmul_affine(
            rotated_activations.activations(),
            &self.packed_weight,
            &self.quantization_scales,
            &self.quantization_biases,
            true,
            QUANTIZATION_GROUP_SIZE,
            QUANTIZATION_BITS,
        )
    }

    pub(crate) const fn input_width(&self) -> i32 {
        self.input_width
    }

    pub(crate) const fn new(
        packed_weight: MlxArray,
        quantization_scales: MlxArray,
        quantization_biases: MlxArray,
        input_width: i32,
    ) -> Self {
        Self {
            packed_weight,
            quantization_scales,
            quantization_biases,
            input_width,
        }
    }

    pub(crate) const fn packed_weight(&self) -> &MlxArray {
        &self.packed_weight
    }

    pub(crate) const fn quantization_scales(&self) -> &MlxArray {
        &self.quantization_scales
    }

    pub(crate) const fn quantization_biases(&self) -> &MlxArray {
        &self.quantization_biases
    }

    /// How many output rows (columns of the projected activation) the packed
    /// matrix produces.
    pub(crate) fn output_rows(&self) -> i32 {
        self.packed_weight.shape()[0]
    }
}

/// The packed token embedding: gather rows, dequantize, then the inverse
/// Hadamard transform restores the unrotated hidden basis.
#[derive(Debug)]
pub(crate) struct PackedEmbedding {
    packed_weight: MlxArray,
    quantization_scales: MlxArray,
    quantization_biases: MlxArray,
}

impl PackedEmbedding {
    pub(crate) fn lookup(
        &self,
        runtime: &MlxRuntime,
        rotation: &HadamardRotation,
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
        rotation.rotate_inverse(runtime, &dequantized)
    }
}

/// The gated-delta update-rate (`in_proj_b`) and decay-interval
/// (`in_proj_a`) float projections fused into one `[hidden, 2 * heads]`
/// matmul. Both read the same normalized input and both are unrotated, so
/// one launch produces both and two views split it. The weight is stored
/// already transposed and materialized at load, so no per-call transpose
/// node enters the graph.
#[derive(Debug)]
pub(crate) struct GatedDeltaRateProjection {
    transposed_weight: MlxArray,
    head_count: i32,
}

/// The two gated-delta rate pre-activations one fused launch produces.
pub(crate) struct GatedDeltaRateLogits {
    pub(crate) update_logits: MlxArray,
    pub(crate) decay_inputs: MlxArray,
}

impl GatedDeltaRateProjection {
    pub(crate) fn forward(
        &self,
        runtime: &MlxRuntime,
        activations: &MlxArray,
    ) -> Result<GatedDeltaRateLogits, MlxCError> {
        let fused_logits = runtime.matmul(activations, &self.transposed_weight)?;
        let update_logits = slice_last_dimension_raw(runtime, &fused_logits, 0, self.head_count)?;
        let decay_inputs =
            slice_last_dimension_raw(runtime, &fused_logits, self.head_count, 2 * self.head_count)?;
        Ok(GatedDeltaRateLogits {
            update_logits,
            decay_inputs,
        })
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

/// One dense SwiGLU MLP's rotated projections: the gate and up projections
/// fused over the normalized input, and the down projection.
#[derive(Debug)]
pub(crate) struct DenseMlpWeights {
    /// Members in order: gate, up.
    pub(crate) gate_and_up: FusedPackedProjection<2>,
    pub(crate) down: PackedProjection,
}

#[derive(Debug)]
pub(crate) struct LinearLayerWeights {
    /// Members in order: the mixed query/key/value projection
    /// (`in_proj_qkv`), the output-gate projection (`in_proj_z`).
    pub(crate) query_key_value_and_gate: FusedPackedProjection<2>,
    pub(crate) output: PackedProjection,
    pub(crate) rates: GatedDeltaRateProjection,
    pub(crate) convolution: MlxArray,
    pub(crate) decay_rate_logarithm: MlxArray,
    pub(crate) decay_interval_bias: MlxArray,
    pub(crate) normalization: MlxArray,
}

#[derive(Debug)]
pub(crate) struct FullLayerWeights {
    /// Members in order: query (with its interleaved output gate), key, value.
    pub(crate) query_key_value: FusedPackedProjection<3>,
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
    /// One activation-side rotation per input width the pack rotates.
    rotations_by_width: BTreeMap<i32, HadamardRotation>,
}

impl PrismHadamardWeights {
    /// The rotation for activations of `width`; every width a packed
    /// projection consumes was registered at load, so a miss is a bug.
    pub(crate) fn rotation(&self, width: i32) -> Result<&HadamardRotation, MlxCError> {
        self.rotations_by_width
            .get(&width)
            .ok_or_else(|| MlxCError {
                operation: "select a Hadamard rotation",
                description: format!("no sign vector is registered for activation width {width}"),
            })
    }

    /// Carries `activations` into the rotated basis for every projection of
    /// its width, through the one-launch kernel when the model retains one.
    pub(crate) fn rotate(
        &self,
        runtime: &MlxRuntime,
        compiled_elementwise_graphs: &MlxCompiledElementwiseGraphs,
        fused_rotation_kernel: Option<&MlxMetalKernel>,
        activations: &MlxArray,
    ) -> Result<RotatedActivations, MlxCError> {
        let width = *activations
            .shape()
            .last()
            .expect("activations always have at least one axis");
        self.rotation(width)?.rotate(
            runtime,
            compiled_elementwise_graphs,
            fused_rotation_kernel,
            activations,
        )
    }
}

/// The tensor-reading context one family pack loads through: the runtime for
/// casting and evaluation, the safetensors map, and the reviewed activation
/// dtype. The former load closures become its methods.
struct FamilyTensorSource<'a> {
    runtime: &'a MlxRuntime,
    loaded_tensors: &'a MlxSafetensors,
    activation_dtype: mlx_c_rust::MlxDtype,
}

impl<'a> FamilyTensorSource<'a> {
    fn take(&self, tensor_name: &str) -> Result<MlxArray, PrismHadamardModelError> {
        self.loaded_tensors.tensor(tensor_name).map_err(|source| {
            PrismHadamardModelError::TensorLookup {
                tensor_name: tensor_name.to_owned(),
                source,
            }
        })
    }

    fn take_cast(&self, tensor_name: &str) -> Result<MlxArray, PrismHadamardModelError> {
        let array = self.take(tensor_name)?;
        self.runtime
            .astype(&array, self.activation_dtype)
            .map_err(PrismHadamardModelError::WeightCasting)
    }

    fn packed_projection(
        &self,
        module_path: &str,
    ) -> Result<PackedProjection, PrismHadamardModelError> {
        let stem = format!("language_model.{module_path}");
        let packed_weight = self.take(&format!("{stem}.weight"))?;
        let quantization_scales = self.take(&format!("{stem}.scales"))?;
        let quantization_biases = self.take(&format!("{stem}.biases"))?;
        let packed_shape = packed_weight.shape();
        let input_width = packed_shape[packed_shape.len() - 1] * 32 / QUANTIZATION_BITS;
        Ok(PackedProjection {
            packed_weight,
            quantization_scales,
            quantization_biases,
            input_width,
        })
    }

    fn rate_projection(
        &self,
        layer_prefix: &str,
    ) -> Result<GatedDeltaRateProjection, PrismHadamardModelError> {
        let update_rate_weight =
            self.take_cast(&format!("{layer_prefix}.linear_attn.in_proj_b.weight"))?;
        let decay_interval_weight =
            self.take_cast(&format!("{layer_prefix}.linear_attn.in_proj_a.weight"))?;
        let head_count = update_rate_weight.shape()[0];
        let fused_weight = self
            .runtime
            .concatenate_axis(&[&update_rate_weight, &decay_interval_weight], 0)
            .map_err(PrismHadamardModelError::WeightCasting)?;
        let transposed_weight = self
            .runtime
            .transpose_axes(&fused_weight, &[1, 0])
            .and_then(|transposed| self.runtime.build_contiguous_row_major_copy(&transposed))
            .map_err(PrismHadamardModelError::WeightCasting)?;
        self.runtime
            .evaluate_arrays(&[&transposed_weight])
            .map_err(PrismHadamardModelError::WeightCasting)?;
        Ok(GatedDeltaRateProjection {
            transposed_weight,
            head_count,
        })
    }
}

pub(crate) fn load_family_weights(
    runtime: &MlxRuntime,
    pack: &ValidatedFoldedPack,
    loaded_tensors: &MlxSafetensors,
) -> Result<PrismHadamardWeights, PrismHadamardModelError> {
    let text = pack.config().text();
    let tensor_source = FamilyTensorSource {
        runtime,
        loaded_tensors,
        activation_dtype: super::fwht::activation_dtype(),
    };

    let embedding_stem = "language_model.model.embed_tokens";
    let embedding = PackedEmbedding {
        packed_weight: tensor_source.take(&format!("{embedding_stem}.weight"))?,
        quantization_scales: tensor_source.take(&format!("{embedding_stem}.scales"))?,
        quantization_biases: tensor_source.take(&format!("{embedding_stem}.biases"))?,
    };

    let mut decoder_layers = Vec::with_capacity(text.num_hidden_layers as usize);
    let mut input_normalization_weights = Vec::with_capacity(text.num_hidden_layers as usize);
    let mut post_attention_normalization_weights =
        Vec::with_capacity(text.num_hidden_layers as usize);
    for (layer_index, layer_type) in text.layer_types.iter().enumerate() {
        let layer_prefix = format!("language_model.model.layers.{layer_index}");
        input_normalization_weights
            .push(tensor_source.take_cast(&format!("{layer_prefix}.input_layernorm.weight"))?);
        post_attention_normalization_weights.push(
            tensor_source.take_cast(&format!("{layer_prefix}.post_attention_layernorm.weight"))?,
        );
        decoder_layers.push(load_decoder_layer_weights(
            &tensor_source,
            layer_index,
            layer_type.as_str(),
            &layer_prefix,
        )?);
    }

    let final_normalization = tensor_source.take_cast("language_model.model.norm.weight")?;
    let language_model_head = tensor_source.packed_projection("lm_head")?;

    // Materialize every fused projection in one evaluation: one
    // synchronization for the whole pack instead of one per layer, and the
    // concatenations never enter a forward graph.
    materialize_stacked_projection_arrays(runtime, &decoder_layers)?;

    let hadamard_contract = pack.hadamard_contract();
    let rotations_by_width = register_family_rotation_widths(
        runtime,
        hadamard_contract,
        text.hidden_size as i32,
        language_model_head.input_width(),
        &decoder_layers,
    )?;

    Ok(PrismHadamardWeights {
        embedding,
        decoder_layers,
        input_normalization_weights,
        post_attention_normalization_weights,
        final_normalization,
        language_model_head,
        rotations_by_width,
    })
}

/// Loads one decoder layer's attention (linear or full, by the layer-type
/// declaration) and its dense MLP.
fn load_decoder_layer_weights(
    tensor_source: &FamilyTensorSource<'_>,
    layer_index: usize,
    layer_type: &str,
    layer_prefix: &str,
) -> Result<DecoderLayerWeights, PrismHadamardModelError> {
    let mlp = DenseMlpWeights {
        gate_and_up: FusedPackedProjection::fuse(
            tensor_source.runtime,
            [
                tensor_source
                    .packed_projection(&format!("model.layers.{layer_index}.mlp.gate_proj"))?,
                tensor_source
                    .packed_projection(&format!("model.layers.{layer_index}.mlp.up_proj"))?,
            ],
        )
        .map_err(PrismHadamardModelError::WeightCasting)?,
        down: tensor_source
            .packed_projection(&format!("model.layers.{layer_index}.mlp.down_proj"))?,
    };
    let attention = match layer_type {
        "linear_attention" => DecoderLayerAttentionWeights::Linear(LinearLayerWeights {
            query_key_value_and_gate: FusedPackedProjection::fuse(
                tensor_source.runtime,
                [
                    tensor_source.packed_projection(&format!(
                        "model.layers.{layer_index}.linear_attn.in_proj_qkv"
                    ))?,
                    tensor_source.packed_projection(&format!(
                        "model.layers.{layer_index}.linear_attn.in_proj_z"
                    ))?,
                ],
            )
            .map_err(PrismHadamardModelError::WeightCasting)?,
            output: tensor_source
                .packed_projection(&format!("model.layers.{layer_index}.linear_attn.out_proj"))?,
            rates: tensor_source.rate_projection(layer_prefix)?,
            convolution: tensor_source
                .take_cast(&format!("{layer_prefix}.linear_attn.conv1d.weight"))?,
            decay_rate_logarithm: tensor_source
                .take_cast(&format!("{layer_prefix}.linear_attn.A_log"))?,
            decay_interval_bias: tensor_source
                .take_cast(&format!("{layer_prefix}.linear_attn.dt_bias"))?,
            normalization: tensor_source
                .take_cast(&format!("{layer_prefix}.linear_attn.norm.weight"))?,
        }),
        "full_attention" => DecoderLayerAttentionWeights::Full(FullLayerWeights {
            query_key_value: FusedPackedProjection::fuse(
                tensor_source.runtime,
                [
                    tensor_source.packed_projection(&format!(
                        "model.layers.{layer_index}.self_attn.q_proj"
                    ))?,
                    tensor_source.packed_projection(&format!(
                        "model.layers.{layer_index}.self_attn.k_proj"
                    ))?,
                    tensor_source.packed_projection(&format!(
                        "model.layers.{layer_index}.self_attn.v_proj"
                    ))?,
                ],
            )
            .map_err(PrismHadamardModelError::WeightCasting)?,
            output: tensor_source
                .packed_projection(&format!("model.layers.{layer_index}.self_attn.o_proj"))?,
            query_normalization: tensor_source
                .take_cast(&format!("{layer_prefix}.self_attn.q_norm.weight"))?,
            key_normalization: tensor_source
                .take_cast(&format!("{layer_prefix}.self_attn.k_norm.weight"))?,
        }),
        other_layer_type => {
            return Err(PrismHadamardModelError::UnsupportedLayerType {
                layer_type: other_layer_type.to_owned(),
            });
        }
    };
    Ok(DecoderLayerWeights { attention, mlp })
}

/// Materializes every fused projection in one evaluation: one
/// synchronization for the whole pack instead of one per layer, and the
/// concatenations never enter a forward graph.
fn materialize_stacked_projection_arrays(
    runtime: &MlxRuntime,
    decoder_layers: &[DecoderLayerWeights],
) -> Result<(), PrismHadamardModelError> {
    let stacked_arrays: Vec<&MlxArray> = decoder_layers
        .iter()
        .flat_map(|decoder_layer| {
            let attention_arrays = match &decoder_layer.attention {
                DecoderLayerAttentionWeights::Linear(linear_layer) => {
                    linear_layer.query_key_value_and_gate.stacked_arrays()
                }
                DecoderLayerAttentionWeights::Full(full_layer) => {
                    full_layer.query_key_value.stacked_arrays()
                }
            };
            attention_arrays
                .into_iter()
                .chain(decoder_layer.mlp.gate_and_up.stacked_arrays())
        })
        .collect();
    runtime
        .evaluate_arrays(&stacked_arrays)
        .map_err(PrismHadamardModelError::WeightCasting)?;
    Ok(())
}

/// Registers one Hadamard rotation per distinct activation width the family
/// touches: the text hidden size, the language-model head's input width, and
/// every layer's projection widths.
fn register_family_rotation_widths(
    runtime: &MlxRuntime,
    hadamard_contract: &crate::prism_hadamard_qwen35::hadamard::HadamardContract,
    text_hidden_size: i32,
    language_model_head_input_width: i32,
    decoder_layers: &[DecoderLayerWeights],
) -> Result<BTreeMap<i32, HadamardRotation>, PrismHadamardModelError> {
    let mut rotations_by_width = BTreeMap::new();
    let mut register_width = |width: i32| -> Result<(), PrismHadamardModelError> {
        if rotations_by_width.contains_key(&width) {
            return Ok(());
        }
        let sign_values = hadamard_contract.sign_vector(width as u32).ok_or_else(|| {
            PrismHadamardModelError::WeightSlicing {
                description: format!(
                    "the hadamard contract declares no sign vector for activation width {width}"
                ),
            }
        })?;
        let rotation = HadamardRotation::new(
            runtime,
            width as u32,
            hadamard_contract.block_size(),
            sign_values,
        )
        .map_err(PrismHadamardModelError::WeightCasting)?;
        rotations_by_width.insert(width, rotation);
        Ok(())
    };
    register_width(text_hidden_size)?;
    register_width(language_model_head_input_width)?;
    for decoder_layer in decoder_layers {
        let input_widths: Vec<i32> = match &decoder_layer.attention {
            DecoderLayerAttentionWeights::Linear(linear_layer) => vec![
                linear_layer.query_key_value_and_gate.input_width(),
                linear_layer.output.input_width(),
            ],
            DecoderLayerAttentionWeights::Full(full_layer) => vec![
                full_layer.query_key_value.input_width(),
                full_layer.output.input_width(),
            ],
        };
        for input_width in input_widths.into_iter().chain([
            decoder_layer.mlp.gate_and_up.input_width(),
            decoder_layer.mlp.down.input_width(),
        ]) {
            register_width(input_width)?;
        }
    }
    Ok(rotations_by_width)
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

pub(super) fn slice_last_dimension_raw(
    runtime: &MlxRuntime,
    array: &MlxArray,
    start: i32,
    end: i32,
) -> Result<MlxArray, MlxCError> {
    tensor_slicing::slice_last_dimension(runtime, array, start, end).map_err(|execution_error| {
        MlxCError {
            operation: "split a fused projection output",
            description: execution_error.to_string(),
        }
    })
}
