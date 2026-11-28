//! The DFlash2 drafter's tensors, bound by their checkpoint names with every
//! shape checked against the configuration before any is used.

use mlx_bridge::{MlxArray, MlxDtype};
use runtime_integration::{MlxRuntime, MlxRuntimeError, MlxSafetensors};
use thiserror::Error;

use super::drafter_checkpoint_format::AffinePacking;
use super::drafter_configuration::{
    CandidateSelectorShape, DFlashDrafterConfiguration, GroupedConvolutionShape,
};
use super::drafter_linear::{DrafterLinear, QUANTIZATION_BITS, QUANTIZATION_GROUP_SIZE};

/// Why a drafter checkpoint could not be bound.
#[derive(Debug, Error)]
pub enum DFlashDrafterWeightsError {
    #[error("the drafter checkpoint has no tensor `{tensor_name}`: {source}")]
    MissingTensor {
        tensor_name: String,
        source: MlxRuntimeError,
    },
    #[error("drafter tensor `{tensor_name}` is {actual_shape:?}, expected {expected_shape:?}")]
    Shape {
        tensor_name: String,
        expected_shape: Vec<i32>,
        actual_shape: Vec<i32>,
    },
    #[error(
        "drafter tensor `{tensor_name}` is packed at {bits} bits in groups of {group_size}; \
         a drafter linear is served at 4 bits in groups of 64"
    )]
    Packing {
        tensor_name: String,
        bits: i32,
        group_size: i32,
    },
    #[error(transparent)]
    Runtime(#[from] MlxRuntimeError),
}

/// One grouped dynamic causal convolution: a fixed kernel per stage plus a
/// kernel the hidden row predicts, `[2 stages, kernel_size, hidden]`.
pub struct GroupedConvolutionWeights {
    pub base_kernel: MlxArray,
    pub kernel_projection: DrafterLinear,
    pub shape: GroupedConvolutionShape,
}

/// A DFlash2 layer's two convolutions, around its attention and its MLP.
pub struct LayerConvolutionWeights {
    pub attention: GroupedConvolutionWeights,
    pub mlp: GroupedConvolutionWeights,
}

pub struct DrafterLayerWeights {
    pub input_normalization: MlxArray,
    pub post_attention_normalization: MlxArray,
    pub query_projection: DrafterLinear,
    pub key_projection: DrafterLinear,
    pub value_projection: DrafterLinear,
    pub output_projection: DrafterLinear,
    pub query_head_normalization: MlxArray,
    pub key_head_normalization: MlxArray,
    pub gate_projection: DrafterLinear,
    pub up_projection: DrafterLinear,
    pub down_projection: DrafterLinear,
    /// DFlash2 only; a DFlash layer is attention and MLP alone.
    pub convolutions: Option<LayerConvolutionWeights>,
}

/// The selector's low-rank score of how well one token follows another at a
/// given block position.
pub struct CandidateSelectorWeights {
    pub hidden_projection: DrafterLinear,
    /// `[vocabulary, rank]`, gathered by the previous token on the path.
    pub predecessor_codebook: MlxArray,
    /// `[vocabulary, rank]`, gathered by each candidate token.
    pub successor_codebook: MlxArray,
}

pub struct DFlashDrafterWeights {
    /// `fc`: the concatenated target taps down to the drafter's width.
    pub tap_fusion: DrafterLinear,
    pub tap_fusion_normalization: MlxArray,
    pub layers: Vec<DrafterLayerWeights>,
    pub final_normalization: MlxArray,
    /// DFlash2 only.
    pub candidate_selector: Option<CandidateSelectorWeights>,
}

/// Binds every tensor of the drafter from an opened checkpoint.
pub fn bind_drafter_weights(
    runtime: &MlxRuntime,
    checkpoint: &MlxSafetensors,
    configuration: &DFlashDrafterConfiguration,
) -> Result<DFlashDrafterWeights, DFlashDrafterWeightsError> {
    let binder = TensorBinder {
        runtime,
        checkpoint,
    };
    let hidden = configuration.hidden_size;
    let mut layers = Vec::with_capacity(configuration.layer_count as usize);
    for layer_index in 0..configuration.layer_count {
        layers.push(bind_layer(
            &binder,
            configuration,
            &format!("layers.{layer_index}"),
        )?);
    }
    Ok(DFlashDrafterWeights {
        tap_fusion: binder.linear("fc.weight", &[hidden, configuration.tap_width()])?,
        tap_fusion_normalization: binder.tensor("hidden_norm.weight", &[hidden])?,
        layers,
        final_normalization: binder.tensor("norm.weight", &[hidden])?,
        candidate_selector: match configuration.candidate_selector {
            Some(selector_shape) => Some(bind_candidate_selector(
                &binder,
                selector_shape,
                configuration,
            )?),
            None => None,
        },
    })
}

fn bind_candidate_selector(
    binder: &TensorBinder<'_>,
    selector_shape: CandidateSelectorShape,
    configuration: &DFlashDrafterConfiguration,
) -> Result<CandidateSelectorWeights, DFlashDrafterWeightsError> {
    let codebook_shape = [configuration.vocabulary_size, selector_shape.rank];
    Ok(CandidateSelectorWeights {
        hidden_projection: binder.linear(
            "candidate_selector.hidden_projection.weight",
            &[selector_shape.rank, configuration.hidden_size],
        )?,
        predecessor_codebook: binder
            .codebook("candidate_selector.predecessor_codebook", &codebook_shape)?,
        successor_codebook: binder
            .codebook("candidate_selector.successor_codebook", &codebook_shape)?,
    })
}

fn bind_layer(
    binder: &TensorBinder<'_>,
    configuration: &DFlashDrafterConfiguration,
    layer_prefix: &str,
) -> Result<DrafterLayerWeights, DFlashDrafterWeightsError> {
    let hidden = configuration.hidden_size;
    let query_width = configuration.attention_head_count * configuration.head_dimension;
    let key_value_width = configuration.key_value_head_count * configuration.head_dimension;
    let intermediate = configuration.intermediate_size;
    let attention = format!("{layer_prefix}.self_attn");
    let mlp = format!("{layer_prefix}.mlp");
    Ok(DrafterLayerWeights {
        input_normalization: binder
            .tensor(&format!("{layer_prefix}.input_layernorm.weight"), &[hidden])?,
        post_attention_normalization: binder.tensor(
            &format!("{layer_prefix}.post_attention_layernorm.weight"),
            &[hidden],
        )?,
        query_projection: binder.linear(
            &format!("{attention}.q_proj.weight"),
            &[query_width, hidden],
        )?,
        key_projection: binder.linear(
            &format!("{attention}.k_proj.weight"),
            &[key_value_width, hidden],
        )?,
        value_projection: binder.linear(
            &format!("{attention}.v_proj.weight"),
            &[key_value_width, hidden],
        )?,
        output_projection: binder.linear(
            &format!("{attention}.o_proj.weight"),
            &[hidden, query_width],
        )?,
        query_head_normalization: binder.tensor(
            &format!("{attention}.q_norm.weight"),
            &[configuration.head_dimension],
        )?,
        key_head_normalization: binder.tensor(
            &format!("{attention}.k_norm.weight"),
            &[configuration.head_dimension],
        )?,
        gate_projection: binder
            .linear(&format!("{mlp}.gate_proj.weight"), &[intermediate, hidden])?,
        up_projection: binder.linear(&format!("{mlp}.up_proj.weight"), &[intermediate, hidden])?,
        down_projection: binder
            .linear(&format!("{mlp}.down_proj.weight"), &[hidden, intermediate])?,
        convolutions: bind_layer_convolutions(binder, configuration, layer_prefix)?,
    })
}

/// A DFlash2 layer's two convolutions, or None for a DFlash drafter.
fn bind_layer_convolutions(
    binder: &TensorBinder<'_>,
    configuration: &DFlashDrafterConfiguration,
    layer_prefix: &str,
) -> Result<Option<LayerConvolutionWeights>, DFlashDrafterWeightsError> {
    let Some(convolution_shape) = configuration.convolution else {
        return Ok(None);
    };
    Ok(Some(LayerConvolutionWeights {
        attention: bind_convolution(
            binder,
            convolution_shape,
            &format!("{layer_prefix}.attention_conv"),
        )?,
        mlp: bind_convolution(
            binder,
            convolution_shape,
            &format!("{layer_prefix}.mlp_conv"),
        )?,
    }))
}

fn bind_convolution(
    binder: &TensorBinder<'_>,
    shape: GroupedConvolutionShape,
    convolution_prefix: &str,
) -> Result<GroupedConvolutionWeights, DFlashDrafterWeightsError> {
    // Two stages (before and after the sublayer), each predicting one weight
    // per kernel tap and convolution group.
    let predicted_kernel_width = 2 * shape.kernel_size * shape.group_count();
    Ok(GroupedConvolutionWeights {
        base_kernel: binder.tensor(
            &format!("{convolution_prefix}.base_kernel"),
            &[2, shape.kernel_size, shape.channel_count],
        )?,
        kernel_projection: binder.linear(
            &format!("{convolution_prefix}.kernel_projection.weight"),
            &[predicted_kernel_width, shape.channel_count],
        )?,
        shape,
    })
}

struct TensorBinder<'binding> {
    runtime: &'binding MlxRuntime,
    checkpoint: &'binding MlxSafetensors,
}

impl TensorBinder<'_> {
    fn tensor(
        &self,
        tensor_name: &str,
        expected_shape: &[i32],
    ) -> Result<MlxArray, DFlashDrafterWeightsError> {
        let tensor = self.checkpoint.tensor(tensor_name).map_err(|source| {
            DFlashDrafterWeightsError::MissingTensor {
                tensor_name: tensor_name.to_owned(),
                source,
            }
        })?;
        let actual_shape = tensor.shape();
        if actual_shape != expected_shape {
            return Err(DFlashDrafterWeightsError::Shape {
                tensor_name: tensor_name.to_owned(),
                expected_shape: expected_shape.to_vec(),
                actual_shape,
            });
        }
        Ok(tensor)
    }

    /// A linear as the checkpoint stores it: bf16 `[output, input]`, or MLX's
    /// 4-bit, group-64 packing beside its `scales` and `biases`, as a
    /// drafter published already quantized ships it. A linear packed any
    /// other way is refused: the served linear multiplies at 4 bits, group 64.
    fn linear(
        &self,
        tensor_name: &str,
        expected_shape: &[i32],
    ) -> Result<DrafterLinear, DFlashDrafterWeightsError> {
        let packed_affine = self.packed_affine(tensor_name, expected_shape)?;
        self.linear_as_stored(tensor_name, expected_shape, packed_affine)
    }

    /// The linear served as stored: packed when the checkpoint packed it,
    /// otherwise bf16.
    fn linear_as_stored(
        &self,
        tensor_name: &str,
        expected_shape: &[i32],
        packed_affine: Option<PackedAffineTensors>,
    ) -> Result<DrafterLinear, DFlashDrafterWeightsError> {
        match packed_affine {
            Some(packed_affine) => served_packed_linear(tensor_name, packed_affine),
            None => self.full_precision_linear(tensor_name, expected_shape),
        }
    }

    /// A linear the checkpoint stores in bf16, `[output, input]`.
    fn full_precision_linear(
        &self,
        tensor_name: &str,
        expected_shape: &[i32],
    ) -> Result<DrafterLinear, DFlashDrafterWeightsError> {
        let weight = self.tensor(tensor_name, expected_shape)?;
        Ok(DrafterLinear::full_precision(self.runtime, &weight)?)
    }

    /// A selector codebook `[vocabulary, rank]` in bf16. A checkpoint
    /// published already quantized stores it as `<name>.weight` in MLX's
    /// affine packing (8 bits in the one published so far); it is
    /// dequantized once here, because the selector gathers a few rows of it
    /// a round and reads them as bf16. Dequantizing yields the scales'
    /// dtype, so a pack with float16 scales is cast to bf16 after.
    fn codebook(
        &self,
        tensor_name: &str,
        expected_shape: &[i32],
    ) -> Result<MlxArray, DFlashDrafterWeightsError> {
        let packed_tensor_name = format!("{tensor_name}.weight");
        match self.packed_affine(&packed_tensor_name, expected_shape)? {
            Some(packed_affine) => {
                let dequantized = self
                    .runtime
                    .dequantize_affine(
                        &packed_affine.packed_weight,
                        &packed_affine.scales,
                        &packed_affine.biases,
                        packed_affine.group_size,
                        packed_affine.bits,
                    )
                    .map_err(MlxRuntimeError::from)?;
                Ok(in_bfloat16(self.runtime, dequantized)?)
            }
            None => self.tensor(tensor_name, expected_shape),
        }
    }

    /// The affine packing of the module whose weight is `tensor_name`, when
    /// the checkpoint holds the module's `scales`; `None` when it holds the
    /// module unquantized. `logical_shape` is the unpacked `[output, input]`;
    /// the bits and group size are read from the packed tensors' widths.
    fn packed_affine(
        &self,
        tensor_name: &str,
        logical_shape: &[i32],
    ) -> Result<Option<PackedAffineTensors>, DFlashDrafterWeightsError> {
        let module_name = tensor_name.strip_suffix(".weight").unwrap_or(tensor_name);
        let scales_name = format!("{module_name}.scales");
        let Ok(scales) = self.checkpoint.tensor(&scales_name) else {
            return Ok(None);
        };
        let packed_weight = self.checkpoint.tensor(tensor_name).map_err(|source| {
            DFlashDrafterWeightsError::MissingTensor {
                tensor_name: tensor_name.to_owned(),
                source,
            }
        })?;
        let packing = AffinePacking::read(logical_shape, &packed_weight.shape(), &scales.shape())
            .ok_or_else(|| DFlashDrafterWeightsError::Shape {
            tensor_name: tensor_name.to_owned(),
            expected_shape: logical_shape.to_vec(),
            actual_shape: packed_weight.shape(),
        })?;
        Ok(Some(PackedAffineTensors {
            packed_weight,
            biases: self.tensor(&format!("{module_name}.biases"), &scales.shape())?,
            scales,
            bits: packing.bits,
            group_size: packing.group_size,
        }))
    }
}

/// A linear stored packed, served as stored when it is packed the one way a
/// drafter linear multiplies.
fn served_packed_linear(
    tensor_name: &str,
    packed_affine: PackedAffineTensors,
) -> Result<DrafterLinear, DFlashDrafterWeightsError> {
    if packed_affine.bits != QUANTIZATION_BITS
        || packed_affine.group_size != QUANTIZATION_GROUP_SIZE
    {
        return Err(DFlashDrafterWeightsError::Packing {
            tensor_name: tensor_name.to_owned(),
            bits: packed_affine.bits,
            group_size: packed_affine.group_size,
        });
    }
    Ok(DrafterLinear::Quantized {
        packed_weight: packed_affine.packed_weight,
        scales: packed_affine.scales,
        biases: packed_affine.biases,
    })
}

/// `tensor` in bf16, the dtype the drafter's weights are stored and run in.
pub(super) fn in_bfloat16(
    runtime: &MlxRuntime,
    tensor: MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    if tensor.dtype() == MlxDtype::BFloat16 {
        return Ok(tensor);
    }
    Ok(runtime.astype(&tensor, MlxDtype::BFloat16)?)
}

/// One module in MLX's affine packing, as a checkpoint stores it.
struct PackedAffineTensors {
    packed_weight: MlxArray,
    scales: MlxArray,
    biases: MlxArray,
    bits: i32,
    group_size: i32,
}

impl DFlashDrafterWeights {
    /// Quantizes every linear layer to 4 bits, group 64, in place: about
    /// 1 GB to stream a round instead of the checkpoint's 3.6 GB for the 27B's
    /// drafter. The selector's codebooks stay bf16: they are gathered, a few
    /// rows a round, never streamed whole. `use_register_order` converts
    /// them into register order for the register-ordered matmul; pass it only
    /// when this GPU's capability probe has proven the kernel.
    pub fn quantize_linears(
        &mut self,
        runtime: &MlxRuntime,
        use_register_order: bool,
    ) -> Result<(), MlxRuntimeError> {
        self.tap_fusion
            .quantize_in_place(runtime, use_register_order)?;
        if let Some(candidate_selector) = self.candidate_selector.as_mut() {
            candidate_selector.quantize_projection(runtime, use_register_order)?;
        }
        for layer in &mut self.layers {
            layer.quantize_linears(runtime, use_register_order)?;
        }
        Ok(())
    }

    /// Everything a bound drafter needs before it is hosted: its linears
    /// quantized, then the tensors served as stored read into memory, so
    /// what the host measures across the load is all the drafter holds.
    pub fn prepare_for_serving(
        &mut self,
        runtime: &MlxRuntime,
        use_register_order: bool,
    ) -> Result<(), MlxRuntimeError> {
        self.quantize_linears(runtime, use_register_order)?;
        self.read_tensors_served_as_stored(runtime)
    }

    /// Reads into memory every tensor served as the checkpoint stores it:
    /// the selector's codebooks, the normalizations and the convolutions'
    /// base kernels.
    ///
    /// A bound tensor is read only when something first evaluates it, and
    /// quantizing reads the linears alone. Left bound, the two codebooks
    /// (254 MB for a 248,320-token vocabulary at rank 256) arrived in the
    /// first draft round, after the host had measured what the drafter
    /// holds: the 35B-A3B's ledger counted 224 MB for a drafter holding
    /// about 480 MB, kept experts in the rest, and replies at an 8 GiB limit
    /// ended with the limit exceeded.
    pub fn read_tensors_served_as_stored(
        &self,
        runtime: &MlxRuntime,
    ) -> Result<(), MlxRuntimeError> {
        runtime.evaluate_arrays(&self.tensors_served_as_stored())?;
        Ok(())
    }

    /// Every tensor [`Self::read_tensors_served_as_stored`] reads.
    #[must_use]
    pub fn tensors_served_as_stored(&self) -> Vec<&MlxArray> {
        let mut stored_tensors = vec![&self.tap_fusion_normalization, &self.final_normalization];
        if let Some(candidate_selector) = self.candidate_selector.as_ref() {
            stored_tensors.push(&candidate_selector.predecessor_codebook);
            stored_tensors.push(&candidate_selector.successor_codebook);
        }
        for layer in &self.layers {
            stored_tensors.extend([
                &layer.input_normalization,
                &layer.post_attention_normalization,
                &layer.query_head_normalization,
                &layer.key_head_normalization,
            ]);
            if let Some(convolutions) = layer.convolutions.as_ref() {
                stored_tensors.push(&convolutions.attention.base_kernel);
                stored_tensors.push(&convolutions.mlp.base_kernel);
            }
        }
        stored_tensors
    }
}

impl DrafterLayerWeights {
    fn quantize_linears(
        &mut self,
        runtime: &MlxRuntime,
        use_register_order: bool,
    ) -> Result<(), MlxRuntimeError> {
        for linear in [
            &mut self.query_projection,
            &mut self.key_projection,
            &mut self.value_projection,
            &mut self.output_projection,
            &mut self.gate_projection,
            &mut self.up_projection,
            &mut self.down_projection,
        ] {
            linear.quantize_in_place(runtime, use_register_order)?;
        }
        if let Some(convolutions) = self.convolutions.as_mut() {
            convolutions.quantize_kernel_projections(runtime, use_register_order)?;
        }
        Ok(())
    }
}

impl CandidateSelectorWeights {
    /// Quantizes the hidden projection; the codebooks stay bf16.
    fn quantize_projection(
        &mut self,
        runtime: &MlxRuntime,
        use_register_order: bool,
    ) -> Result<(), MlxRuntimeError> {
        self.hidden_projection
            .quantize_in_place(runtime, use_register_order)
    }
}

impl LayerConvolutionWeights {
    /// Quantizes both convolutions' kernel predictions; the fixed kernels
    /// stay as the checkpoint holds them.
    fn quantize_kernel_projections(
        &mut self,
        runtime: &MlxRuntime,
        use_register_order: bool,
    ) -> Result<(), MlxRuntimeError> {
        self.attention
            .kernel_projection
            .quantize_in_place(runtime, use_register_order)?;
        self.mlp
            .kernel_projection
            .quantize_in_place(runtime, use_register_order)
    }
}
