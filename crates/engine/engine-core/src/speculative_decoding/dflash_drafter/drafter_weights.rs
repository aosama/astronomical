//! The DFlash2 drafter's tensors, bound by their checkpoint names with every
//! shape checked against the configuration before any is used.

use mlx_bridge::MlxArray;
use runtime_integration::{MlxRuntime, MlxRuntimeError, MlxSafetensors};
use thiserror::Error;

use super::drafter_configuration::{
    CandidateSelectorShape, DFlashDrafterConfiguration, GroupedConvolutionShape,
};
use super::drafter_linear::DrafterLinear;

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
            .tensor("candidate_selector.predecessor_codebook", &codebook_shape)?,
        successor_codebook: binder
            .tensor("candidate_selector.successor_codebook", &codebook_shape)?,
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

    fn linear(
        &self,
        tensor_name: &str,
        expected_shape: &[i32],
    ) -> Result<DrafterLinear, DFlashDrafterWeightsError> {
        let weight = self.tensor(tensor_name, expected_shape)?;
        Ok(DrafterLinear::full_precision(self.runtime, &weight)?)
    }
}

/// The most a drafter's load holds at once, as a share of its checkpoint
/// file, when the checkpoint is let go right after binding so each linear's
/// full-precision copy is freed once it is requantized: the 4-bit linears
/// (9/32 of their bf16 bytes with group-64 scales and biases), the tensors
/// that stay bf16, and one linear still in bf16. Measured at 0.32 of the
/// checkpoint on both shipped drafters (`dflash_drafter_load_memory`); 3/8
/// leaves room for a drafter whose largest linear is a larger share.
const LOAD_PEAK_EIGHTHS_OF_CHECKPOINT: u64 = 3;

/// What loading a drafter whose checkpoint file is `checkpoint_bytes` holds at
/// its peak, so a load check asks for that room and no more.
#[must_use]
pub fn drafter_load_peak_bytes(checkpoint_bytes: u64) -> u64 {
    checkpoint_bytes.div_ceil(8) * LOAD_PEAK_EIGHTHS_OF_CHECKPOINT
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

#[cfg(test)]
#[path = "drafter_weights_tests.rs"]
mod drafter_weights_tests;
