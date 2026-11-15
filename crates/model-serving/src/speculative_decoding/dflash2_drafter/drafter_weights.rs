//! The DFlash2 drafter's tensors, bound by their checkpoint names with every
//! shape checked against the configuration before any is used.

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError, MlxSafetensors};
use mlx_bridge::MlxArray;
use thiserror::Error;

use super::drafter_configuration::DFlash2DrafterConfiguration;
use super::drafter_linear::DrafterLinear;

/// Why a drafter checkpoint could not be bound.
#[derive(Debug, Error)]
pub enum DFlash2DrafterWeightsError {
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
    pub attention_convolution: GroupedConvolutionWeights,
    pub mlp_convolution: GroupedConvolutionWeights,
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

pub struct DFlash2DrafterWeights {
    /// `fc`: the concatenated target taps down to the drafter's width.
    pub tap_fusion: DrafterLinear,
    pub tap_fusion_normalization: MlxArray,
    pub layers: Vec<DrafterLayerWeights>,
    pub final_normalization: MlxArray,
    pub candidate_selector: CandidateSelectorWeights,
}

/// Binds every tensor of the drafter from an opened checkpoint.
pub fn bind_drafter_weights(
    runtime: &MlxRuntime,
    checkpoint: &MlxSafetensors,
    configuration: &DFlash2DrafterConfiguration,
) -> Result<DFlash2DrafterWeights, DFlash2DrafterWeightsError> {
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
    Ok(DFlash2DrafterWeights {
        tap_fusion: binder.linear("fc.weight", &[hidden, configuration.tap_width()])?,
        tap_fusion_normalization: binder.tensor("hidden_norm.weight", &[hidden])?,
        layers,
        final_normalization: binder.tensor("norm.weight", &[hidden])?,
        candidate_selector: CandidateSelectorWeights {
            hidden_projection: binder.linear(
                "candidate_selector.hidden_projection.weight",
                &[configuration.selector_rank, hidden],
            )?,
            predecessor_codebook: binder.tensor(
                "candidate_selector.predecessor_codebook",
                &[configuration.vocabulary_size, configuration.selector_rank],
            )?,
            successor_codebook: binder.tensor(
                "candidate_selector.successor_codebook",
                &[configuration.vocabulary_size, configuration.selector_rank],
            )?,
        },
    })
}

fn bind_layer(
    binder: &TensorBinder<'_>,
    configuration: &DFlash2DrafterConfiguration,
    layer_prefix: &str,
) -> Result<DrafterLayerWeights, DFlash2DrafterWeightsError> {
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
        attention_convolution: bind_convolution(
            binder,
            configuration,
            &format!("{layer_prefix}.attention_conv"),
        )?,
        mlp_convolution: bind_convolution(
            binder,
            configuration,
            &format!("{layer_prefix}.mlp_conv"),
        )?,
    })
}

fn bind_convolution(
    binder: &TensorBinder<'_>,
    configuration: &DFlash2DrafterConfiguration,
    convolution_prefix: &str,
) -> Result<GroupedConvolutionWeights, DFlash2DrafterWeightsError> {
    let kernel_size = configuration.convolution_kernel_size;
    // Two stages (before and after the sublayer), each predicting one weight
    // per kernel tap and convolution group.
    let predicted_kernel_width = 2 * kernel_size * configuration.convolution_group_count();
    Ok(GroupedConvolutionWeights {
        base_kernel: binder.tensor(
            &format!("{convolution_prefix}.base_kernel"),
            &[2, kernel_size, configuration.hidden_size],
        )?,
        kernel_projection: binder.linear(
            &format!("{convolution_prefix}.kernel_projection.weight"),
            &[predicted_kernel_width, configuration.hidden_size],
        )?,
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
    ) -> Result<MlxArray, DFlash2DrafterWeightsError> {
        let tensor = self.checkpoint.tensor(tensor_name).map_err(|source| {
            DFlash2DrafterWeightsError::MissingTensor {
                tensor_name: tensor_name.to_owned(),
                source,
            }
        })?;
        let actual_shape = tensor.shape();
        if actual_shape != expected_shape {
            return Err(DFlash2DrafterWeightsError::Shape {
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
    ) -> Result<DrafterLinear, DFlash2DrafterWeightsError> {
        let weight = self.tensor(tensor_name, expected_shape)?;
        Ok(DrafterLinear::full_precision(self.runtime, &weight)?)
    }
}

impl DFlash2DrafterWeights {
    /// Quantizes every linear layer to 4 bits, group 64, in place: about
    /// 1 GB to stream a round instead of the checkpoint's 3.6 GB for the 27B's
    /// drafter. The selector's codebooks stay bf16: they are gathered, a few
    /// rows a round, never streamed whole. `use_row_batched_matmul` routes
    /// the 8-row block through the row-batched matmul; pass it only when this
    /// GPU's capability probe has proven the kernel.
    pub fn quantize_linears(
        &mut self,
        runtime: &MlxRuntime,
        use_row_batched_matmul: bool,
    ) -> Result<(), MlxRuntimeError> {
        self.tap_fusion
            .quantize_in_place(runtime, use_row_batched_matmul)?;
        self.candidate_selector
            .hidden_projection
            .quantize_in_place(runtime, use_row_batched_matmul)?;
        for layer in &mut self.layers {
            for linear in [
                &mut layer.query_projection,
                &mut layer.key_projection,
                &mut layer.value_projection,
                &mut layer.output_projection,
                &mut layer.gate_projection,
                &mut layer.up_projection,
                &mut layer.down_projection,
                &mut layer.attention_convolution.kernel_projection,
                &mut layer.mlp_convolution.kernel_projection,
            ] {
                linear.quantize_in_place(runtime, use_row_batched_matmul)?;
            }
        }
        Ok(())
    }
}
