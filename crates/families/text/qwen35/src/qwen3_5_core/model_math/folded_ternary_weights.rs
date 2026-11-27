//! A folded ternary pack's weights as the shared core serves them.
//!
//! The core binds a pack's tensors by their Qwen 3.5 names like any
//! checkpoint's. Two things then differ, and this module applies both, to a
//! whole model after binding and to each layer the dense layer streamer
//! reads:
//!
//! - **Rotated modules.** Every module the pack's sign table names was stored
//!   in a rotated basis, so its input takes the signed block Hadamard of its
//!   input width before the matmul (`HadamardRotated` storage), and the
//!   embedding table's rows take the inverse rotation after the lookup.
//! - **Float16 auxiliaries.** The pack stores its norms, convolution, decay
//!   parameters and DeltaNet rate projections in float32 beside float16
//!   modules. Bound as stored they would widen every activation and the
//!   recurrent state to float32; the pack's reference runtime narrows them
//!   to the float16 its modules declare, and so does this.

use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use mlx_bridge::{MlxArray, MlxDtype};
use runtime_integration::MlxRuntime;
use text_generation::execution_error::TextExecutionError;

use super::decoder_layer_weights::{
    Qwen3_5AffineWeights, Qwen3_5AttentionWeights, Qwen3_5DecoderLayerWeights,
};
use super::dense_projection::{
    HadamardRotatedWeights, Qwen3_5DenseProjection, Qwen3_5ProjectionStorage,
};
use super::hadamard_rotation::HadamardRotation;
use super::weights::Qwen3_5Weights;
use crate::prism_hadamard_qwen35::hadamard::HadamardContract;
use crate::qwen3_5_core::configuration::Qwen3_5Config;

/// The dtype a folded pack's modules declare and its activations run in.
const FOLDED_ACTIVATION_DTYPE: MlxDtype = MlxDtype::Float16;
/// The embedding table the sign table names as inverse-rotated.
const EMBEDDING_MODULE_NAME: &str = "language_model.model.embed_tokens";

/// What a folded pack's rotated modules need, built once at load and shared
/// with the dense layer streamer.
#[derive(Debug)]
pub struct FoldedTernaryRotations {
    rotations_by_width: BTreeMap<u32, Rc<HadamardRotation>>,
    rotated_weight_names: BTreeSet<String>,
    hidden_size: u32,
    embedding_is_inverse_rotated: bool,
    /// The pack's Hadamard block, which decides whether the one-launch
    /// signed rotation can reproduce the composed route.
    block_size: u32,
}

impl FoldedTernaryRotations {
    /// One rotation per activation width a rotated module of this model
    /// reads: the hidden width, the attention output's, the DeltaNet value
    /// width and the MLP's intermediate width.
    pub fn from_contract(
        runtime: &MlxRuntime,
        hadamard_contract: &HadamardContract,
        config: &Qwen3_5Config,
    ) -> Result<Self, TextExecutionError> {
        let mut rotations_by_width = BTreeMap::new();
        for activation_width in rotated_activation_widths(config) {
            if rotations_by_width.contains_key(&activation_width) {
                continue;
            }
            let rotation = rotation_at_width(runtime, hadamard_contract, activation_width)?;
            rotations_by_width.insert(activation_width, Rc::new(rotation));
        }
        Ok(Self {
            rotations_by_width,
            rotated_weight_names: hadamard_contract.rotated_tensor_names().clone(),
            hidden_size: config.hidden_size(),
            embedding_is_inverse_rotated: hadamard_contract
                .is_inverse(&format!("{EMBEDDING_MODULE_NAME}.weight")),
            block_size: hadamard_contract.block_size(),
        })
    }

    /// Applies the pack's rotations and float16 auxiliaries to a whole bound
    /// model: every layer, the output head, the final norm and the
    /// embedding table.
    pub fn apply_to_model(
        &self,
        runtime: &MlxRuntime,
        weights: &mut Qwen3_5Weights,
    ) -> Result<(), TextExecutionError> {
        for layer_weights in &mut weights.decoder_layer_weights {
            self.apply_to_layer(runtime, layer_weights)?;
        }
        self.rotate_projection(&mut weights.language_model_head_weights)?;
        weights.final_normalization_weight =
            narrowed(runtime, &weights.final_normalization_weight)?;
        if self.embedding_is_inverse_rotated {
            weights.embedding_rotation = Some(self.rotation_for_width(self.hidden_size)?);
        }
        Ok(())
    }

    /// Applies the pack's rotations and float16 auxiliaries to one bound
    /// decoder layer.
    pub fn apply_to_layer(
        &self,
        runtime: &MlxRuntime,
        layer_weights: &mut Qwen3_5DecoderLayerWeights,
    ) -> Result<(), TextExecutionError> {
        narrow_layer_auxiliaries(runtime, layer_weights)?;
        for projection in
            crate::qwen3_5_core::model::register_ordered_projections::layer_projections(
                &mut layer_weights.attention_weights,
                &mut layer_weights.mlp_weights,
            )
        {
            self.rotate_projection(projection)?;
        }
        Ok(())
    }

    /// Gives `projection` its input rotation when the sign table names it.
    fn rotate_projection(
        &self,
        projection: &mut Qwen3_5DenseProjection,
    ) -> Result<(), TextExecutionError> {
        let weight_name = format!("{}.weight", projection.module_name);
        if !self.rotated_weight_names.contains(&weight_name) {
            return Ok(());
        }
        let Qwen3_5ProjectionStorage::Bound(bound_weights) = &projection.storage else {
            return Ok(());
        };
        let rotation = self.rotation_for_width(stored_input_width(&weight_name, bound_weights)?)?;
        let stored_weights = bound_weights.retained_reference()?;
        projection.storage = Qwen3_5ProjectionStorage::HadamardRotated(HadamardRotatedWeights {
            weights: stored_weights,
            rotation,
        });
        Ok(())
    }

    /// The one-launch signed rotation for this pack's block, when this GPU's
    /// probe found it bit for bit equal to the composed three-launch route;
    /// `None` keeps the composed route, which gives the same bits. Decode on
    /// a folded pack is bound by launches a token, and the composed route is
    /// about 500 of them. A kernel that fails to build costs that speed, never
    /// the model's load.
    pub fn signed_rotation_kernel(
        &self,
        runtime: &MlxRuntime,
        performance_attribution: &mut crate::PerformanceAttribution,
    ) -> Option<mlx_bridge::MlxMetalKernel> {
        let kernel_capabilities = crate::kernel_capability::worker_process_kernel_capabilities(
            runtime,
            performance_attribution,
        );
        let is_retained = kernel_capabilities
            .is_custom_kernel_supported(crate::kernel_capability::PRISM_HADAMARD_SIGNED_ROTATION)
            && super::signed_rotation_kernel::is_signed_rotation_eligible(self.block_size);
        if !is_retained {
            tracing::info!("the signed Hadamard rotation stays on the composed MLX route");
            return None;
        }
        super::signed_rotation_kernel::signed_rotation_kernel(self.block_size)
            .inspect_err(|build_error| {
                tracing::warn!(%build_error, "the signed Hadamard rotation kernel did not build; the composed route serves");
            })
            .ok()
    }

    fn rotation_for_width(
        &self,
        activation_width: u32,
    ) -> Result<Rc<HadamardRotation>, TextExecutionError> {
        self.rotations_by_width
            .get(&activation_width)
            .map(Rc::clone)
            .ok_or_else(|| TextExecutionError::InvalidTensor {
                tensor_name: format!("a rotated module of input width {activation_width}"),
                description: "the folded pack's sign table has no rotation for this width",
            })
    }
}

/// The activation widths a folded Qwen 3.5 model's rotated modules read.
fn rotated_activation_widths(config: &Qwen3_5Config) -> [u32; 4] {
    [
        config.hidden_size(),
        config.query_head_count() * config.head_dimension(),
        config.linear_value_head_count() * config.linear_value_head_dimension(),
        config.dense_intermediate_size(),
    ]
}

fn rotation_at_width(
    runtime: &MlxRuntime,
    hadamard_contract: &HadamardContract,
    activation_width: u32,
) -> Result<HadamardRotation, TextExecutionError> {
    let sign_values = hadamard_contract
        .sign_vector(activation_width)
        .ok_or_else(|| TextExecutionError::InvalidTensor {
            tensor_name: format!("the sign vector of width {activation_width}"),
            description: "the folded pack's sign table has no vector for this width",
        })?;
    Ok(HadamardRotation::new(
        runtime,
        activation_width,
        hadamard_contract.block_size(),
        sign_values,
    )?)
}

/// The input width of a bound module: MLX's affine packing stores
/// `32 / bits` inputs a 32-bit word. A folded pack stores its rotated
/// modules affine-packed or unquantized, never as codebooks.
fn stored_input_width(
    weight_name: &str,
    bound_weights: &Qwen3_5AffineWeights,
) -> Result<u32, TextExecutionError> {
    let stored_width = match bound_weights {
        Qwen3_5AffineWeights::Quantized {
            packed_weight,
            quantization_bits,
            ..
        } => packed_weight.shape()[1] * 32 / quantization_bits,
        Qwen3_5AffineWeights::NativeBfloat16 { weight } => weight.shape()[1],
        Qwen3_5AffineWeights::Jangtq2Codebook { .. } => {
            return Err(TextExecutionError::InvalidTensor {
                tensor_name: weight_name.to_owned(),
                description: "a rotated module of a folded pack cannot be stored as codebooks",
            });
        }
    };
    u32::try_from(stored_width).map_err(|_| TextExecutionError::InvalidTensor {
        tensor_name: weight_name.to_owned(),
        description: "the rotated module's stored input width is negative",
    })
}

/// Narrows one layer's float32 auxiliaries to the pack's float16: the
/// norms, the convolution, the decay parameters and the DeltaNet rate
/// projections, which the sign table does not rotate.
fn narrow_layer_auxiliaries(
    runtime: &MlxRuntime,
    layer_weights: &mut Qwen3_5DecoderLayerWeights,
) -> Result<(), TextExecutionError> {
    layer_weights.input_normalization_weight =
        narrowed(runtime, &layer_weights.input_normalization_weight)?;
    layer_weights.post_attention_normalization_weight =
        narrowed(runtime, &layer_weights.post_attention_normalization_weight)?;
    match &mut layer_weights.attention_weights {
        Qwen3_5AttentionWeights::Linear(linear) => {
            linear.convolution_weight = narrowed(runtime, &linear.convolution_weight)?;
            linear.decay_interval_bias = narrowed(runtime, &linear.decay_interval_bias)?;
            linear.decay_rate_logarithm = narrowed(runtime, &linear.decay_rate_logarithm)?;
            linear.normalization_weight = narrowed(runtime, &linear.normalization_weight)?;
            narrow_unquantized_projection(runtime, &mut linear.update_rate_projection)?;
            narrow_unquantized_projection(runtime, &mut linear.decay_interval_projection)?;
        }
        Qwen3_5AttentionWeights::Full(full) => {
            full.query_normalization_weight = narrowed(runtime, &full.query_normalization_weight)?;
            full.key_normalization_weight = narrowed(runtime, &full.key_normalization_weight)?;
        }
    }
    Ok(())
}

fn narrow_unquantized_projection(
    runtime: &MlxRuntime,
    projection: &mut Qwen3_5DenseProjection,
) -> Result<(), TextExecutionError> {
    if let Qwen3_5ProjectionStorage::Bound(Qwen3_5AffineWeights::NativeBfloat16 { weight }) =
        &mut projection.storage
    {
        *weight = narrowed(runtime, weight)?;
    }
    Ok(())
}

/// `tensor` in the pack's float16, unchanged when it already is.
fn narrowed(runtime: &MlxRuntime, tensor: &MlxArray) -> Result<MlxArray, TextExecutionError> {
    if tensor.dtype() == FOLDED_ACTIVATION_DTYPE {
        return Ok(tensor.retain()?);
    }
    Ok(runtime.astype(tensor, FOLDED_ACTIVATION_DTYPE)?)
}
