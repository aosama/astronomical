//! What a decision model adds to the shared Qwen3.5 forward pass.
//!
//! Kev and Clef read a Qwen3.5 backbone through the same layers chat uses.
//! A whole-model training gives that backbone add-on weights on every dense
//! projection, and while it builds its gradient graph it also needs a Gated
//! DeltaNet recurrence MLX can differentiate and attention MLX can
//! differentiate (`empty_state_forward`). Both live in one slot on the model:
//! empty for chat, the folder's add-ons for a trained decision model, the
//! weights being moved while a training builds its graph.
//!
//! Each dense projection knows its shipped module name, so it finds its own
//! add-on (`Qwen3_5DenseProjection::project_rows`) and no forward function
//! carries the add-ons as an argument. A chat model pays one borrow of an
//! empty slot per projection.
//!
//! The slot is single-threaded (`RefCell`, `Rc`) because the model already
//! is: its MLX arrays are tied to the thread that made them.

use std::rc::Rc;

use mlx_bridge::MlxArray;

use super::Qwen3_5ModelBase;
use super::differentiable_attention::Qwen3_5AttentionKernel;
use crate::decision_models::decision_adapters::AdapterTarget;
use crate::linear_adapters::LinearAdapters;
use crate::qwen3_5_core::model_math::decoder_layer_weights::{
    Qwen3_5AffineWeights, Qwen3_5AttentionWeights, Qwen3_5DecoderFeedForwardWeights,
};
use crate::qwen3_5_core::model_math::dense_projection::{
    Qwen3_5DenseProjection, Qwen3_5ProjectionStorage,
};
use crate::qwen3_5_core::model_math::differentiable_gated_delta::DifferentiableGatedDelta;
use crate::qwen3_5_core::model_math::error::Qwen3_5ExecutionError;

/// What a decision model adds to the forward pass.
pub(crate) struct Qwen3_5LayerAdditions {
    /// Add-ons by the shipped module name of their projection.
    pub(crate) adapters: LinearAdapters,
    /// Set only while a training builds its graph: serving's recurrence
    /// with a backward pass attached.
    pub(crate) differentiable_recurrence: Option<Rc<DifferentiableGatedDelta>>,
}

// The model base derives `Debug`; the add-ons are GPU arrays and closures,
// so only what is in force is shown.
impl std::fmt::Debug for Qwen3_5LayerAdditions {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Qwen3_5LayerAdditions")
            .field("has_adapters", &!self.adapters.is_empty())
            .field(
                "has_differentiable_recurrence",
                &self.differentiable_recurrence.is_some(),
            )
            .finish()
    }
}

impl Qwen3_5ModelBase {
    /// Puts `layer_additions` in force for every forward pass built after
    /// this, or clears them.
    pub(crate) fn set_layer_additions(&self, layer_additions: Option<Rc<Qwen3_5LayerAdditions>>) {
        *self.layer_additions.borrow_mut() = layer_additions;
    }

    /// `base_output` plus the add-on filed under `module_name`, if any.
    pub(crate) fn with_layer_add_on(
        &self,
        module_name: &str,
        activations: &MlxArray,
        base_output: MlxArray,
    ) -> Result<MlxArray, Qwen3_5ExecutionError> {
        let layer_additions = self.layer_additions.borrow();
        let Some(layer_additions) = layer_additions.as_ref() else {
            return Ok(base_output);
        };
        Ok(layer_additions.adapters.adapted_output(
            &self.runtime,
            module_name,
            activations,
            base_output,
        )?)
    }

    /// The attention kernels the forward pass uses: plain operations while a
    /// training builds its graph (the differentiable recurrence is in force
    /// exactly then), MLX's fused attention otherwise.
    pub(crate) fn attention_kernel(&self) -> Qwen3_5AttentionKernel {
        match self.layer_additions.borrow().as_ref() {
            Some(layer_additions) if layer_additions.differentiable_recurrence.is_some() => {
                Qwen3_5AttentionKernel::PlainOperations
            }
            _ => Qwen3_5AttentionKernel::Fused,
        }
    }

    /// The additions in force, if any.
    pub(crate) fn layer_additions_in_force(&self) -> Option<Rc<Qwen3_5LayerAdditions>> {
        self.layer_additions.borrow().clone()
    }

    /// Every dense projection of the decoder layers that can take an add-on,
    /// layer by layer in the order the forward pass runs them, with widths
    /// read from the loaded weights. The output head and the embeddings
    /// stay as shipped.
    ///
    /// # Errors
    /// See [`Self::adapter_targets_by_layer`].
    pub(crate) fn adapter_targets(&self) -> Result<Vec<AdapterTarget>, Qwen3_5ExecutionError> {
        Ok(self
            .adapter_targets_by_layer()?
            .into_iter()
            .flatten()
            .collect())
    }

    /// The add-on targets of each decoder layer, one list per layer in
    /// forward order: what a layer recomputed during the backward pass
    /// takes as its own inputs.
    ///
    /// # Errors
    /// Refuses a model whose feed-forward layers are mixtures of experts, or
    /// whose projections are stored in a form an add-on cannot be sized
    /// from: whole-model training covers dense Qwen3.5 backbones.
    pub(crate) fn adapter_targets_by_layer(
        &self,
    ) -> Result<Vec<Vec<AdapterTarget>>, Qwen3_5ExecutionError> {
        let mut targets_by_layer = Vec::with_capacity(self.weights.decoder_layer_weights.len());
        for decoder_layer_weights in &self.weights.decoder_layer_weights {
            let mut layer_projections: Vec<&Qwen3_5DenseProjection> = Vec::with_capacity(8);
            match &decoder_layer_weights.attention_weights {
                Qwen3_5AttentionWeights::Linear(linear_attention_weights) => layer_projections
                    .extend([
                        &linear_attention_weights.input_queries_keys_values_projection,
                        &linear_attention_weights.output_gate_projection,
                        &linear_attention_weights.update_rate_projection,
                        &linear_attention_weights.decay_interval_projection,
                        &linear_attention_weights.output_projection,
                    ]),
                Qwen3_5AttentionWeights::Full(full_attention_weights) => {
                    layer_projections.extend([
                        &full_attention_weights.query_projection,
                        &full_attention_weights.key_projection,
                        &full_attention_weights.value_projection,
                        &full_attention_weights.output_projection,
                    ])
                }
            }
            let Qwen3_5DecoderFeedForwardWeights::Dense(dense_mlp_weights) =
                &decoder_layer_weights.mlp_weights
            else {
                return Err(unsized_add_on(
                    "a mixture-of-experts model cannot train its whole model",
                ));
            };
            layer_projections.extend([
                &dense_mlp_weights.gate_projection,
                &dense_mlp_weights.up_projection,
                &dense_mlp_weights.down_projection,
            ]);
            let mut layer_targets = Vec::with_capacity(layer_projections.len());
            for projection in layer_projections {
                layer_targets.push(adapter_target_of(projection)?);
            }
            targets_by_layer.push(layer_targets);
        }
        Ok(targets_by_layer)
    }
}

/// A projection's add-on target: its name, and its widths from the stored
/// weight (`[outputs, inputs]` for bf16; MLX's affine packing stores
/// `32 / bits` inputs per 32-bit word).
fn adapter_target_of(
    projection: &Qwen3_5DenseProjection,
) -> Result<AdapterTarget, Qwen3_5ExecutionError> {
    let (output_width, input_width) = match &projection.storage {
        Qwen3_5ProjectionStorage::Bound(Qwen3_5AffineWeights::NativeBfloat16 { weight }) => {
            let weight_shape = weight.shape();
            (weight_shape[0], weight_shape[1])
        }
        Qwen3_5ProjectionStorage::Bound(Qwen3_5AffineWeights::Quantized {
            packed_weight,
            quantization_bits,
            ..
        }) => {
            let packed_shape = packed_weight.shape();
            (packed_shape[0], packed_shape[1] * 32 / quantization_bits)
        }
        _ => {
            return Err(unsized_add_on(
                "a projection is stored in a form whole-model training does not cover",
            ));
        }
    };
    Ok(AdapterTarget {
        name: projection.module_name.clone(),
        input_width: input_width as usize,
        output_width: output_width as usize,
    })
}

fn unsized_add_on(description: &'static str) -> Qwen3_5ExecutionError {
    Qwen3_5ExecutionError::InvalidInput { description }
}
