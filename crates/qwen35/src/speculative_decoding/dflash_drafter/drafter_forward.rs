//! The drafter's pass over one block: the anchor token and the mask slots go
//! through every layer beside the fused target context, and the rows after the
//! anchor come out normalized, ready for the target's head.

use mlx_bridge::MlxArray;
use runtime_integration::{MlxRuntime, MlxRuntimeError};

use super::drafter_attention::{BlockAttentionRows, attend_block_to_context};
use super::drafter_configuration::DFlashDrafterConfiguration;
use super::drafter_context_state::{DrafterContextStates, DrafterLayerContextState};
use super::drafter_weights::{DFlashDrafterWeights, DrafterLayerWeights, LayerConvolutionWeights};
use super::grouped_dynamic_convolution::{convolve_sublayer_input, convolve_sublayer_output};

/// What the drafter borrows from its target: it has no vocabulary of its own.
pub trait DFlashDraftTarget {
    /// The target's input embedding rows for `token_ids` [1, L]: [1, L, H].
    fn embedded_rows(
        &self,
        runtime: &MlxRuntime,
        token_ids: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError>;

    /// The target's output-head logits for `hidden_rows` [1, L, H]: [1, L, vocabulary].
    fn head_logits(
        &self,
        runtime: &MlxRuntime,
        hidden_rows: &MlxArray,
    ) -> Result<MlxArray, MlxRuntimeError>;
}

/// The drafter's final hidden rows for the proposal slots, [1, block - 1, H].
///
/// `block_token_ids` [1, block] is the anchor (the last token the target
/// committed) followed by mask tokens. `target_tap_rows` [1, S, taps x H] are
/// the target's tapped hidden rows for tokens committed since the last round;
/// they extend every layer's kept context.
pub fn drafted_hidden_rows(
    runtime: &MlxRuntime,
    weights: &DFlashDrafterWeights,
    configuration: &DFlashDrafterConfiguration,
    target: &dyn DFlashDraftTarget,
    context_states: &mut DrafterContextStates,
    block_token_ids: &MlxArray,
    target_tap_rows: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    let fused_context_rows = runtime.rms_norm(
        &weights.tap_fusion.forward(runtime, target_tap_rows)?,
        &weights.tap_fusion_normalization,
        configuration.rms_norm_epsilon,
    )?;
    let mut block_rows = target.embedded_rows(runtime, block_token_ids)?;
    let layer_states = weights.layers.iter().zip(context_states.layers.iter_mut());
    for (layer_index, (layer, context_state)) in layer_states.enumerate() {
        let attention_inputs = LayerAttentionInputs {
            configuration,
            context_state,
            fused_context_rows: &fused_context_rows,
            block_is_causal: configuration.layer_block_is_causal(layer_index),
        };
        block_rows = drafter_layer(runtime, layer, attention_inputs, &block_rows)?;
    }
    let block_size = block_rows.shape()[1];
    let proposal_rows = runtime.slice(
        &block_rows,
        &[0, 1, 0],
        &[1, block_size, configuration.hidden_size],
        &[1, 1, 1],
    )?;
    Ok(runtime.rms_norm(
        &proposal_rows,
        &weights.final_normalization,
        configuration.rms_norm_epsilon,
    )?)
}

/// One layer's attention sublayer: the normalized, possibly convolved block
/// rows and what they attend to.
struct LayerAttentionInputs<'layer> {
    configuration: &'layer DFlashDrafterConfiguration,
    context_state: &'layer mut DrafterLayerContextState,
    fused_context_rows: &'layer MlxArray,
    block_is_causal: bool,
}

/// One drafter layer: the plain DFlash layer, or DFlash2's with a
/// convolution around each sublayer.
fn drafter_layer(
    runtime: &MlxRuntime,
    layer: &DrafterLayerWeights,
    attention_inputs: LayerAttentionInputs<'_>,
    layer_input: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    match &layer.convolutions {
        Some(convolutions) => {
            convolved_drafter_layer(runtime, layer, convolutions, attention_inputs, layer_input)
        }
        None => plain_drafter_layer(runtime, layer, attention_inputs, layer_input),
    }
}

/// DFlash: attention then MLP, each around a residual.
fn plain_drafter_layer(
    runtime: &MlxRuntime,
    layer: &DrafterLayerWeights,
    attention_inputs: LayerAttentionInputs<'_>,
    layer_input: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    let epsilon = attention_inputs.configuration.rms_norm_epsilon;
    let normalized_input = runtime.rms_norm(layer_input, &layer.input_normalization, epsilon)?;
    let attended = attend(runtime, layer, attention_inputs, &normalized_input)?;
    let after_attention = runtime.add(layer_input, &attended)?;
    let mlp_input = runtime.rms_norm(
        &after_attention,
        &layer.post_attention_normalization,
        epsilon,
    )?;
    Ok(runtime.add(&after_attention, &swiglu(runtime, layer, &mlp_input)?)?)
}

/// DFlash2: convolution, attention, convolution around the first residual;
/// convolution, MLP, convolution around the second.
fn convolved_drafter_layer(
    runtime: &MlxRuntime,
    layer: &DrafterLayerWeights,
    convolutions: &LayerConvolutionWeights,
    attention_inputs: LayerAttentionInputs<'_>,
    layer_input: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    let epsilon = attention_inputs.configuration.rms_norm_epsilon;
    let attention_input = convolve_sublayer_input(
        runtime,
        &convolutions.attention,
        &runtime.rms_norm(layer_input, &layer.input_normalization, epsilon)?,
    )?;
    let attended = attend(
        runtime,
        layer,
        attention_inputs,
        &attention_input.convolved_rows,
    )?;
    let after_attention = runtime.add(
        layer_input,
        &convolve_sublayer_output(
            runtime,
            &convolutions.attention,
            &attended,
            &attention_input.finishing_kernel,
        )?,
    )?;
    let mlp_input = convolve_sublayer_input(
        runtime,
        &convolutions.mlp,
        &runtime.rms_norm(
            &after_attention,
            &layer.post_attention_normalization,
            epsilon,
        )?,
    )?;
    let mlp_output = swiglu(runtime, layer, &mlp_input.convolved_rows)?;
    Ok(runtime.add(
        &after_attention,
        &convolve_sublayer_output(
            runtime,
            &convolutions.mlp,
            &mlp_output,
            &mlp_input.finishing_kernel,
        )?,
    )?)
}

fn attend(
    runtime: &MlxRuntime,
    layer: &DrafterLayerWeights,
    attention_inputs: LayerAttentionInputs<'_>,
    block_rows: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    attend_block_to_context(
        runtime,
        layer,
        attention_inputs.configuration,
        attention_inputs.context_state,
        BlockAttentionRows {
            block_rows,
            fused_context_rows: attention_inputs.fused_context_rows,
            block_is_causal: attention_inputs.block_is_causal,
        },
    )
}

fn swiglu(
    runtime: &MlxRuntime,
    layer: &DrafterLayerWeights,
    rows: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    let gate = runtime.silu(&layer.gate_projection.forward(runtime, rows)?)?;
    let gated = runtime.multiply(&gate, &layer.up_projection.forward(runtime, rows)?)?;
    layer.down_projection.forward(runtime, &gated)
}
