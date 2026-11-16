//! The drafter's pass over one block: the anchor token and the mask slots go
//! through every layer beside the fused target context, and the rows after the
//! anchor come out normalized, ready for the target's head.

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};
use mlx_bridge::MlxArray;

use super::drafter_attention::attend_block_to_context;
use super::drafter_configuration::DFlash2DrafterConfiguration;
use super::drafter_context_state::{DrafterContextStates, DrafterLayerContextState};
use super::drafter_weights::{DFlash2DrafterWeights, DrafterLayerWeights};
use super::grouped_dynamic_convolution::{convolve_sublayer_input, convolve_sublayer_output};

/// What the drafter borrows from its target: it has no vocabulary of its own.
pub trait DFlash2DraftTarget {
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
    weights: &DFlash2DrafterWeights,
    configuration: &DFlash2DrafterConfiguration,
    target: &dyn DFlash2DraftTarget,
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
    for (layer, context_state) in weights.layers.iter().zip(context_states.layers.iter_mut()) {
        block_rows = drafter_layer(
            runtime,
            layer,
            configuration,
            context_state,
            &block_rows,
            &fused_context_rows,
        )?;
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

/// One DFlash2 layer: convolution, attention, convolution around the first
/// residual; convolution, MLP, convolution around the second.
fn drafter_layer(
    runtime: &MlxRuntime,
    layer: &DrafterLayerWeights,
    configuration: &DFlash2DrafterConfiguration,
    context_state: &mut DrafterLayerContextState,
    layer_input: &MlxArray,
    fused_context_rows: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    let attention_input = convolve_sublayer_input(
        runtime,
        &layer.attention_convolution,
        configuration,
        &runtime.rms_norm(
            layer_input,
            &layer.input_normalization,
            configuration.rms_norm_epsilon,
        )?,
    )?;
    let attended = attend_block_to_context(
        runtime,
        layer,
        configuration,
        context_state,
        &attention_input.convolved_rows,
        fused_context_rows,
    )?;
    let after_attention = runtime.add(
        layer_input,
        &convolve_sublayer_output(
            runtime,
            &layer.attention_convolution,
            configuration,
            &attended,
            &attention_input.finishing_kernel,
        )?,
    )?;
    let mlp_input = convolve_sublayer_input(
        runtime,
        &layer.mlp_convolution,
        configuration,
        &runtime.rms_norm(
            &after_attention,
            &layer.post_attention_normalization,
            configuration.rms_norm_epsilon,
        )?,
    )?;
    let mlp_output = swiglu(runtime, layer, &mlp_input.convolved_rows)?;
    Ok(runtime.add(
        &after_attention,
        &convolve_sublayer_output(
            runtime,
            &layer.mlp_convolution,
            configuration,
            &mlp_output,
            &mlp_input.finishing_kernel,
        )?,
    )?)
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
