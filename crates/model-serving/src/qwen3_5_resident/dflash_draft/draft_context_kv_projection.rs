//! Projects conditioned target states into the draft's context key/value.
//!
//! For every draft layer, the conditioned states (`fc`-projected target
//! hidden states) pass through that layer's own key and value projections,
//! per-head key normalization, and rotary position embedding at the tokens'
//! absolute positions, then land in the layer's key-value state. This is how
//! the draft's cache is built from the target's belief state rather than
//! from the draft's own tokens — the core of the conditioning contract.

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};
use mlx_c_rust::MlxArray;

use super::draft_configuration::DFlashDraftConfiguration;
use super::draft_key_value_state::DFlashDraftKeyValueStates;
use super::draft_weights::{DFlashDraftLayerWeights, DFlashDraftWeights};

/// Appends one span of context key/value, derived from conditioned states.
///
/// `conditioned_states` must have shape `[1, token_count, hidden_size]` and
/// represent tokens at absolute positions
/// `first_absolute_position .. first_absolute_position + token_count`.
/// Every layer's key-value state advances by `token_count`.
pub fn append_conditioned_context_key_values(
    runtime: &MlxRuntime,
    weights: &DFlashDraftWeights,
    configuration: &DFlashDraftConfiguration,
    key_value_states: &mut DFlashDraftKeyValueStates,
    conditioned_states: &MlxArray,
    first_absolute_position: i32,
) -> Result<(), MlxRuntimeError> {
    for layer_index in 0..configuration.layer_count() as usize {
        let layer_weights = &weights.layers[layer_index];
        let layer_state = key_value_states.layer_state_mut(layer_index).ok_or(
            MlxRuntimeError::RuntimeOperation {
                operation: "append DFlash draft context key values",
                description: format!("draft key-value layer {layer_index} is missing"),
            },
        )?;
        let new_keys = projected_rotated_keys(
            runtime,
            layer_weights,
            configuration,
            conditioned_states,
            first_absolute_position,
        )?;
        let new_values =
            projected_values(runtime, layer_weights, configuration, conditioned_states)?;
        layer_state.update_and_fetch(runtime, &new_keys, &new_values, first_absolute_position)?;
    }
    Ok(())
}

/// Projects, normalizes, rotates, and lays out one layer's keys.
///
/// Shared by the context projection and the trunk forward so both paths put
/// keys through the identical pipeline (projection → per-head RMS norm →
/// rotary embedding → `[1, kv heads, tokens, head dim]` layout).
pub fn projected_rotated_keys(
    runtime: &MlxRuntime,
    layer_weights: &DFlashDraftLayerWeights,
    configuration: &DFlashDraftConfiguration,
    hidden_states: &MlxArray,
    first_absolute_position: i32,
) -> Result<MlxArray, MlxRuntimeError> {
    let key_value_head_count = configuration.key_value_head_count();
    let head_dimension = configuration.head_dimension();
    let transposed_projection = runtime.transpose_axes(&layer_weights.key_projection, &[1, 0])?;
    let keys = runtime.matmul(hidden_states, &transposed_projection)?;
    let key_token_count = keys.shape()[1];
    let keys = runtime.reshape(
        &keys,
        &[1, key_token_count, key_value_head_count, head_dimension],
    )?;
    let keys = runtime.rms_norm(
        &keys,
        &layer_weights.key_head_normalization,
        configuration.rms_norm_epsilon(),
    )?;
    let keys = runtime.transpose_axes(&keys, &[0, 2, 1, 3])?;
    Ok(runtime.rope(
        &keys,
        head_dimension,
        configuration.rope_theta(),
        first_absolute_position,
    )?)
}

/// Projects and lays out one layer's values (no norm, no rotation).
pub fn projected_values(
    runtime: &MlxRuntime,
    layer_weights: &DFlashDraftLayerWeights,
    configuration: &DFlashDraftConfiguration,
    hidden_states: &MlxArray,
) -> Result<MlxArray, MlxRuntimeError> {
    let key_value_head_count = configuration.key_value_head_count();
    let head_dimension = configuration.head_dimension();
    let transposed_projection = runtime.transpose_axes(&layer_weights.value_projection, &[1, 0])?;
    let values = runtime.matmul(hidden_states, &transposed_projection)?;
    let value_token_count = values.shape()[1];
    let values = runtime.reshape(
        &values,
        &[1, value_token_count, key_value_head_count, head_dimension],
    )?;
    Ok(runtime.transpose_axes(&values, &[0, 2, 1, 3])?)
}
