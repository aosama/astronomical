//! Extraction and installation of one verify window's fixed state leaves.
//!
//! The compiled window is a pure function: every mutable state it consumes
//! enters as an input array and leaves as an output array. This module owns
//! the mapping between those flat input/output vectors and the live
//! `RequestDecoderStateStack`, so the compiled lane never mutates the eager
//! path's state types and never needs new accessors on them.
//!
//! Leaf order is the compiled graph's frozen ABI: layer-major, and within a
//! layer the resident variant's natural fields (full attention: keys, values;
//! composite: convolution rolling buffer, recurrent state).
//!
//! Full-attention leaves carry the eager owner's storage. New keys and values
//! are installed through that owner's append path so its existing capacity
//! policy remains authoritative.

use astronomical_runtime_integration::MlxArray;

use crate::decoder_cache::{DecoderCacheState, FullAttentionKeyValueState};
use crate::qwen3_5::decoder::RequestDecoderStateStack;

/// One layer's read view for the compiled window's input vector.
#[derive(Debug)]
pub(crate) enum WindowStateLeaves {
    /// Append-only attention: keys tensor, values tensor, logical offset.
    FullAttention {
        keys: MlxArray,
        values: MlxArray,
        offset_tokens: i32,
    },
    /// Hybrid layer: convolution rolling buffer, float32 recurrent state.
    GatedDelta {
        convolution: MlxArray,
        recurrent: MlxArray,
    },
}

/// One layer's output view the compiled window returns.
#[derive(Debug)]
pub(crate) enum WindowStateUpdate {
    FullAttention {
        keys: MlxArray,
        values: MlxArray,
    },
    GatedDelta {
        convolution: MlxArray,
        recurrent: MlxArray,
        /// Rolling-window snapshots for verifier-prefix rows one through
        /// rows minus one, in ascending row order.
        boundary_convolution_states: Vec<MlxArray>,
        /// Recurrent snapshots for the same rows, in ascending row order.
        boundary_recurrent_states: Vec<MlxArray>,
    },
}

/// Extracts the flat compile-input leaves for every layer, in layer order.
///
/// Returns an error when any layer has not yet allocated its storage: the
/// compiled lane runs only after prefill has seated every state.
pub(crate) fn extract_window_input_leaves(
    state_stack: &RequestDecoderStateStack,
) -> Result<Vec<WindowStateLeaves>, String> {
    let mut window_state_leaves = Vec::with_capacity(state_stack.layer_count());
    for layer_index in 0..state_stack.layer_count() {
        let layer_state = state_stack.layer(layer_index).ok_or_else(|| {
            format!("decoder state layer {layer_index} disappeared while extracting window leaves")
        })?;
        match layer_state {
            DecoderCacheState::AppendOnlyAttention { attention } => {
                let (keys, values, offset_tokens) = full_attention_leaves(attention)?;
                window_state_leaves.push(WindowStateLeaves::FullAttention {
                    keys,
                    values,
                    offset_tokens,
                });
            }
            DecoderCacheState::Composite {
                convolution,
                recurrent,
            } => {
                let convolution_state = convolution.state().ok_or_else(|| {
                    format!("convolution state for layer {layer_index} is not allocated")
                })?;
                let recurrent_state = recurrent.state().ok_or_else(|| {
                    format!("recurrent state for layer {layer_index} is not allocated")
                })?;
                window_state_leaves.push(WindowStateLeaves::GatedDelta {
                    convolution: convolution_state
                        .retain()
                        .map_err(|error| error.to_string())?,
                    recurrent: recurrent_state
                        .retain()
                        .map_err(|error| error.to_string())?,
                });
            }
        }
    }
    Ok(window_state_leaves)
}

/// Installs one hybrid layer's compiled-window outputs back into the live
/// state. The compiled graph returns states with the same shapes it consumed;
/// the install trusts the compiled contract and lets admission checks reject
/// any drift loudly on the next eager forward.
pub(crate) fn install_window_output_leaves(
    state_stack: &mut RequestDecoderStateStack,
    layer_updates: &[WindowStateUpdate],
) -> Result<(), String> {
    if state_stack.layer_count() != layer_updates.len() {
        return Err(format!(
            "compiled window returned {} layer updates for {} layers",
            layer_updates.len(),
            state_stack.layer_count()
        ));
    }
    for (layer_index, layer_update) in layer_updates.iter().enumerate() {
        let layer_state = state_stack.layer_mut(layer_index).ok_or_else(|| {
            format!("decoder state layer {layer_index} disappeared while installing window leaves")
        })?;
        match (layer_state, layer_update) {
            (
                DecoderCacheState::Composite {
                    convolution,
                    recurrent,
                },
                WindowStateUpdate::GatedDelta {
                    convolution: next_convolution,
                    recurrent: next_recurrent,
                    ..
                },
            ) => {
                convolution
                    .replace_state(
                        next_convolution
                            .retain()
                            .map_err(|error| error.to_string())?,
                    )
                    .map_err(|error| format!("convolution install failed: {error}"))?;
                recurrent.set_next(next_recurrent.retain().map_err(|error| error.to_string())?);
            }
            // Full-attention installs go through the append-only owner's own
            // update path after this pass; they are untouched here.
            (DecoderCacheState::AppendOnlyAttention { .. }, _) => {}
            (DecoderCacheState::Composite { .. }, _) => {
                return Err(format!(
                    "layer {layer_index} received no compiled gated-delta update"
                ));
            }
        }
    }
    Ok(())
}

fn full_attention_leaves(
    attention: &FullAttentionKeyValueState,
) -> Result<(MlxArray, MlxArray, i32), String> {
    let keys = attention
        .keys_state()
        .ok_or_else(|| "full-attention keys are not allocated".to_owned())?
        .retain()
        .map_err(|error| error.to_string())?;
    let values = attention
        .values_state()
        .ok_or_else(|| "full-attention values are not allocated".to_owned())?
        .retain()
        .map_err(|error| error.to_string())?;
    Ok((keys, values, attention.offset_tokens()))
}
