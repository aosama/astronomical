//! Fuses the target model's hidden states into draft conditioning states.
//!
//! The draft does not build its own representation of the context. It reads
//! the target's mind: the hidden states captured at the configured
//! `target_layer_ids` taps are concatenated (one target hidden width per tap)
//! and projected by the draft's learned `fc` matrix into the draft's trunk
//! width, then normalized by `hidden_norm`. The result feeds every draft
//! layer's key/value projections, which is what keeps a six-layer trunk
//! synchronized with a forty-layer target.

use astronomical_runtime_integration::{MlxRuntime, MlxRuntimeError};
use mlx_c_rust::MlxArray;

use super::draft_weights::DFlashDraftWeights;

/// Projects concatenated target hidden-state taps into conditioned states.
///
/// Each tap must have shape `[1, token_count, target_hidden_size]`; they are
/// concatenated in tap order (the order the distillation trained against),
/// projected, and normalized. The returned array has shape
/// `[1, token_count, draft_hidden_size]`.
pub fn condition_target_hidden_states(
    runtime: &MlxRuntime,
    weights: &DFlashDraftWeights,
    target_hidden_state_taps: &[MlxArray],
    rms_norm_epsilon: f32,
) -> Result<MlxArray, MlxRuntimeError> {
    if target_hidden_state_taps.is_empty() {
        return Err(MlxRuntimeError::RuntimeOperation {
            operation: "condition target hidden states for the DFlash draft",
            description: "at least one target hidden-state tap is required".to_owned(),
        });
    }
    let tap_references: Vec<&MlxArray> = target_hidden_state_taps.iter().collect();
    let concatenated_taps = runtime.concatenate_axis(&tap_references, -1)?;
    // The fc weight ships as [draft hidden size, tap count × target hidden
    // size]; the dense-linear house form transposes it into matmul order.
    let transposed_projection =
        runtime.transpose_axes(&weights.target_conditioning_projection, &[1, 0])?;
    let projected_taps = runtime.matmul(&concatenated_taps, &transposed_projection)?;
    Ok(runtime.rms_norm(
        &projected_taps,
        &weights.target_conditioning_normalization,
        rms_norm_epsilon,
    )?)
}
