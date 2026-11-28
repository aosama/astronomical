//! Qwen3.5's Gated DeltaNet recurrence in a form MLX can differentiate, for
//! training add-ons inside Kev and Clef.
//!
//! Serving runs the recurrence through Astronomical's fused Metal kernel
//! when the worker's capability probe kept it, and through one-token public
//! MLX operations when the probe demoted it, as on MLX's CUDA backend, which
//! cannot run a Metal kernel. Neither route has a backward pass of its own
//! that training can afford. Training needs two things from the recurrence:
//! scores that equal serving's exactly, so the parity check before training
//! and the proof of the written folder can compare them, and a gradient that
//! is merely accurate. So the recurrence here runs serving's own route
//! forward, the one the loaded model chose, and borrows its backward pass
//! from MLX's `gated_delta_update`, attached as a custom vector-Jacobian
//! product.
//!
//! The two other routes were measured on Metal, on one Clef-shaped layer
//! (16 key heads, 32 value heads of 128), and not taken:
//! - MLX's `gated_delta_update` on its own: a backward pass of 0.10 GiB and
//!   0.07 s over 512 tokens, but its forward differs from serving's kernel
//!   by about 8e-4 relative per layer, so a trainable copy built on it could
//!   never score as the served model does;
//! - serving's one-token plain-op route: identical to the kernel (1.5e-8)
//!   and differentiable, but its backward pass keeps every token's state,
//!   2.48 GiB and 0.34 s per layer over 512 tokens, about 60 GiB across
//!   Clef's 24 such layers.
//!
//! That 0.10 GiB backward pass is Metal's fused kernel. MLX 0.32.5's CUDA
//! backend has a forward kernel for `gated_delta_update` but no backward one
//! (`GatedDeltaUpdateVJP::use_fallback` is always true there), so on CUDA
//! the borrowed backward pass differentiates MLX's one-token loop and keeps
//! every token's state, as the plain-op route above does.
//!
//! The borrowed backward pass is MLX's derivative of a recurrence that
//! differs from the forward by those 8e-4, so the gradient is off by about
//! as much; that changes the size of a training step in the fourth decimal,
//! which no training notices.
//!
//! The inputs are serving's: `decays` are the per-token multiplicative
//! forgetting factors already exponentiated, `update_rates` are the delta
//! rule's write strengths, and the state is laid out
//! `(batch, value heads, value dims, key dims)` as serving keeps it.

use mlx_bridge::{
    MlxArray, MlxArrayVector, MlxCError, MlxClosure, MlxClosureCustom, vjp_of_closure,
};

use super::gated_delta_inputs::GatedDeltaInputs;
use crate::{qwen3_5_gated_delta_kernel, qwen3_5_gated_delta_sequence};

/// The recurrence's six inputs, in the order both routes take them.
const RECURRENCE_INPUT_COUNT: usize = 6;

/// Serving's recurrence with MLX's backward pass attached.
pub struct DifferentiableGatedDelta {
    recurrence: MlxClosure,
}

impl DifferentiableGatedDelta {
    /// Builds the recurrence once; every layer and pass reuses it.
    /// `serving_uses_fused_kernel` is whether the loaded model kept the fused
    /// kernel (its `gated_delta_kernel` is present): the forward then runs
    /// that kernel, and otherwise the one-token operations serving falls
    /// back to, so trained and served scores come from the same arithmetic.
    ///
    /// # Errors
    /// Returns the MLX description when the kernel or a closure cannot be made.
    pub fn new(serving_uses_fused_kernel: bool) -> Result<Self, MlxCError> {
        let served_forward = served_forward_closure(serving_uses_fused_kernel)?;
        let borrowed_backward = MlxClosureCustom::from_fn(borrowed_backward_pass)?;
        Ok(Self {
            recurrence: served_forward.with_custom_vjp(&borrowed_backward)?,
        })
    }

    /// The recurrence's outputs, then the state after the last token.
    ///
    /// # Errors
    /// Returns the MLX description when the graph cannot be built.
    pub fn sequence(
        &self,
        inputs: &GatedDeltaInputs<'_>,
    ) -> Result<(MlxArray, MlxArray), MlxCError> {
        let recurrence_outputs = self.recurrence.apply(&MlxArrayVector::new(&[
            inputs.queries,
            inputs.keys,
            inputs.values,
            inputs.decays,
            inputs.update_rates,
            inputs.recurrent_state,
        ])?)?;
        Ok((
            recurrence_outputs.array_at(0, "read the gated-delta outputs")?,
            recurrence_outputs.array_at(1, "read the gated-delta state")?,
        ))
    }
}

/// Serving's route as an MLX closure: the fused kernel when serving kept
/// it, the one-token operations otherwise. MLX's transforms take a
/// function, not a graph, so this is one of the closures MLX's calling
/// convention requires; it owns its kernel and bindings context.
fn served_forward_closure(serving_uses_fused_kernel: bool) -> Result<MlxClosure, MlxCError> {
    let forward_context = runtime_integration::platform_bindings_context()?;
    let served_kernel = serving_uses_fused_kernel
        .then(qwen3_5_gated_delta_kernel)
        .transpose()
        .map_err(recurrence_failure)?;
    MlxClosure::from_fn(move |recurrence_inputs: &[MlxArray]| {
        let [queries, keys, values, decays, update_rates, recurrent_state] = recurrence_inputs
        else {
            return Err(input_count_failure(recurrence_inputs.len()));
        };
        let (sequence_outputs, next_recurrent_state) = qwen3_5_gated_delta_sequence(
            &forward_context,
            served_kernel.as_ref(),
            &GatedDeltaInputs {
                queries,
                keys,
                values,
                decays,
                update_rates,
                recurrent_state,
            },
        )
        .map_err(recurrence_failure)?;
        Ok(vec![sequence_outputs, next_recurrent_state])
    })
}

/// The cotangents of the six inputs, from MLX's own derivative of
/// `gated_delta_update` at the same inputs.
///
/// MLX calls a custom backward pass with the inputs, then the outputs'
/// cotangents, then the outputs (MLX's `custom_vjp` order); the bridge's
/// trampoline forwards them in that order whatever its parameter names say.
fn borrowed_backward_pass(
    recurrence_inputs: &[MlxArray],
    output_cotangents: &[MlxArray],
    _served_outputs: &[MlxArray],
) -> Result<Vec<MlxArray>, MlxCError> {
    if recurrence_inputs.len() != RECURRENCE_INPUT_COUNT {
        return Err(input_count_failure(recurrence_inputs.len()));
    }
    let mlx_recurrence = MlxClosure::from_fn(mlx_gated_delta_update)?;
    let mut borrowed_inputs = Vec::with_capacity(RECURRENCE_INPUT_COUNT);
    for recurrence_input in recurrence_inputs {
        borrowed_inputs.push(recurrence_input);
    }
    let mut borrowed_cotangents = Vec::with_capacity(output_cotangents.len());
    for output_cotangent in output_cotangents {
        borrowed_cotangents.push(output_cotangent);
    }
    let (_mlx_outputs, input_cotangents) = vjp_of_closure(
        &mlx_recurrence,
        &MlxArrayVector::new(&borrowed_inputs)?,
        &MlxArrayVector::new(&borrowed_cotangents)?,
    )?;
    let mut cotangents = Vec::with_capacity(RECURRENCE_INPUT_COUNT);
    for input_position in 0..input_cotangents.len() {
        cotangents.push(input_cotangents.array_at(input_position, "read an input cotangent")?);
    }
    Ok(cotangents)
}

/// MLX's `gated_delta_update` over the six inputs, as a closure body.
fn mlx_gated_delta_update(recurrence_inputs: &[MlxArray]) -> Result<Vec<MlxArray>, MlxCError> {
    let [queries, keys, values, decays, update_rates, recurrent_state] = recurrence_inputs else {
        return Err(input_count_failure(recurrence_inputs.len()));
    };
    let mlx_context = runtime_integration::platform_bindings_context()?;
    let recurrence_outputs = mlx_context.gated_delta_update(
        queries,
        keys,
        values,
        decays,
        update_rates,
        Some(recurrent_state),
        None,
    )?;
    Ok(vec![
        recurrence_outputs.array_at(0, "read the gated-delta outputs")?,
        recurrence_outputs.array_at(1, "read the gated-delta state")?,
    ])
}

/// MLX's transforms carry `MlxCError`, so the served recurrence's runtime
/// error is restated as one.
fn recurrence_failure(recurrence_error: runtime_integration::MlxRuntimeError) -> MlxCError {
    MlxCError {
        operation: "apply the gated-delta recurrence",
        description: recurrence_error.to_string(),
    }
}

fn input_count_failure(input_count: usize) -> MlxCError {
    MlxCError {
        operation: "apply the gated-delta recurrence",
        description: format!("expected {RECURRENCE_INPUT_COUNT} inputs, got {input_count}"),
    }
}
