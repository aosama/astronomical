//! One loss and its gradient with respect to the trainable arrays, worked out
//! by MLX itself.
//!
//! Final-layer training wrote each family's backward pass by hand because a head is
//! a few small layers. Training add-on weights inside every layer of a
//! backbone means differentiating through attention, rotary embeddings,
//! normalization and 8-bit matrix products, so this asks MLX for the
//! gradient instead: the loss is built inside an MLX closure, and MLX's
//! value-and-gradient transform traces it backwards.
//!
//! MLX keeps a closure beyond the call that made it, so it may borrow nothing
//! (`'static`). A [`TrainableLoss`] therefore owns everything its graph
//! reads, as retained array handles that share their buffers with the served
//! model, and builds its graph through a bindings context of its own; the
//! context only names the default GPU stream, so both submit to the same
//! queue.

use mlx_bridge::{
    MlxArray, MlxArrayVector, MlxBindingsContext, MlxCError, MlxClosure, value_and_grad_of_closure,
};

/// A scalar loss built from the trainable arrays, owning everything else it reads.
pub(crate) trait TrainableLoss: 'static {
    /// Builds the lazy graph of the loss, a scalar, from `trainable_arrays`
    /// in the order they were handed to [`loss_gradients`].
    fn build_loss(
        &self,
        context: &MlxBindingsContext,
        trainable_arrays: &[MlxArray],
    ) -> Result<MlxArray, MlxCError>;
}

/// One gradient per trainable array, in the same order and shape. They are
/// lazy: evaluating them runs the forward and backward pass together.
pub(crate) struct LossGradients {
    pub(crate) gradients: Vec<MlxArray>,
}

/// Differentiates `trainable_loss` at `trainable_arrays`.
///
/// # Errors
/// Returns the MLX description when the graph cannot be built or traced,
/// for example when an operation in it has no gradient.
pub(crate) fn loss_gradients<Loss: TrainableLoss>(
    trainable_loss: Loss,
    trainable_arrays: &[MlxArray],
) -> Result<LossGradients, MlxCError> {
    let graph_context = MlxBindingsContext::new()?;
    // The one closure here is MLX's own calling convention: its transforms
    // take a function, not a graph.
    let loss_closure = MlxClosure::from_fn(move |closure_inputs: &[MlxArray]| {
        Ok(vec![
            trainable_loss.build_loss(&graph_context, closure_inputs)?,
        ])
    })?;
    let mut differentiated_positions = Vec::with_capacity(trainable_arrays.len());
    for trainable_position in 0..trainable_arrays.len() {
        // A model's trainable arrays number in the hundreds, far below `i32`.
        differentiated_positions.push(trainable_position as i32);
    }
    let value_and_gradient = value_and_grad_of_closure(&loss_closure, &differentiated_positions)?;
    let mut borrowed_inputs = Vec::with_capacity(trainable_arrays.len());
    for trainable_array in trainable_arrays {
        borrowed_inputs.push(trainable_array);
    }
    // MLX's transform returns the loss's value too; training reads only the
    // gradients, and the anchored optimizer checks their length instead.
    let (_loss_values, gradient_values) =
        value_and_gradient.apply(&MlxArrayVector::new(&borrowed_inputs)?)?;
    let mut gradients = Vec::with_capacity(gradient_values.len());
    for gradient_position in 0..gradient_values.len() {
        gradients.push(gradient_values.array_at(gradient_position, "read a gradient")?);
    }
    Ok(LossGradients { gradients })
}

#[cfg(test)]
#[path = "loss_gradient_tests.rs"]
mod tests;
