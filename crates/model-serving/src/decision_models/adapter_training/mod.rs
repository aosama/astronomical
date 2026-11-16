//! Training add-on weights inside every layer of a decision model, with
//! gradients from MLX itself, and the parts every family's whole-model
//! training shares: the loss transform, the micro-batch objective and the
//! parameter layout.

pub(crate) mod loss_gradient;
pub(crate) mod micro_batch_objective;
pub(crate) mod whole_model_layout;
