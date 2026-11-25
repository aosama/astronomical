//! The role names a model's `astro_decoder_state_cache` layout gives its
//! state tensors. A family's layout builder writes them and the request
//! state reads them back, so one copy keeps the two in step.

pub const LINEAR_CONVOLUTION_TENSOR_ROLE: &str = "linear.convolution";
pub const LINEAR_RECURRENCE_TENSOR_ROLE: &str = "linear.gated_delta_recurrent";
pub const ATTENTION_KEYS_TENSOR_ROLE: &str = "attention.keys";
pub const ATTENTION_VALUES_TENSOR_ROLE: &str = "attention.values";
