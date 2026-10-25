//! The DFlash speculative-decoding draft model.
//!
//! One resident Qwen3.5 engine may host one DFlash draft beside its target
//! model. The draft is a small dense trunk that never runs alone: its context
//! keys and values are projections of the target's hidden states (captured at
//! the `target_layer_ids` taps), its query block is one verified bonus token
//! plus masked slots, and its proposals are verified by the target in one
//! multi-token forward. This root only wires the children into the resident
//! engine module.

pub mod draft_attention_masks;
pub mod draft_configuration;
pub mod draft_context_kv_projection;
pub mod draft_forward;
pub mod draft_key_value_state;
pub mod draft_weights;
pub mod draft_weights_loading;
pub mod target_hidden_state_conditioning;

#[cfg(test)]
mod draft_attention_masks_tests;
#[cfg(test)]
mod draft_configuration_tests;
