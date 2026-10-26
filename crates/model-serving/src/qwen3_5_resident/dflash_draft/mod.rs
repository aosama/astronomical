//! The DFlash speculative-decoding draft model.
//!
//! COMPARTMENT STATUS: DISABLED BY DESIGN — NOT REACHABLE FROM ENGINE
//! WIRING. The configuration key is accepted and ignored, the worker wire
//! contract carries no block length, and model loading never installs a
//! draft runtime, so no session can reach this trunk. Corrected journey
//! accounting measured token-equality verification at the realistic
//! serving temperature (1.0) as a throughput loss on this workload
//! (0.50x-1.15x across trajectories; the greedy ceiling is only ~1.6x),
//! and the open question — an unresolved defect versus a weak draft
//! artifact — is to be settled by independently reproducing the vendor's
//! published speedups under vLLM before any re-enable. The modules stay
//! compiled and unit-tested as the diagnostic compartment for that day;
//! the proof journey exercises them directly, bypassing engine wiring.
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
