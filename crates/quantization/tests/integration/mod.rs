//! Integration journeys, one module per seam:
//!
//! - `fixtures`: loaders for the Python-generated parity documents.
//! - `allocation_parity`: the Rust allocator reproduces the Python
//!   allocation bit for bit on the synthetic table, across five constraint
//!   sets.
//! - `taxonomy_parity`: the module taxonomy agrees with OptiQ's predicates
//!   on every name in the fixture.
//! - `static_scorer_parity`: structural priorities match to float rounding.
//! - `encoder_reference_parity`: the pure reference encoder reproduces the
//!   Python range search's packed codes exactly and its scales and biases
//!   within float rounding, and the KL formula matches on a logits pair.
//! - `calibration_mix`: the bundled mix loads end to end with a fake
//!   tokenizer.
//! - `module_inventory_journeys`: inventory against real safetensors files
//!   written with the `safetensors` crate into a temporary directory.
//! - `requantize_plan`: plan construction and byte estimates against those
//!   files.
//! - `artifact_round_trip`: the writers produce documents this crate's own
//!   readers parse back; the cross-crate contract with model-serving's
//!   strict readers lives in model-serving's integration tests.

mod allocation_parity;
mod artifact_round_trip;
mod calibration_mix;
mod encoder_reference_parity;
mod fixtures;
mod module_inventory_journeys;
mod requantize_plan;
mod static_scorer_parity;
mod taxonomy_parity;
