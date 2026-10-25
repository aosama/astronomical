//! Unit contracts for folded-pack validation.
//!
//! The fixture builds a tiny but structurally exact pack: real safetensors
//! bytes, a real Hadamard manifest, and the full module/auxiliary inventory
//! the family contract requires, at dimensions where every packed width
//! divides into 128-element ternary groups.

pub(crate) mod acceptance_tests;
pub(crate) mod pack_fixture;
mod refusal_tests;
