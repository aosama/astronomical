//! Family-neutral decode expert cache.
//!
//! The irreducible unit is one expert id in one sparse-layer slot. Bytes are
//! accounting. This module does not import a model family, MLX, or a page type.

mod cache;
mod demand_ledger;
mod eviction;
mod resident_set;
mod weight;

pub use cache::DecodeExpertCache;
pub use weight::ResidentExpertWeight;
