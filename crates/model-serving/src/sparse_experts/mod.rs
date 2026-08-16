//! Family-neutral stacked-expert assignment sort and weighted reduction.
//!
//! Callers own routing, top-K policy, score formulas, shared-expert combination,
//! and whether to sort. This package only permutes stacked assignments, reduces
//! scored outputs, and documents that `sorted_indices=true` is legal only after
//! the assignments were sorted.

mod assignment_permutation;
mod error;

#[cfg(feature = "direct-mlx")]
mod assignment_sort;
#[cfg(feature = "direct-mlx")]
mod weighted_sum;

pub use assignment_permutation::{gathered_indices_use_sorted_contract, invert_assignment_order};
#[cfg(feature = "direct-mlx")]
pub use assignment_sort::{
    SortedExpertAssignments, restore_expert_assignment_order, sort_expert_assignments,
};
pub use error::SparseExpertError;
#[cfg(feature = "direct-mlx")]
pub use weighted_sum::{
    router_weighted_expert_inputs, sorted_expert_weighted_sum, sorted_expert_weighted_sum_kernel,
    unsorted_expert_weighted_sum,
};
