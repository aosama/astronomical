//! Mixed-precision bit allocation: turn a sensitivity table into one bit
//! width per module under a bits-per-weight target.
//!
//! The allocator is a greedy knapsack with two block-aware constraints that
//! OptiQ added after field failures:
//!
//! 1. Start every module at the lowest candidate width.
//! 2. Block-aware floor ([`block_floor`]): pre-upgrade the most sensitive
//!    components of every transformer block to the second-lowest width,
//!    reserving the first slots for attention. Without this the feed-forward
//!    projections win every per-block ranking and whole attention stacks stay
//!    at the lowest width, which collapsed instruction following on late
//!    layers of Qwen3.5-4B.
//! 3. Greedy upgrades ([`knapsack`]): repeatedly promote the module whose next
//!    step up buys the most KL reduction per bit spent, until the running
//!    bits-per-weight meets the target. The target is a floor, not a cap:
//!    slight overshoot from the discrete grid is accepted because the user
//!    asked for a quality floor.
//! 4. Block-run guard ([`block_run_guard`]): after the greedy pass, break any
//!    run of more than `maximum_low_bit_block_run` consecutive blocks whose
//!    majority of components still sits at the lowest width.
//!
//! Protected modules ([`protected_modules`]) take part in the greedy pass like
//! every other module but with their KL multiplied by 100, so they are
//! upgraded first rather than exempted from the budget.
//!
//! [`optimizer`] drives the four steps above and summarizes the result.
//!
//! The allocator is pure: no MLX, no I/O, deterministic for a given table.
//! That makes it the natural place to introduce Astronomical's own policy
//! modifications, and the parity fixtures under `tests/fixtures/` pin the
//! current behavior to the Python reference so a modification is a visible,
//! intended diff rather than drift.

pub mod bits_per_weight;
pub mod block_floor;
pub mod block_run_guard;
pub mod knapsack;
pub mod optimizer;
pub mod predicate;
pub mod protected_modules;

pub use optimizer::{
    AllocationConstraints, AllocationResult, ModuleQuantizationDecision, optimize_mixed_precision,
};
pub use predicate::QuantizationPredicate;
