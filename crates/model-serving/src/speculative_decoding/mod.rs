//! Speculative decoding: a small drafter proposes several tokens, and the
//! target checks them all in one pass and keeps the ones it would have
//! written itself.
//!
//! This module is family-neutral. A target lends the drafter its embedding
//! rows, output head and tapped hidden rows; nothing here knows which model
//! family that is.

pub mod dflash2_drafter;
pub mod keyed_sampling;
pub mod tree_rows;
pub mod verification_decision;

#[cfg(test)]
mod keyed_sampling_tests;
#[cfg(test)]
mod tree_rows_tests;
#[cfg(test)]
mod verification_decision_tests;
