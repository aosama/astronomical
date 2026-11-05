//! The Neural Engine research bench (issue #161): what the hardware journeys in
//! `tests/` measure against and report in.
//!
//! - [`accelerate_reference`]: the fp32 numerical oracle.
//! - [`numerical_comparison`], [`timing_summary`], and [`process_memory`]: the
//!   evidence format.
//!
//! `experiments.md` beside this crate logs the A/B series run on real models.

pub mod accelerate_reference;
pub mod numerical_comparison;
pub mod process_memory;
pub mod timing_summary;
