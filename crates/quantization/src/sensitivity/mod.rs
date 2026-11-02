//! Per-module quantization sensitivity: the one measurement that drives
//! weight bits, key-value cache bits, and (in OptiQ) adapter ranks.
//!
//! Two methods produce the same [`SensitivityTable`]:
//!
//! - [`exact_sweep`]: for every module and every candidate bit width,
//!   simulate-quantize just that module, forward the calibration samples, and
//!   measure the KL divergence of the output logits against an unmodified
//!   reference. Cost is `modules × widths × samples` forward passes. The
//!   forward pass is supplied by a [`SensitivityProbeModel`], which
//!   model-serving implements for Qwen3.5 so this crate never depends on an
//!   engine.
//! - [`static_scorer`]: architecture-only priority with no measurement, the
//!   fallback for bases too large to sweep. OptiQ measured it at the same
//!   downstream accuracy as the exact method on small dense models and 125
//!   times faster.
//!
//! [`sensitivity_table`] holds the sweep output both methods produce, and
//! [`kl_divergence`] pins the formula every probe implements on device.
//! [`checkpoint`] persists a partially complete sweep so a killed run resumes
//! instead of restarting, and [`report`] prints the top-N ranking.

pub mod checkpoint;
pub mod exact_sweep;
pub mod kl_divergence;
pub mod probe;
pub mod report;
pub mod sensitivity_table;
pub mod static_scorer;

pub use probe::{CalibrationForwardSample, ModuleRestoreToken, SensitivityProbeModel};
pub use sensitivity_table::{SensitivityMethod, SensitivityResult, SensitivityTable};
