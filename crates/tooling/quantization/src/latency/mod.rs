//! Hardware-aware latency model for Apple Silicon MLX inference.
//!
//! Models per-module inference cost with the roofline model: decode
//! (sequence length one) is memory-bandwidth-bound, so latency scales with
//! bytes read; prefill is compute-bound, so latency scales with floating
//! point operations over peak throughput. Lower bit widths read fewer bytes
//! per weight and so run faster at decode, but not linearly: per-group
//! scale and bias overhead, fixed dequantization cost per element, and
//! per-launch dispatch overhead all intrude.
//!
//! The port keeps OptiQ's analytical form and its per-width factors. Two
//! things change: the chip table is data the caller supplies (a published
//! specification table, or a profile measured on the host), never a
//! hardwired assumption about the machine running the code, and the
//! calibration against a real throughput measurement is the preferred path
//! because Astronomical already measures tokens per second in its
//! performance lanes.

pub mod calibrated_model;
pub mod hardware_profile;
pub mod roofline;

pub use calibrated_model::CalibratedLatencyModel;
pub use hardware_profile::HardwareProfile;
pub use roofline::{
    LatencyModelParameters, ModuleByteFootprint, ModuleLatencyEstimate, estimate_module_bytes,
    estimate_module_latency, estimate_module_operations, infer_module_dimensions,
};

#[cfg(test)]
#[path = "latency_tests.rs"]
mod latency_tests;
