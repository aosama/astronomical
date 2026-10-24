//! Laptop-only performance-throughput journeys.
//!
//! These journeys measure serving throughput for the Qwen3.5-family sparse-MoE
//! artifacts over the worker IPC boundary and append a durable historical
//! record of the numbers they acquire. Measurement journeys report rates;
//! performance tests additionally assert their documented throughput floors.
//! The full-fit diagnostic asserts byte-level residency and source-read
//! invariants rather than a hardware-sensitive throughput threshold.
//!
//! They are intentionally separate from the serving-acceptance journeys, which
//! validate functional behavior over the REST boundary. The measurement surface
//! is gated behind the `performance_throughput` feature and is not wired into
//! CI because each journey loads a real multi-gigabyte model into wired GPU
//! memory, which requires an Apple-Silicon host.

#[cfg(feature = "performance_throughput")]
#[path = "throughput/mod.rs"]
mod performance_throughput;

#[cfg(feature = "performance_throughput")]
#[path = "../support/mod.rs"]
mod support;
