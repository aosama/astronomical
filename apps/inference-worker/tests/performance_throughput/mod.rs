//! Performance-throughput journeys for the Qwen3.5-family sparse-MoE models.
//!
//! The surface measures serving throughput over the worker IPC boundary, records
//! the numbers against this machine's specifications, and appends them to a
//! durable historical log. It asserts nothing about the measured rates.

#[cfg(feature = "performance_throughput")]
mod historical_record;

#[cfg(feature = "performance_throughput")]
mod machine_specs;

#[cfg(feature = "performance_throughput")]
mod qwen3_5_moe;

#[cfg(feature = "performance_throughput")]
mod support;
