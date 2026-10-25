//! Performance-throughput journeys for the Qwen3.5-family sparse-MoE models
//! and the folded ternary dense pack.
//!
//! The surface measures serving throughput over the worker IPC boundary, records
//! the numbers against this machine's specifications, and appends them to a
//! durable historical log. It asserts the measured token counts and cache state.

#[cfg(feature = "performance_throughput")]
mod completion;

#[cfg(feature = "performance_throughput")]
mod decode_cache_experiment;

#[cfg(feature = "performance_throughput")]
mod decode_full_fit;

#[cfg(feature = "performance_throughput")]
mod historical_record;

#[cfg(feature = "performance_throughput")]
mod jangtq2_qwen35;

#[cfg(feature = "performance_throughput")]
mod machine_specs;

#[cfg(feature = "performance_throughput")]
mod memory_ceiling_sweep;

#[cfg(feature = "performance_throughput")]
mod prism_hadamard_qwen35;

#[cfg(feature = "performance_throughput")]
mod resident_memory_ceiling_probe;

#[cfg(feature = "performance_throughput")]
mod qwen3_5_moe;

#[cfg(feature = "performance_throughput")]
mod qwen3_5_moe_vision;

#[cfg(feature = "performance_throughput")]
mod support;

#[cfg(feature = "performance_throughput")]
mod worker_environment;
