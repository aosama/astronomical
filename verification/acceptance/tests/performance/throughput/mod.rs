//! Performance-throughput journeys for the Qwen3.5-family sparse-MoE models
//! the folded ternary dense pack, and a dense 27B decoding with DFlash2 draft
//! trees.
//!
//! The surface measures serving throughput over the worker IPC boundary, records
//! the numbers against this machine's specifications, and appends them to a
//! durable historical log. It asserts the measured token counts and astro expert cache state.

#[cfg(feature = "performance_throughput")]
mod completion;

#[cfg(feature = "performance_throughput")]
mod decode_astro_expert_cache_experiment;

#[cfg(feature = "performance_throughput")]
mod decode_full_fit;

#[cfg(feature = "performance_throughput")]
mod dflash2_draft_trees;

#[cfg(feature = "performance_throughput")]
mod engine_versus_memory_ceiling;

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
mod qwen3_5_moe;

#[cfg(feature = "performance_throughput")]
mod qwen3_5_moe_vision;

#[cfg(feature = "performance_throughput")]
mod streaming_scoreboard;

#[cfg(feature = "performance_throughput")]
mod support;

#[cfg(feature = "performance_throughput")]
mod worker_environment;
