//! Opt-in prompt reading that runs part of each full chunk of a dense model
//! on the Apple Neural Engine beside the GPU (issue #161), and everything
//! that prepares, measures, keeps and releases its programs.
//!
//! It lives in the shared core so that any engine holding a dense model's
//! layers in memory can attach the programs (issue #2585): the programs and
//! the choice of engines for a chunk hang off the model base, and nothing
//! here names an engine.

pub mod astro_neural_engine_program_cache;
pub mod host_description;
pub mod neural_engine_chunk_comparison;
pub mod neural_engine_idle_watch;
pub mod neural_engine_mlp_prefill;
pub mod neural_engine_output_gate_prefill;
pub mod neural_engine_prefill_loading;
pub mod neural_engine_program_measurement;
pub mod neural_engine_program_memory_record;
pub mod neural_engine_program_placement_record;
pub mod neural_engine_program_preparation;
pub mod neural_engine_program_settlement;
pub mod neural_engine_program_speed_record;
