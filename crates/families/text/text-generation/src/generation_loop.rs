//! The steps of the generation loop every text family shares.
//!
//! Each step was first written once per Qwen 3.5 engine, then shared by them
//! (issue #2473, Step 2), and now serves any text family (issue #2533). A
//! step takes only what it reads, either plain values or a model through
//! [`crate::model::text_model::TextModel`], and reaches what differs between
//! engines through a hook trait the engine implements. The traits are
//! reached through static generics, so a step costs the same per token as
//! the copies it replaced.

pub mod astro_ssd_prompt_cache_restore;
pub mod astro_ssd_prompt_cache_startup_logging;
pub mod astro_ssd_prompt_cache_state_install;
pub mod decode_step;
pub mod decode_step_decisions;
pub mod engine_construction;
pub mod engine_request_installation;
pub mod execution_core;
pub mod forward_memory_governance;
pub mod generated_token_emission;
pub mod generation_finalization_collection;
pub mod generation_finalization_steps;
pub mod generation_request_validation;
pub mod generation_session_position;
pub mod generation_start_steps;
pub mod generation_thinking_end_markers;
pub mod generation_visual_embedding_validation;
pub mod generation_vocabulary_boundaries;
pub mod inference_execution_bridge;
pub mod input_injection;
pub mod mlx_memory_limit_change;
pub mod model_loading_lifecycle;
pub mod model_loading_steps;
pub mod prompt_prefill_errors;
pub mod prompt_processing_chunk_sizer;
pub mod prompt_reading_steps;
pub mod request_end;
pub mod request_memory_release;
pub mod restore_transient_workspace;
pub mod serving_artifact;
pub mod test_controls;
