//! The execution steps the resident and streaming Qwen3.5 engines share.
//!
//! Each step here was written once per engine before (issue #2473, Step 2).
//! A step takes only what it reads, either plain values or a model through
//! [`text_model::Qwen3_5TextModel`], so both engines call the same code and a
//! fix to a step reaches both. The trait is reached through static generics,
//! so a step costs the same per token as the copies it replaces.

pub mod astro_ssd_prompt_cache_restore;
pub mod astro_ssd_prompt_cache_startup_logging;
pub mod astro_ssd_prompt_cache_state_install;
pub mod astro_ssd_prompt_cache_visual_identity;
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
pub mod text_model;
