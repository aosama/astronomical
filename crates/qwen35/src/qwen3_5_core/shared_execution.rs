//! The execution steps the resident and streaming Qwen3.5 engines share.
//!
//! Each step here was written once per engine before (issue #2473, Step 2).
//! A step takes only what it reads, either plain values or a model through
//! [`text_model::Qwen3_5TextModel`], so both engines call the same code and a
//! fix to a step reaches both. The trait is reached through static generics,
//! so a step costs the same per token as the copies it replaces.

pub mod astro_ssd_prompt_cache_startup_logging;
pub mod astro_ssd_prompt_cache_visual_identity;
pub mod generated_token_emission;
pub mod generation_request_validation;
pub mod generation_thinking_end_markers;
pub mod generation_vocabulary_boundaries;
pub mod prompt_processing_chunk_sizer;
pub mod request_memory_release;
pub mod text_model;
