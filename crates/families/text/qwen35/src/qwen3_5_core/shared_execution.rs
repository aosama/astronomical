//! The request steps both Qwen 3.5 engines share that stay with the family:
//! the ones that read its vision tower or its validated artifact. Every
//! other shared step lives in the text shell, `text_generation`, where any
//! text family reaches it (issue #2533).

pub mod astro_ssd_prompt_cache_visual_identity;
pub mod execution_core;
pub mod serving_artifact;
pub mod session_visual_embeddings;
