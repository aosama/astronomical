//! Hermetic (CPU-only, no GPU) tests for the Qwen-Image-2.1 image-generation family.

mod artifact;
mod artifact_fixture;
mod engine;
mod kv_cache;
mod kv_cache_fixture;
mod latent_geometry;
mod latent_layout;
mod mask;
mod mask_fixture;
mod modulation;
mod modulation_fixture;
mod norm;
mod norm_fixture;
mod rope;
mod scheduler;
mod scheduler_fixture;
mod support;
mod text_conditioning;
mod time;
