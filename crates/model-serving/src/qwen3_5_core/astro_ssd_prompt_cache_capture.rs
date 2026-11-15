//! Persistent astro SSD prompt cache capture shared by the resident and streaming
//! Qwen3.5 engines. Each engine implements [`Qwen3_5AstroSsdPromptCacheCapture`]
//! for the one step that differs between them.

mod block_capture;
mod boundary_checkpoint;
mod capture_hook;
mod tail_capture;

pub use block_capture::astro_ssd_prompt_cache_publication_advances_parent_chain;
pub(crate) use block_capture::capture_astro_ssd_prompt_cache_blocks;
pub(crate) use boundary_checkpoint::record_astro_ssd_prompt_cache_boundary_checkpoint;
pub(crate) use capture_hook::Qwen3_5AstroSsdPromptCacheCapture;
pub(crate) use tail_capture::capture_astro_ssd_prompt_cache_partial_tail_block;
