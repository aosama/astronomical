//! Capture into the `astro_ssd_prompt_cache`, shared by every text engine.
//! Each engine implements [`TextAstroSsdPromptCacheCapture`] for the one
//! step that differs between them.

mod block_capture;
mod boundary_checkpoint;
mod capture_hook;
mod tail_capture;

pub use block_capture::astro_ssd_prompt_cache_publication_advances_parent_chain;
pub use block_capture::capture_astro_ssd_prompt_cache_blocks;
pub use boundary_checkpoint::record_astro_ssd_prompt_cache_boundary_checkpoint;
pub use capture_hook::TextAstroSsdPromptCacheCapture;
pub use tail_capture::capture_astro_ssd_prompt_cache_partial_tail_block;
