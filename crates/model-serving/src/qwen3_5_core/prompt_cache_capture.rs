//! Persistent prompt-cache capture shared by the resident and streaming
//! Qwen3.5 engines. Each engine implements [`Qwen3_5PromptCacheCapture`]
//! for the one step that differs between them.

mod block_capture;
mod boundary_checkpoint;
mod capture_hook;
mod tail_capture;

pub(crate) use block_capture::capture_persistent_prompt_cache_blocks;
pub use block_capture::persistent_prompt_cache_publication_advances_parent_chain;
pub(crate) use boundary_checkpoint::record_persistent_prompt_cache_boundary_checkpoint;
pub(crate) use capture_hook::Qwen3_5PromptCacheCapture;
pub(crate) use tail_capture::capture_persistent_prompt_cache_partial_tail_block;
