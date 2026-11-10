//! Persistent prompt-cache capture shared by the resident and streaming
//! Qwen3.5 engines. Each engine implements [`Qwen3_5PromptCacheCapture`]
//! for the one step that differs between them.

mod block_capture;
mod capture_engine;
mod tail_capture;

pub use block_capture::persistent_prompt_cache_publication_advances_parent_chain;
pub(crate) use block_capture::{
    capture_persistent_prompt_cache_blocks, required_prompt_state_persistence_failure,
};
pub(crate) use capture_engine::Qwen3_5PromptCacheCapture;
pub(crate) use tail_capture::capture_persistent_prompt_cache_partial_tail_block;
