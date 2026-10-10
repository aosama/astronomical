//! The shared Qwen3.5 family kernel (issue #1132).
//!
//! This module owns everything both Qwen3.5 engines consume and neither engine
//! may fork: configuration, artifact validation and shard plumbing, OptiQ
//! quantization profiles, the text stack (tokenizer, prompt, output parser,
//! sampler), the vision stack, dense MLP math, decoder state and cache
//! layouts, and the persistent prompt-cache machinery that must stay bridged
//! across engines so a resident-to-streaming retry can restore a cached
//! prefix.
//!
//! Dependency direction: `qwen3_5_resident` and `qwen3_5_streaming` may import
//! from here; neither may import from the other. This module never imports
//! from the engine modules. Until the engine fork lands (migration steps 2
//! and 3 of issue #1132), the legacy `qwen3_5` module re-exports these trees
//! so every existing `crate::qwen3_5::` path keeps resolving.

pub(crate) mod artifacts;
pub(crate) mod configuration;
pub(crate) mod decoder;
pub(crate) mod dense;
pub(crate) mod quantizations;
pub(crate) mod text;
pub(crate) mod vision;
