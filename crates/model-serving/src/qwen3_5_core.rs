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
//! from the engine modules.

pub(crate) mod artifacts;
pub(crate) mod configuration;
pub(crate) mod decoder;
pub(crate) mod dense;
pub(crate) mod model;
pub(crate) mod model_math;
pub(crate) mod quantizations;
pub(crate) mod route_observation;
pub(crate) mod text;
pub(crate) mod vision;
