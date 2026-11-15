//! Mixed-precision key-value cache allocation.
//!
//! The key-value cache dominates memory at long context, and uniform 4-bit
//! quantization of it hurts quality because some layers' caches are far more
//! sensitive than others (OptiQ measured layer 0 at 56 times the average).
//! Protecting the one or two most sensitive layers at 8 bits and running the
//! rest at 4 keeps quality while cutting memory.
//!
//! The measurement is the same probe-and-KL idea as the weight sweep, with
//! the cache quantized per layer instead of the weight; the allocation is
//! the same greedy knapsack on bits per cached element. Both are ported
//! here as pure logic over a [`AstroDecoderStateCacheProbeModel`] trait. Wiring the result
//! into Astronomical's decoder cache (which already supports per-layer
//! quantized full-attention state) is serving work owned by model-serving's
//! `memory` package, not this crate.

pub mod astro_decoder_state_cache_allocator;
pub mod astro_decoder_state_cache_config_document;
pub mod astro_decoder_state_cache_sensitivity;

pub use astro_decoder_state_cache_allocator::{
    AstroDecoderStateCacheLayerConfig, estimate_astro_decoder_state_cache_bytes,
    optimize_astro_decoder_state_cache,
};
pub use astro_decoder_state_cache_config_document::AstroDecoderStateCacheConfigDocument;
pub use astro_decoder_state_cache_sensitivity::{
    AstroDecoderStateCacheProbeModel, AstroDecoderStateCacheSensitivityResult,
    CachedLayerDescriptor, measure_astro_decoder_state_cache_sensitivity,
};

#[cfg(test)]
#[path = "astro_decoder_state_cache_tests.rs"]
mod astro_decoder_state_cache_tests;
