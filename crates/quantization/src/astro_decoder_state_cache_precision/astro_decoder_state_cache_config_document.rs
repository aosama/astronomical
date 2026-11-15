//! The `kv_config.json` document written beside an artifact.

use serde::{Deserialize, Serialize};

use super::astro_decoder_state_cache_allocator::AstroDecoderStateCacheLayerConfig;

/// The `kv_config.json` document OptiQ's server consumes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AstroDecoderStateCacheConfigDocument {
    pub layers: Vec<AstroDecoderStateCacheLayerConfig>,
    pub target_average_bits: f64,
    pub achieved_average_bits: f64,
}
