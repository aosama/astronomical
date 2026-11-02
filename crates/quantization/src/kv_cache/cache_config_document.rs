//! The `kv_config.json` document written beside an artifact.

use serde::{Deserialize, Serialize};

use super::cache_allocator::KvCacheLayerConfig;

/// The `kv_config.json` document OptiQ's server consumes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KvCacheConfigDocument {
    pub layers: Vec<KvCacheLayerConfig>,
    pub target_average_bits: f64,
    pub achieved_average_bits: f64,
}
