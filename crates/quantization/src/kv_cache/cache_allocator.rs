//! The greedy cache-bit allocator and the byte estimate that reports what
//! an allocation costs at a given context length.

use serde::{Deserialize, Serialize};

use crate::bit_width::{BitWidth, CandidateBitWidths, GroupSize};
use crate::error::QuantizationError;

use super::cache_sensitivity::KvCacheSensitivityResult;

/// One layer's cache decision.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct KvCacheLayerConfig {
    pub layer_index: usize,
    pub layer_name: String,
    pub bit_width: BitWidth,
    pub group_size: GroupSize,
}

/// Greedy allocation under an average cache bits target, weighting each
/// layer by its bytes per token, with the same KL-reduction-per-bit heap as
/// the weight allocator and no block rules.
pub fn optimize_kv_cache(
    results: &[KvCacheSensitivityResult],
    target_average_bits: f64,
    candidate_bit_widths: &CandidateBitWidths,
    group_size: GroupSize,
) -> Result<Vec<KvCacheLayerConfig>, QuantizationError> {
    let _ = (
        results,
        target_average_bits,
        candidate_bit_widths,
        group_size,
    );
    todo!("start at the lowest width, upgrade by efficiency until the average meets the target")
}

/// Cache bytes for a context length under a per-layer config, for the
/// memory report.
#[must_use]
pub fn estimate_kv_cache_bytes(
    configs: &[KvCacheLayerConfig],
    results: &[KvCacheSensitivityResult],
    context_length_tokens: usize,
) -> u64 {
    let _ = (configs, results, context_length_tokens);
    todo!(
        "sum over layers of context * cache_bytes_per_token * bits / 16, plus per-group scale and bias overhead"
    )
}
