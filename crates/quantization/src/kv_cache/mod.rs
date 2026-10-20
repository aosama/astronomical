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
//! here as pure logic over a [`KvCacheProbeModel`] trait. Wiring the result
//! into Astronomical's decoder cache (which already supports per-layer
//! quantized full-attention state) is serving work owned by model-serving's
//! `memory` package, not this crate.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::bit_width::{BitWidth, CandidateBitWidths, GroupSize};
use crate::error::QuantizationError;
use crate::sensitivity::CalibrationForwardSample;

#[cfg(test)]
#[path = "kv_cache_tests.rs"]
mod kv_cache_tests;

/// Per-layer cache sensitivity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KvCacheSensitivityResult {
    pub layer_index: usize,
    pub layer_name: String,
    pub kl_divergence_by_bit_width: BTreeMap<BitWidth, f64>,
    pub key_value_head_count: u32,
    pub head_dimension: u32,
    /// Bytes per token at full precision: `2 * heads * head_dim * 2`.
    pub cache_bytes_per_token: f64,
}

/// One decoder layer that owns a key-value cache, as the probe reports it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CachedLayerDescriptor {
    pub layer_index: usize,
    pub layer_name: String,
    pub key_value_head_count: u32,
    pub head_dimension: u32,
}

/// A loaded model whose per-layer cache precision can be set for a probe.
pub trait KvCacheProbeModel {
    type Logits;

    /// Layers that own a key-value cache (linear-attention layers do not).
    fn cached_layers(&self) -> Vec<CachedLayerDescriptor>;

    /// Generates with the given per-layer cache widths (`None` = full
    /// precision) and returns logits for the sample's positions.
    fn forward_with_cache_bit_widths(
        &mut self,
        sample: &CalibrationForwardSample,
        bit_width_by_layer: &BTreeMap<usize, BitWidth>,
        group_size: GroupSize,
    ) -> Result<Self::Logits, QuantizationError>;

    fn kl_divergence(
        &self,
        reference_logits: &Self::Logits,
        probe_logits: &Self::Logits,
    ) -> Result<f64, QuantizationError>;
}

/// Measures per-layer cache sensitivity: full-precision reference logits,
/// then for each cached layer and each candidate width, one generation with
/// only that layer's cache quantized. Refuses (`ProbeModelFailure`) when no
/// layer reports a cache or when every KL is below `1e-6`, which means the
/// quantization never engaged.
pub fn measure_kv_cache_sensitivity<ProbeModel: KvCacheProbeModel>(
    probe_model: &mut ProbeModel,
    calibration_samples: &[CalibrationForwardSample],
    candidate_bit_widths: &CandidateBitWidths,
    group_size: GroupSize,
) -> Result<Vec<KvCacheSensitivityResult>, QuantizationError> {
    let _ = (
        probe_model,
        calibration_samples,
        candidate_bit_widths,
        group_size,
    );
    todo!("reference pass, per-layer per-width probes, the engaged guard")
}

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

/// The `kv_config.json` document OptiQ's server consumes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KvCacheConfigDocument {
    pub layers: Vec<KvCacheLayerConfig>,
    pub target_average_bits: f64,
    pub achieved_average_bits: f64,
}
