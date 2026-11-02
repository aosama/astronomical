//! Per-layer cache sensitivity: the probe seam and the KL sweep over it.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::bit_width::{BitWidth, CandidateBitWidths, GroupSize};
use crate::error::QuantizationError;
use crate::sensitivity::CalibrationForwardSample;

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
