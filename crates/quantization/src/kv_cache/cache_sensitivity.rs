//! Per-layer cache sensitivity: the probe seam and the KL sweep over it.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::bit_width::{BitWidth, CandidateBitWidths, GroupSize};
use crate::error::QuantizationError;
use crate::progress::{ProgressReporter, StageProgressTracker};
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

/// A layer's measured KL below this on every width means the quantized
/// cache never engaged (degenerate calibration text, or a cache the forward
/// ignored), not that the model is robust.
pub const MINIMUM_ENGAGED_KL_DIVERGENCE: f64 = 1e-6;

/// Measures per-layer cache sensitivity, OptiQ's `measure_kv_sensitivity`:
/// full-precision reference logits, then for each cached layer and each
/// candidate width, one forward per sample with only that layer's cache
/// quantized, the mean KL clamped at zero (negative means are rounding
/// noise). The keep-bfloat16 tier is full precision by definition and
/// records zero without a forward. Refuses (`ProbeModelFailure`) when no
/// layer reports a cache or when no KL reaches
/// [`MINIMUM_ENGAGED_KL_DIVERGENCE`]. Progress advances once per layer.
pub fn measure_kv_cache_sensitivity<ProbeModel: KvCacheProbeModel>(
    probe_model: &mut ProbeModel,
    calibration_samples: &[CalibrationForwardSample],
    candidate_bit_widths: &CandidateBitWidths,
    group_size: GroupSize,
    progress_reporter: &mut dyn ProgressReporter,
) -> Result<Vec<KvCacheSensitivityResult>, QuantizationError> {
    let cached_layers = probe_model.cached_layers();
    if cached_layers.is_empty() {
        return Err(probe_failure(
            "no layer has a key-value cache this tool can quantize",
        ));
    }
    let full_precision = BTreeMap::new();
    let mut reference_logits = Vec::with_capacity(calibration_samples.len());
    for calibration_sample in calibration_samples {
        reference_logits.push(probe_model.forward_with_cache_bit_widths(
            calibration_sample,
            &full_precision,
            group_size,
        )?);
    }
    let mut progress_tracker = StageProgressTracker::new(
        "key-value cache sweep",
        cached_layers.len() as u64,
        0,
        StageProgressTracker::DEFAULT_REPORT_INTERVAL,
        progress_reporter,
    );
    let mut results = Vec::with_capacity(cached_layers.len());
    for cached_layer in cached_layers {
        let mut kl_divergence_by_bit_width = BTreeMap::new();
        for bit_width in candidate_bit_widths.as_slice() {
            let probe = LayerProbe {
                cached_layer: &cached_layer,
                bit_width: *bit_width,
                group_size,
            };
            let mean_kl =
                probe.mean_kl_divergence(probe_model, calibration_samples, &reference_logits)?;
            kl_divergence_by_bit_width.insert(*bit_width, mean_kl);
        }
        progress_tracker.advance(&cached_layer.layer_name);
        results.push(layer_result(cached_layer, kl_divergence_by_bit_width));
    }
    progress_tracker.report_now("key-value cache sweep complete");
    require_engaged_measurement(&results)?;
    Ok(results)
}

/// One layer at one width.
struct LayerProbe<'layer> {
    cached_layer: &'layer CachedLayerDescriptor,
    bit_width: BitWidth,
    group_size: GroupSize,
}

impl LayerProbe<'_> {
    /// Mean KL over the samples with only this layer's cache quantized,
    /// clamped at zero.
    fn mean_kl_divergence<ProbeModel: KvCacheProbeModel>(
        &self,
        probe_model: &mut ProbeModel,
        calibration_samples: &[CalibrationForwardSample],
        reference_logits: &[ProbeModel::Logits],
    ) -> Result<f64, QuantizationError> {
        if self.bit_width.keeps_bfloat16() || calibration_samples.is_empty() {
            return Ok(0.0);
        }
        let bit_width_by_layer = BTreeMap::from([(self.cached_layer.layer_index, self.bit_width)]);
        let mut kl_sum = 0.0;
        for (calibration_sample, sample_reference) in
            calibration_samples.iter().zip(reference_logits)
        {
            let probe_logits = probe_model.forward_with_cache_bit_widths(
                calibration_sample,
                &bit_width_by_layer,
                self.group_size,
            )?;
            kl_sum += probe_model.kl_divergence(sample_reference, &probe_logits)?;
        }
        Ok((kl_sum / calibration_samples.len() as f64).max(0.0))
    }
}

fn layer_result(
    cached_layer: CachedLayerDescriptor,
    kl_divergence_by_bit_width: BTreeMap<BitWidth, f64>,
) -> KvCacheSensitivityResult {
    // Keys and values, each float16, per head dimension per head.
    let cache_bytes_per_token = 2.0
        * f64::from(cached_layer.key_value_head_count)
        * f64::from(cached_layer.head_dimension)
        * 2.0;
    KvCacheSensitivityResult {
        layer_index: cached_layer.layer_index,
        layer_name: cached_layer.layer_name,
        kl_divergence_by_bit_width,
        key_value_head_count: cached_layer.key_value_head_count,
        head_dimension: cached_layer.head_dimension,
        cache_bytes_per_token,
    }
}

/// OptiQ's `_assert_measured`: a peak KL below the floor is a broken
/// measurement.
fn require_engaged_measurement(
    results: &[KvCacheSensitivityResult],
) -> Result<(), QuantizationError> {
    let mut peak_kl_divergence = 0.0_f64;
    for result in results {
        for kl_divergence in result.kl_divergence_by_bit_width.values() {
            peak_kl_divergence = peak_kl_divergence.max(*kl_divergence);
        }
    }
    if peak_kl_divergence < MINIMUM_ENGAGED_KL_DIVERGENCE {
        return Err(probe_failure(&format!(
            "key-value cache sensitivity is about zero on every layer (peak {peak_kl_divergence:.1e}): quantizing a layer's cache did not change the output, so the measurement is broken"
        )));
    }
    Ok(())
}

fn probe_failure(description: &str) -> QuantizationError {
    QuantizationError::ProbeModelFailure {
        description: description.to_owned(),
    }
}
