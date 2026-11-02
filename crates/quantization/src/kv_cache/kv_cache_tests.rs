//! Unit tests for the key-value cache allocator.

use super::*;
use crate::bit_width::{BitWidth, CandidateBitWidths, GroupSize};
use std::collections::BTreeMap;

fn bits(value: u32) -> BitWidth {
    BitWidth::new(value).expect("valid")
}

fn result(layer_index: usize, kl_at_four: f64) -> KvCacheSensitivityResult {
    KvCacheSensitivityResult {
        layer_index,
        layer_name: format!("model.layers.{layer_index}.self_attn"),
        kl_divergence_by_bit_width: BTreeMap::from([(bits(4), kl_at_four), (bits(8), 0.0)]),
        key_value_head_count: 8,
        head_dimension: 128,
        cache_bytes_per_token: 2.0 * 8.0 * 128.0 * 2.0,
    }
}

#[test]
fn should_protect_the_most_sensitive_layer_first() {
    // Layer 0 is 56x more sensitive than the others; at a 5.0 average over
    // four equal layers exactly one layer goes to 8 bits and it is layer 0.
    let results = vec![
        result(0, 0.56),
        result(1, 0.01),
        result(2, 0.01),
        result(3, 0.01),
    ];
    let candidate_bit_widths = CandidateBitWidths::new(&[4, 8]).expect("valid ladder");

    let configs = optimize_kv_cache(
        &results,
        5.0,
        &candidate_bit_widths,
        GroupSize::new(64).expect("valid"),
    )
    .expect("allocation should succeed");

    let mut eight_bit_layer_indices = Vec::new();
    for config in &configs {
        if config.bit_width == bits(8) {
            eight_bit_layer_indices.push(config.layer_index);
        }
    }
    assert_eq!(eight_bit_layer_indices, vec![0]);
}

#[test]
fn should_scale_cache_bytes_with_bit_width() {
    let results = vec![result(0, 0.1)];
    let four_bit = vec![KvCacheLayerConfig {
        layer_index: 0,
        layer_name: "model.layers.0.self_attn".to_owned(),
        bit_width: bits(4),
        group_size: GroupSize::new(64).expect("valid"),
    }];
    let eight_bit = vec![KvCacheLayerConfig {
        bit_width: bits(8),
        ..four_bit[0].clone()
    }];

    assert!(
        estimate_kv_cache_bytes(&four_bit, &results, 4096)
            < estimate_kv_cache_bytes(&eight_bit, &results, 4096)
    );
}

#[test]
fn should_keep_every_layer_at_the_lowest_width_when_no_upgrade_reduces_kl() {
    let flat = vec![result(0, 0.0), result(1, 0.0)];
    let candidate_bit_widths = CandidateBitWidths::new(&[4, 8]).expect("valid ladder");

    let configs = optimize_kv_cache(
        &flat,
        8.0,
        &candidate_bit_widths,
        GroupSize::new(64).expect("valid"),
    )
    .expect("allocation should succeed");

    for config in &configs {
        assert_eq!(config.bit_width, bits(4));
    }
}

#[test]
fn should_return_no_configs_for_no_cached_layers() {
    let candidate_bit_widths = CandidateBitWidths::new(&[4, 8]).expect("valid ladder");

    let configs = optimize_kv_cache(
        &[],
        6.0,
        &candidate_bit_widths,
        GroupSize::new(64).expect("valid"),
    )
    .expect("allocation should succeed");

    assert!(configs.is_empty());
}

#[test]
fn should_skip_configs_without_a_measured_layer_when_estimating_bytes() {
    let orphan = vec![KvCacheLayerConfig {
        layer_index: 9,
        layer_name: "model.layers.9.self_attn".to_owned(),
        bit_width: bits(4),
        group_size: GroupSize::new(64).expect("valid"),
    }];

    assert_eq!(estimate_kv_cache_bytes(&orphan, &[result(0, 0.1)], 4096), 0);
}

/// A probe whose KL is scripted per layer: quantizing layer `n` moves the
/// output by `kl_by_layer[n] / bits`, so the sweep's bookkeeping is visible.
struct ScriptedCacheProbe {
    cached_layer_indices: Vec<usize>,
    kl_by_layer: BTreeMap<usize, f64>,
    forward_count: usize,
}

impl KvCacheProbeModel for ScriptedCacheProbe {
    type Logits = f64;

    fn cached_layers(&self) -> Vec<CachedLayerDescriptor> {
        let mut cached_layers = Vec::new();
        for layer_index in &self.cached_layer_indices {
            cached_layers.push(CachedLayerDescriptor {
                layer_index: *layer_index,
                layer_name: format!("model.layers.{layer_index}"),
                key_value_head_count: 4,
                head_dimension: 64,
            });
        }
        cached_layers
    }

    fn forward_with_cache_bit_widths(
        &mut self,
        _sample: &crate::sensitivity::CalibrationForwardSample,
        bit_width_by_layer: &BTreeMap<usize, BitWidth>,
        _group_size: GroupSize,
    ) -> Result<f64, crate::error::QuantizationError> {
        self.forward_count += 1;
        let mut shifted_logit = 0.0;
        for (layer_index, bit_width) in bit_width_by_layer {
            shifted_logit += self.kl_by_layer[layer_index] / f64::from(bit_width.bits());
        }
        Ok(shifted_logit)
    }

    fn kl_divergence(
        &self,
        reference_logits: &f64,
        probe_logits: &f64,
    ) -> Result<f64, crate::error::QuantizationError> {
        Ok(probe_logits - reference_logits)
    }
}

fn two_samples() -> Vec<crate::sensitivity::CalibrationForwardSample> {
    vec![
        crate::sensitivity::CalibrationForwardSample {
            token_ids: vec![1, 2],
        },
        crate::sensitivity::CalibrationForwardSample {
            token_ids: vec![3, 4],
        },
    ]
}

#[test]
fn should_measure_each_cached_layer_at_each_width_against_full_precision() {
    let mut probe = ScriptedCacheProbe {
        cached_layer_indices: vec![3, 7],
        kl_by_layer: BTreeMap::from([(3, 0.8), (7, 0.08)]),
        forward_count: 0,
    };
    let mut silent_reporter = crate::progress::SilentProgressReporter;

    let results = measure_kv_cache_sensitivity(
        &mut probe,
        &two_samples(),
        &CandidateBitWidths::new(&[4, 8]).expect("valid ladder"),
        GroupSize::new(64).expect("valid"),
        &mut silent_reporter,
    )
    .expect("the sweep should succeed");

    // Two reference forwards, then two layers x two widths x two samples.
    assert_eq!(probe.forward_count, 2 + 8);
    assert_eq!(results[0].layer_index, 3);
    assert_eq!(results[0].kl_divergence_by_bit_width[&bits(4)], 0.2);
    assert_eq!(results[1].kl_divergence_by_bit_width[&bits(8)], 0.01);
    assert_eq!(results[0].cache_bytes_per_token, 2.0 * 4.0 * 64.0 * 2.0);
}

#[test]
fn should_refuse_a_model_without_any_cached_layer() {
    let mut probe = ScriptedCacheProbe {
        cached_layer_indices: Vec::new(),
        kl_by_layer: BTreeMap::new(),
        forward_count: 0,
    };
    let mut silent_reporter = crate::progress::SilentProgressReporter;

    let refusal = measure_kv_cache_sensitivity(
        &mut probe,
        &two_samples(),
        &CandidateBitWidths::new(&[4, 8]).expect("valid ladder"),
        GroupSize::new(64).expect("valid"),
        &mut silent_reporter,
    );

    assert!(matches!(
        refusal,
        Err(crate::error::QuantizationError::ProbeModelFailure { .. })
    ));
}

#[test]
fn should_refuse_a_measurement_where_quantization_never_engaged() {
    let mut probe = ScriptedCacheProbe {
        cached_layer_indices: vec![0],
        kl_by_layer: BTreeMap::from([(0, 1e-9)]),
        forward_count: 0,
    };
    let mut silent_reporter = crate::progress::SilentProgressReporter;

    let refusal = measure_kv_cache_sensitivity(
        &mut probe,
        &two_samples(),
        &CandidateBitWidths::new(&[4, 8]).expect("valid ladder"),
        GroupSize::new(64).expect("valid"),
        &mut silent_reporter,
    );

    assert!(matches!(
        refusal,
        Err(crate::error::QuantizationError::ProbeModelFailure { .. })
    ));
}
