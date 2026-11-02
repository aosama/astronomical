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

/// Greedy allocation under an average cache bits target, OptiQ's
/// `optimize_kv_cache`: every layer starts at the lowest width, and the
/// upgrade with the largest KL reduction per added bit is taken until the
/// plain mean of the per-layer widths meets the target. The target is a
/// floor, so the discrete ladder may overshoot it. The mean is unweighted
/// because every cached layer of one model holds the same heads and head
/// dimension, so bytes per token are equal across layers.
pub fn optimize_kv_cache(
    results: &[KvCacheSensitivityResult],
    target_average_bits: f64,
    candidate_bit_widths: &CandidateBitWidths,
    group_size: GroupSize,
) -> Result<Vec<KvCacheLayerConfig>, QuantizationError> {
    let ladder = candidate_bit_widths.as_slice();
    let mut bit_width_by_result = vec![candidate_bit_widths.lowest(); results.len()];
    while average_bits(&bit_width_by_result) < target_average_bits {
        let Some((result_index, next_bit_width)) =
            most_efficient_upgrade(results, &bit_width_by_result, ladder)
        else {
            break;
        };
        bit_width_by_result[result_index] = next_bit_width;
    }
    let mut configs = Vec::with_capacity(results.len());
    for (result, bit_width) in results.iter().zip(bit_width_by_result) {
        configs.push(KvCacheLayerConfig {
            layer_index: result.layer_index,
            layer_name: result.layer_name.clone(),
            bit_width,
            group_size,
        });
    }
    Ok(configs)
}

/// Cache bytes for a context length under a per-layer config, for the
/// memory report: each layer's float16 elements per token at its width,
/// plus a float16 scale and bias per group when it is quantized.
#[must_use]
pub fn estimate_kv_cache_bytes(
    configs: &[KvCacheLayerConfig],
    results: &[KvCacheSensitivityResult],
    context_length_tokens: usize,
) -> u64 {
    let mut result_by_layer_index = std::collections::BTreeMap::new();
    for result in results {
        result_by_layer_index.insert(result.layer_index, result);
    }
    let mut total_bytes = 0.0_f64;
    for config in configs {
        let Some(result) = result_by_layer_index.get(&config.layer_index) else {
            continue;
        };
        let elements_per_token = result.cache_bytes_per_token / FLOAT16_BYTES;
        let mut bytes_per_token = elements_per_token * f64::from(config.bit_width.bits()) / 8.0;
        if !config.bit_width.keeps_bfloat16() {
            let groups_per_token = elements_per_token / f64::from(config.group_size.elements());
            bytes_per_token += groups_per_token * FLOAT16_BYTES * 2.0;
        }
        total_bytes += bytes_per_token * context_length_tokens as f64;
    }
    total_bytes.round() as u64
}

const FLOAT16_BYTES: f64 = 2.0;

fn average_bits(bit_widths: &[BitWidth]) -> f64 {
    if bit_widths.is_empty() {
        return f64::INFINITY;
    }
    let mut bits_sum = 0.0;
    for bit_width in bit_widths {
        bits_sum += f64::from(bit_width.bits());
    }
    bits_sum / bit_widths.len() as f64
}

/// The upgrade with the largest KL reduction per added bit; ties go to the
/// lower layer index, as OptiQ's heap orders them. An upgrade that does not
/// reduce KL is never taken.
fn most_efficient_upgrade(
    results: &[KvCacheSensitivityResult],
    bit_width_by_result: &[BitWidth],
    ladder: &[BitWidth],
) -> Option<(usize, BitWidth)> {
    let mut best_upgrade: Option<(f64, usize, usize, BitWidth)> = None;
    for (result_index, result) in results.iter().enumerate() {
        let Some((efficiency, next_bit_width)) =
            upgrade_efficiency(result, bit_width_by_result[result_index], ladder)
        else {
            continue;
        };
        if is_better_upgrade(best_upgrade, efficiency, result.layer_index) {
            best_upgrade = Some((efficiency, result.layer_index, result_index, next_bit_width));
        }
    }
    let (_, _, result_index, next_bit_width) = best_upgrade?;
    Some((result_index, next_bit_width))
}

/// KL reduction per added bit for one layer's next rung, or `None` at the
/// top of the ladder or when the next rung does not reduce KL.
fn upgrade_efficiency(
    result: &KvCacheSensitivityResult,
    current_bit_width: BitWidth,
    ladder: &[BitWidth],
) -> Option<(f64, BitWidth)> {
    let next_bit_width = next_rung(ladder, current_bit_width)?;
    let kl_at = &result.kl_divergence_by_bit_width;
    let kl_reduction = kl_at.get(&current_bit_width).copied().unwrap_or(0.0)
        - kl_at.get(&next_bit_width).copied().unwrap_or(0.0);
    if kl_reduction <= 0.0 {
        return None;
    }
    let added_bits = f64::from(next_bit_width.bits() - current_bit_width.bits()).max(1.0);
    Some((kl_reduction / added_bits, next_bit_width))
}

fn next_rung(ladder: &[BitWidth], current_bit_width: BitWidth) -> Option<BitWidth> {
    for (ladder_position, bit_width) in ladder.iter().enumerate() {
        if *bit_width == current_bit_width {
            return ladder.get(ladder_position + 1).copied();
        }
    }
    None
}

fn is_better_upgrade(
    best_upgrade: Option<(f64, usize, usize, BitWidth)>,
    efficiency: f64,
    layer_index: usize,
) -> bool {
    match best_upgrade {
        None => true,
        Some((best_efficiency, best_layer_index, _, _)) => {
            efficiency > best_efficiency
                || (efficiency == best_efficiency && layer_index < best_layer_index)
        }
    }
}
