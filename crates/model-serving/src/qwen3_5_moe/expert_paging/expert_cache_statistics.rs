//! Cache accounting DTOs and exact retained-weight payload sizing.

use super::super::model::decoder_layer_weights::Qwen3_5MoEAffineWeights;
use super::expert_pager::PagedExpertWeights;

/// One point-in-time report for a cache-assisted expert paging request.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExpertWeightMemoryCacheRequestReport {
    pub cache_hit_count: usize,
    pub cache_miss_count: usize,
    pub disk_page_load_count: usize,
    pub disk_batch_load_count: usize,
}

/// Cumulative cache counters for transparent low-level performance tests.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExpertWeightMemoryCacheStatistics {
    pub entry_count: usize,
    pub complete_layer_count: usize,
    pub resident_payload_byte_count: u64,
    pub maximum_resident_payload_byte_count: u64,
    pub eviction_count: u64,
    pub cache_hit_count: u64,
    pub complete_layer_hit_count: u64,
    pub cache_miss_count: u64,
    pub disk_page_load_count: u64,
    pub disk_batch_load_count: u64,
}

pub(crate) fn paged_expert_payload_byte_count(paged_expert_weights: &PagedExpertWeights) -> u64 {
    let payload_byte_count = affine_payload_byte_count(&paged_expert_weights.gate_projection)
        + affine_payload_byte_count(&paged_expert_weights.up_projection)
        + affine_payload_byte_count(&paged_expert_weights.down_projection);
    payload_byte_count as u64
}

fn affine_payload_byte_count(affine_weights: &Qwen3_5MoEAffineWeights) -> usize {
    match affine_weights {
        Qwen3_5MoEAffineWeights::NativeBfloat16 { weight } => weight.byte_count(),
        Qwen3_5MoEAffineWeights::Quantized {
            packed_weight,
            quantization_scales,
            quantization_biases,
            ..
        } => {
            packed_weight.byte_count()
                + quantization_scales.byte_count()
                + quantization_biases.byte_count()
        }
    }
}
