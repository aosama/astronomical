//! Cache accounting DTOs and exact retained-weight payload sizing.

use super::ExpertWeightPage;

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

pub(crate) fn paged_expert_payload_byte_count<ExpertPage: ExpertWeightPage>(
    paged_expert_weights: &ExpertPage,
) -> u64 {
    paged_expert_weights.resident_payload_byte_count()
}
