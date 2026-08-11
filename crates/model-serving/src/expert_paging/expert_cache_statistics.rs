//! Cache accounting DTOs and exact retained-weight payload sizing.

/// Cumulative cache counters for transparent low-level performance tests.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExpertWeightMemoryCacheStatistics {
    pub entry_count: usize,
    pub resident_payload_byte_count: u64,
    pub maximum_resident_payload_byte_count: u64,
    pub eviction_count: u64,
    pub cache_hit_count: u64,
    pub cache_miss_count: u64,
    pub disk_page_load_count: u64,
    pub disk_batch_load_count: u64,
}
