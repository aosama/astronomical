//! Owned Rust copies of native expert-cache diagnostics.
//!
//! The C++ cache mutates caller-provided plain scalar structures during one
//! synchronous operation. Converting them immediately keeps no native pointer
//! alive and separates request-local evidence from process-lifetime totals.

use crate::raw;

/// Plain numerical evidence for one route from analysis through commit.
///
/// Assignment count can be much larger than distinct count because many tokens
/// select the same expert. Missing count is the subset that needs disk-backed
/// page allocation. Comparing those values explains why exact admission avoids
/// the former worst-case reservation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MlxNativeExpertCacheRequestReport {
    cache_hit_count: u64,
    cache_miss_count: u64,
    disk_page_load_count: u64,
    disk_batch_load_count: u64,
    successful_source_read_count: u64,
    successful_source_read_byte_count: u64,
    successful_source_read_elapsed_nanoseconds: u64,
    route_dependency_synchronization_count: u64,
    route_dependency_synchronization_elapsed_nanoseconds: u64,
    maximum_route_dependency_synchronization_elapsed_nanoseconds: u64,
    payload_copy_byte_count: u64,
    page_table_publication_count: u64,
    complete_layer_route_synchronization_elision_count: u64,
    selected_expert_assignment_count: u64,
    distinct_route_expert_count: u64,
    missing_route_expert_count: u64,
    selected_route_payload_byte_count: u64,
    missing_route_payload_byte_count: u64,
    evicted_payload_byte_count: u64,
    retention_ceiling_before_byte_count: u64,
    retention_ceiling_after_byte_count: u64,
}

impl MlxNativeExpertCacheRequestReport {
    #[must_use]
    pub const fn cache_hit_count(self) -> u64 {
        self.cache_hit_count
    }

    #[must_use]
    pub const fn cache_miss_count(self) -> u64 {
        self.cache_miss_count
    }

    #[must_use]
    pub const fn disk_page_load_count(self) -> u64 {
        self.disk_page_load_count
    }

    #[must_use]
    pub const fn disk_batch_load_count(self) -> u64 {
        self.disk_batch_load_count
    }

    #[must_use]
    pub const fn successful_source_read_count(self) -> u64 {
        self.successful_source_read_count
    }

    #[must_use]
    pub const fn successful_source_read_byte_count(self) -> u64 {
        self.successful_source_read_byte_count
    }

    #[must_use]
    pub const fn successful_source_read_elapsed_nanoseconds(self) -> u64 {
        self.successful_source_read_elapsed_nanoseconds
    }

    #[must_use]
    pub const fn route_dependency_synchronization_count(self) -> u64 {
        self.route_dependency_synchronization_count
    }

    #[must_use]
    pub const fn route_dependency_synchronization_elapsed_nanoseconds(self) -> u64 {
        self.route_dependency_synchronization_elapsed_nanoseconds
    }

    #[must_use]
    pub const fn maximum_route_dependency_synchronization_elapsed_nanoseconds(self) -> u64 {
        self.maximum_route_dependency_synchronization_elapsed_nanoseconds
    }

    #[must_use]
    pub const fn payload_copy_byte_count(self) -> u64 {
        self.payload_copy_byte_count
    }

    #[must_use]
    pub const fn page_table_publication_count(self) -> u64 {
        self.page_table_publication_count
    }

    #[must_use]
    pub const fn complete_layer_route_synchronization_elision_count(self) -> u64 {
        self.complete_layer_route_synchronization_elision_count
    }

    #[must_use]
    pub const fn distinct_route_expert_count(self) -> u64 {
        self.distinct_route_expert_count
    }

    #[must_use]
    pub const fn selected_expert_assignment_count(self) -> u64 {
        self.selected_expert_assignment_count
    }

    #[must_use]
    pub const fn missing_route_expert_count(self) -> u64 {
        self.missing_route_expert_count
    }

    #[must_use]
    pub const fn selected_route_payload_byte_count(self) -> u64 {
        self.selected_route_payload_byte_count
    }

    #[must_use]
    pub const fn missing_route_payload_byte_count(self) -> u64 {
        self.missing_route_payload_byte_count
    }

    #[must_use]
    pub const fn evicted_payload_byte_count(self) -> u64 {
        self.evicted_payload_byte_count
    }

    #[must_use]
    pub const fn retention_ceiling_before_byte_count(self) -> u64 {
        self.retention_ceiling_before_byte_count
    }

    #[must_use]
    pub const fn retention_ceiling_after_byte_count(self) -> u64 {
        self.retention_ceiling_after_byte_count
    }
}

/// Process-lifetime cache policy totals plus the current residency snapshot.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MlxNativeExpertCacheStatistics {
    resident_expert_count: u64,
    resident_payload_byte_count: u64,
    maximum_resident_payload_byte_count: u64,
    eviction_count: u64,
    cache_hit_count: u64,
    cache_miss_count: u64,
    disk_page_load_count: u64,
    disk_batch_load_count: u64,
}

impl MlxNativeExpertCacheStatistics {
    #[must_use]
    pub const fn resident_expert_count(self) -> u64 {
        self.resident_expert_count
    }

    #[must_use]
    pub const fn resident_payload_byte_count(self) -> u64 {
        self.resident_payload_byte_count
    }

    #[must_use]
    pub const fn maximum_resident_payload_byte_count(self) -> u64 {
        self.maximum_resident_payload_byte_count
    }

    #[must_use]
    pub const fn eviction_count(self) -> u64 {
        self.eviction_count
    }

    #[must_use]
    pub const fn cache_hit_count(self) -> u64 {
        self.cache_hit_count
    }

    #[must_use]
    pub const fn cache_miss_count(self) -> u64 {
        self.cache_miss_count
    }

    #[must_use]
    pub const fn disk_page_load_count(self) -> u64 {
        self.disk_page_load_count
    }

    #[must_use]
    pub const fn disk_batch_load_count(self) -> u64 {
        self.disk_batch_load_count
    }
}

pub(crate) const fn zero_raw_request_report() -> raw::astronomical_native_expert_cache_request_report
{
    raw::astronomical_native_expert_cache_request_report {
        cache_hit_count: 0,
        cache_miss_count: 0,
        disk_page_load_count: 0,
        disk_batch_load_count: 0,
        successful_source_read_count: 0,
        successful_source_read_byte_count: 0,
        successful_source_read_elapsed_nanoseconds: 0,
        route_dependency_synchronization_count: 0,
        route_dependency_synchronization_elapsed_nanoseconds: 0,
        maximum_route_dependency_synchronization_elapsed_nanoseconds: 0,
        payload_copy_byte_count: 0,
        page_table_publication_count: 0,
        complete_layer_route_synchronization_elision_count: 0,
        selected_expert_assignment_count: 0,
        distinct_route_expert_count: 0,
        missing_route_expert_count: 0,
        selected_route_payload_byte_count: 0,
        missing_route_payload_byte_count: 0,
        evicted_payload_byte_count: 0,
        retention_ceiling_before_byte_count: 0,
        retention_ceiling_after_byte_count: 0,
    }
}

pub(crate) const fn request_report_from_raw(
    report: raw::astronomical_native_expert_cache_request_report,
) -> MlxNativeExpertCacheRequestReport {
    MlxNativeExpertCacheRequestReport {
        cache_hit_count: report.cache_hit_count,
        cache_miss_count: report.cache_miss_count,
        disk_page_load_count: report.disk_page_load_count,
        disk_batch_load_count: report.disk_batch_load_count,
        successful_source_read_count: report.successful_source_read_count,
        successful_source_read_byte_count: report.successful_source_read_byte_count,
        successful_source_read_elapsed_nanoseconds: report
            .successful_source_read_elapsed_nanoseconds,
        route_dependency_synchronization_count: report.route_dependency_synchronization_count,
        route_dependency_synchronization_elapsed_nanoseconds: report
            .route_dependency_synchronization_elapsed_nanoseconds,
        maximum_route_dependency_synchronization_elapsed_nanoseconds: report
            .maximum_route_dependency_synchronization_elapsed_nanoseconds,
        payload_copy_byte_count: report.payload_copy_byte_count,
        page_table_publication_count: report.page_table_publication_count,
        complete_layer_route_synchronization_elision_count: report
            .complete_layer_route_synchronization_elision_count,
        selected_expert_assignment_count: report.selected_expert_assignment_count,
        distinct_route_expert_count: report.distinct_route_expert_count,
        missing_route_expert_count: report.missing_route_expert_count,
        selected_route_payload_byte_count: report.selected_route_payload_byte_count,
        missing_route_payload_byte_count: report.missing_route_payload_byte_count,
        evicted_payload_byte_count: report.evicted_payload_byte_count,
        retention_ceiling_before_byte_count: report.retention_ceiling_before_byte_count,
        retention_ceiling_after_byte_count: report.retention_ceiling_after_byte_count,
    }
}

pub(crate) const fn statistics_from_raw(
    statistics: raw::astronomical_native_expert_cache_statistics,
) -> MlxNativeExpertCacheStatistics {
    MlxNativeExpertCacheStatistics {
        resident_expert_count: statistics.resident_expert_count,
        resident_payload_byte_count: statistics.resident_payload_byte_count,
        maximum_resident_payload_byte_count: statistics.maximum_resident_payload_byte_count,
        eviction_count: statistics.eviction_count,
        cache_hit_count: statistics.cache_hit_count,
        cache_miss_count: statistics.cache_miss_count,
        disk_page_load_count: statistics.disk_page_load_count,
        disk_batch_load_count: statistics.disk_batch_load_count,
    }
}
