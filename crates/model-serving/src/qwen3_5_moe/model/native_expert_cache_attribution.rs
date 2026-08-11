//! Converts one native cache report into request performance counters.
//!
//! Keeping the whole conversion together lets a report answer one causal chain:
//! assignments became distinct experts, some experts were missing, memory policy
//! selected a ceiling and evicted bytes, then storage loaded the remaining pages.

use crate::{PerformanceAttribution, PerformanceCounter};

pub(super) fn record_native_expert_cache_request(
    performance_attribution: &mut PerformanceAttribution,
    request_report: astronomical_runtime_integration::MlxNativeExpertCacheRequestReport,
) {
    // Without this complete chain, a storage read can look like a poor
    // least-recently-used decision when an unnecessarily low memory ceiling was
    // the actual cause.
    performance_attribution.record_counter(
        PerformanceCounter::NativeExpertCacheHitCount,
        request_report.cache_hit_count(),
    );
    performance_attribution.record_counter(
        PerformanceCounter::NativeExpertCacheMissCount,
        request_report.cache_miss_count(),
    );
    performance_attribution.record_counter(
        PerformanceCounter::NativeExpertCacheSelectedExpertAssignmentCount,
        request_report.selected_expert_assignment_count(),
    );
    performance_attribution.record_counter(
        PerformanceCounter::NativeExpertCacheDistinctRouteExpertCount,
        request_report.distinct_route_expert_count(),
    );
    performance_attribution.record_counter(
        PerformanceCounter::NativeExpertCacheMissingRouteExpertCount,
        request_report.missing_route_expert_count(),
    );
    performance_attribution.record_counter(
        PerformanceCounter::NativeExpertCacheSelectedRoutePayloadByteCount,
        request_report.selected_route_payload_byte_count(),
    );
    performance_attribution.record_counter(
        PerformanceCounter::NativeExpertCacheMissingRoutePayloadByteCount,
        request_report.missing_route_payload_byte_count(),
    );
    performance_attribution.record_counter(
        PerformanceCounter::NativeExpertCacheDiskPageLoadCount,
        request_report.disk_page_load_count(),
    );
    performance_attribution.record_counter(
        PerformanceCounter::NativeExpertCacheDiskBatchLoadCount,
        request_report.disk_batch_load_count(),
    );
    performance_attribution.record_counter(
        PerformanceCounter::NativeExpertCacheSuccessfulSourceReadCount,
        request_report.successful_source_read_count(),
    );
    performance_attribution.record_counter(
        PerformanceCounter::NativeExpertCacheSuccessfulSourceReadByteCount,
        request_report.successful_source_read_byte_count(),
    );
    performance_attribution.record_counter(
        PerformanceCounter::NativeExpertCacheSuccessfulSourceReadElapsedNanoseconds,
        request_report.successful_source_read_elapsed_nanoseconds(),
    );
    performance_attribution.record_counter(
        PerformanceCounter::NativeExpertCacheRouteDependencySynchronizationCount,
        request_report.route_dependency_synchronization_count(),
    );
    performance_attribution.record_counter(
        PerformanceCounter::NativeExpertCacheRouteDependencySynchronizationElapsedNanoseconds,
        request_report.route_dependency_synchronization_elapsed_nanoseconds(),
    );
    performance_attribution.record_maximum_counter(
        PerformanceCounter::NativeExpertCacheMaximumRouteDependencySynchronizationElapsedNanoseconds,
        request_report.maximum_route_dependency_synchronization_elapsed_nanoseconds(),
    );
    performance_attribution.record_counter(
        PerformanceCounter::NativeExpertCacheSnapshotPublicationCount,
        request_report.page_table_publication_count(),
    );
    performance_attribution.record_counter(
        PerformanceCounter::NativeExpertCachePayloadCopyByteCount,
        request_report.payload_copy_byte_count(),
    );
    performance_attribution.record_counter(
        PerformanceCounter::NativeExpertCacheEvictedPayloadByteCount,
        request_report.evicted_payload_byte_count(),
    );
    performance_attribution.record_maximum_counter(
        PerformanceCounter::NativeExpertCacheMaximumRetentionCeilingBeforeByteCount,
        request_report.retention_ceiling_before_byte_count(),
    );
    performance_attribution.record_maximum_counter(
        PerformanceCounter::NativeExpertCacheMaximumRetentionCeilingAfterByteCount,
        request_report.retention_ceiling_after_byte_count(),
    );
    performance_attribution.record_counter(
        PerformanceCounter::CompleteLayerRouteSynchronizationElisionCount,
        request_report.complete_layer_route_synchronization_elision_count(),
    );
}
