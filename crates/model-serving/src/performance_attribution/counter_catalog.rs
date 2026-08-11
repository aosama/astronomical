//! Stable scalar evidence emitted beside critical-path timing spans.
//!
//! Enum discriminants index a fixed array on the hot path. `COUNT`, `ALL`, and
//! `identifier` must therefore change together: their shared order is the
//! zero-allocation bridge from recording to serialized diagnostics.

/// One bounded numerical counter attached to a performance-attribution report.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum PerformanceCounter {
    PromptTokenCount,
    RestoredPersistentPromptCacheTokenCount,
    GeneratedTokenCount,
    PrefillChunckCount,
    PrefillCapacityRejectionCount,
    PrefillCapacityRetryCount,
    NativeExpertCacheHitCount,
    NativeExpertCacheMissCount,
    NativeExpertCacheSelectedExpertAssignmentCount,
    NativeExpertCacheDistinctRouteExpertCount,
    NativeExpertCacheMissingRouteExpertCount,
    NativeExpertCacheSelectedRoutePayloadByteCount,
    NativeExpertCacheMissingRoutePayloadByteCount,
    NativeExpertCacheEvictionCount,
    NativeExpertCacheEvictedPayloadByteCount,
    NativeExpertCacheMaximumRetentionCeilingBeforeByteCount,
    NativeExpertCacheMaximumRetentionCeilingAfterByteCount,
    NativeExpertCacheDiskPageLoadCount,
    NativeExpertCacheDiskBatchLoadCount,
    NativeExpertCacheSuccessfulSourceReadCount,
    NativeExpertCacheSuccessfulSourceReadByteCount,
    NativeExpertCacheSuccessfulSourceReadElapsedNanoseconds,
    NativeExpertCacheRouteDependencySynchronizationCount,
    NativeExpertCacheRouteDependencySynchronizationElapsedNanoseconds,
    NativeExpertCacheMaximumRouteDependencySynchronizationElapsedNanoseconds,
    NativeExpertCacheSnapshotPublicationCount,
    NativeExpertCachePayloadCopyByteCount,
    NativePagedExpertProjectionGraphCount,
    CompleteLayerRouteSynchronizationElisionCount,
    PositionalFileReadCallCount,
    PositionalFileReadByteCount,
    PositionalFileReadElapsedNanoseconds,
    PositionalFileReadMaximumElapsedNanoseconds,
    PositionalFileReadMaximumConcurrentCount,
    PositionalFileReadFailureCount,
    ExpertRoutePredictedExpertCount,
    ExpertRouteMatchedExpertCount,
    ExpertRouteCompletelyMatchedLayerCount,
    ExpertRouteExaminedLayerCount,
    MtpMemoryAdmissionFallbackCount,
    MtpAdmittedAttemptCount,
    SpeculativePrefillTargetOnlyPrefixChunckCount,
    SpeculativePrefillTargetOnlyPrefixTokenCount,
    SpeculativePrefillTerminalCaptureChunckCount,
    SpeculativePrefillTerminalMtpHistoryTokenCount,
    SpeculativePrefillDraftScoringCount,
    SpeculativePrefillDraftPrefixStoreHitCount,
    SpeculativePrefillDraftPrefixStoreWriteCount,
    SpeculativePrefillDraftPersistentPrefixHitCount,
    SpeculativePrefillDraftPersistentPrefixRestoredTokenCount,
    SpeculativePrefillContextTargetExpertReclaimedPayloadBytes,
    SpeculativePrefillDraftTargetExpertReclaimedPayloadBytes,
    SpeculativePrefillSelectionStoreHitCount,
    SpeculativePrefillSelectionPersistentHitCount,
    SpeculativePrefillMandatoryVisualTokenCount,
    SpeculativePrefillSelectedTokenCount,
    SpeculativePrefillSparseTargetChunckCount,
    SpeculativePrefillDraftScoredSuffixTokenCount,
    SpeculativePrefillTargetPersistentStateWriteCount,
    SpeculativePrefillTargetPersistentStateRestoredTokenCount,
    SpeculativePrefillTargetExpertRepopulatedPayloadBytes,
    SpeculativePrefillOrdinaryControlSpanTokenCount,
    SpeculativePrefillFallbackCount,
    MtpPromptHistoryInitializationFallbackCount,
    MtpFeedbackHistoryReseedCount,
    MtpAcceptedDraftCount,
    MtpRejectedDraftCount,
    MtpOperationalFallbackCount,
}

impl PerformanceCounter {
    // The final discriminant makes enabled storage exact while disabled
    // attribution remains pointer-sized and performs no counter allocation.
    pub(super) const COUNT: usize = Self::MtpOperationalFallbackCount as usize + 1;
    pub(super) const ALL: [Self; Self::COUNT] = [
        Self::PromptTokenCount,
        Self::RestoredPersistentPromptCacheTokenCount,
        Self::GeneratedTokenCount,
        Self::PrefillChunckCount,
        Self::PrefillCapacityRejectionCount,
        Self::PrefillCapacityRetryCount,
        Self::NativeExpertCacheHitCount,
        Self::NativeExpertCacheMissCount,
        Self::NativeExpertCacheSelectedExpertAssignmentCount,
        Self::NativeExpertCacheDistinctRouteExpertCount,
        Self::NativeExpertCacheMissingRouteExpertCount,
        Self::NativeExpertCacheSelectedRoutePayloadByteCount,
        Self::NativeExpertCacheMissingRoutePayloadByteCount,
        Self::NativeExpertCacheEvictionCount,
        Self::NativeExpertCacheEvictedPayloadByteCount,
        Self::NativeExpertCacheMaximumRetentionCeilingBeforeByteCount,
        Self::NativeExpertCacheMaximumRetentionCeilingAfterByteCount,
        Self::NativeExpertCacheDiskPageLoadCount,
        Self::NativeExpertCacheDiskBatchLoadCount,
        Self::NativeExpertCacheSuccessfulSourceReadCount,
        Self::NativeExpertCacheSuccessfulSourceReadByteCount,
        Self::NativeExpertCacheSuccessfulSourceReadElapsedNanoseconds,
        Self::NativeExpertCacheRouteDependencySynchronizationCount,
        Self::NativeExpertCacheRouteDependencySynchronizationElapsedNanoseconds,
        Self::NativeExpertCacheMaximumRouteDependencySynchronizationElapsedNanoseconds,
        Self::NativeExpertCacheSnapshotPublicationCount,
        Self::NativeExpertCachePayloadCopyByteCount,
        Self::NativePagedExpertProjectionGraphCount,
        Self::CompleteLayerRouteSynchronizationElisionCount,
        Self::PositionalFileReadCallCount,
        Self::PositionalFileReadByteCount,
        Self::PositionalFileReadElapsedNanoseconds,
        Self::PositionalFileReadMaximumElapsedNanoseconds,
        Self::PositionalFileReadMaximumConcurrentCount,
        Self::PositionalFileReadFailureCount,
        Self::ExpertRoutePredictedExpertCount,
        Self::ExpertRouteMatchedExpertCount,
        Self::ExpertRouteCompletelyMatchedLayerCount,
        Self::ExpertRouteExaminedLayerCount,
        Self::MtpMemoryAdmissionFallbackCount,
        Self::MtpAdmittedAttemptCount,
        Self::SpeculativePrefillTargetOnlyPrefixChunckCount,
        Self::SpeculativePrefillTargetOnlyPrefixTokenCount,
        Self::SpeculativePrefillTerminalCaptureChunckCount,
        Self::SpeculativePrefillTerminalMtpHistoryTokenCount,
        Self::SpeculativePrefillDraftScoringCount,
        Self::SpeculativePrefillDraftPrefixStoreHitCount,
        Self::SpeculativePrefillDraftPrefixStoreWriteCount,
        Self::SpeculativePrefillDraftPersistentPrefixHitCount,
        Self::SpeculativePrefillDraftPersistentPrefixRestoredTokenCount,
        Self::SpeculativePrefillContextTargetExpertReclaimedPayloadBytes,
        Self::SpeculativePrefillDraftTargetExpertReclaimedPayloadBytes,
        Self::SpeculativePrefillSelectionStoreHitCount,
        Self::SpeculativePrefillSelectionPersistentHitCount,
        Self::SpeculativePrefillMandatoryVisualTokenCount,
        Self::SpeculativePrefillSelectedTokenCount,
        Self::SpeculativePrefillSparseTargetChunckCount,
        Self::SpeculativePrefillDraftScoredSuffixTokenCount,
        Self::SpeculativePrefillTargetPersistentStateWriteCount,
        Self::SpeculativePrefillTargetPersistentStateRestoredTokenCount,
        Self::SpeculativePrefillTargetExpertRepopulatedPayloadBytes,
        Self::SpeculativePrefillOrdinaryControlSpanTokenCount,
        Self::SpeculativePrefillFallbackCount,
        Self::MtpPromptHistoryInitializationFallbackCount,
        Self::MtpFeedbackHistoryReseedCount,
        Self::MtpAcceptedDraftCount,
        Self::MtpRejectedDraftCount,
        Self::MtpOperationalFallbackCount,
    ];

    pub(super) const fn identifier(self) -> &'static str {
        match self {
            Self::PromptTokenCount => "prompt_token_count",
            Self::RestoredPersistentPromptCacheTokenCount => {
                "restored_persistent_prompt_cache_token_count"
            }
            Self::GeneratedTokenCount => "generated_token_count",
            Self::PrefillChunckCount => "prefill_chunck_count",
            Self::PrefillCapacityRejectionCount => "prefill_capacity_rejection_count",
            Self::PrefillCapacityRetryCount => "prefill_capacity_retry_count",
            Self::NativeExpertCacheHitCount => "native_expert_cache_hit_count",
            Self::NativeExpertCacheMissCount => "native_expert_cache_miss_count",
            Self::NativeExpertCacheSelectedExpertAssignmentCount => {
                "native_expert_cache_selected_expert_assignment_count"
            }
            Self::NativeExpertCacheDistinctRouteExpertCount => {
                "native_expert_cache_distinct_route_expert_count"
            }
            Self::NativeExpertCacheMissingRouteExpertCount => {
                "native_expert_cache_missing_route_expert_count"
            }
            Self::NativeExpertCacheSelectedRoutePayloadByteCount => {
                "native_expert_cache_selected_route_payload_byte_count"
            }
            Self::NativeExpertCacheMissingRoutePayloadByteCount => {
                "native_expert_cache_missing_route_payload_byte_count"
            }
            Self::NativeExpertCacheEvictionCount => "native_expert_cache_eviction_count",
            Self::NativeExpertCacheEvictedPayloadByteCount => {
                "native_expert_cache_evicted_payload_byte_count"
            }
            Self::NativeExpertCacheMaximumRetentionCeilingBeforeByteCount => {
                "native_expert_cache_maximum_retention_ceiling_before_byte_count"
            }
            Self::NativeExpertCacheMaximumRetentionCeilingAfterByteCount => {
                "native_expert_cache_maximum_retention_ceiling_after_byte_count"
            }
            Self::NativeExpertCacheDiskPageLoadCount => "native_expert_cache_disk_page_load_count",
            Self::NativeExpertCacheDiskBatchLoadCount => {
                "native_expert_cache_disk_batch_load_count"
            }
            Self::NativeExpertCacheSuccessfulSourceReadCount => {
                "native_expert_cache_successful_source_read_count"
            }
            Self::NativeExpertCacheSuccessfulSourceReadByteCount => {
                "native_expert_cache_successful_source_read_byte_count"
            }
            Self::NativeExpertCacheSuccessfulSourceReadElapsedNanoseconds => {
                "native_expert_cache_successful_source_read_elapsed_nanoseconds"
            }
            Self::NativeExpertCacheRouteDependencySynchronizationCount => {
                "native_expert_cache_route_dependency_synchronization_count"
            }
            Self::NativeExpertCacheRouteDependencySynchronizationElapsedNanoseconds => {
                "native_expert_cache_route_dependency_synchronization_elapsed_nanoseconds"
            }
            Self::NativeExpertCacheMaximumRouteDependencySynchronizationElapsedNanoseconds => {
                "native_expert_cache_maximum_route_dependency_synchronization_elapsed_nanoseconds"
            }
            Self::NativeExpertCacheSnapshotPublicationCount => {
                "native_expert_cache_snapshot_publication_count"
            }
            Self::NativeExpertCachePayloadCopyByteCount => {
                "native_expert_cache_payload_copy_byte_count"
            }
            Self::NativePagedExpertProjectionGraphCount => {
                "native_paged_expert_projection_graph_count"
            }
            Self::CompleteLayerRouteSynchronizationElisionCount => {
                "complete_layer_route_synchronization_elision_count"
            }
            Self::PositionalFileReadCallCount => "positional_file_read_call_count",
            Self::PositionalFileReadByteCount => "positional_file_read_byte_count",
            Self::PositionalFileReadElapsedNanoseconds => {
                "positional_file_read_elapsed_nanoseconds"
            }
            Self::PositionalFileReadMaximumElapsedNanoseconds => {
                "positional_file_read_maximum_elapsed_nanoseconds"
            }
            Self::PositionalFileReadMaximumConcurrentCount => {
                "positional_file_read_maximum_concurrent_count"
            }
            Self::PositionalFileReadFailureCount => "positional_file_read_failure_count",
            Self::ExpertRoutePredictedExpertCount => "expert_route_predicted_expert_count",
            Self::ExpertRouteMatchedExpertCount => "expert_route_matched_expert_count",
            Self::ExpertRouteCompletelyMatchedLayerCount => {
                "expert_route_completely_matched_layer_count"
            }
            Self::ExpertRouteExaminedLayerCount => "expert_route_examined_layer_count",
            Self::MtpMemoryAdmissionFallbackCount => "mtp_memory_admission_fallback_count",
            Self::MtpAdmittedAttemptCount => "mtp_admitted_attempt_count",
            Self::SpeculativePrefillTargetOnlyPrefixChunckCount => {
                "speculative_prefill_target_only_prefix_chunck_count"
            }
            Self::SpeculativePrefillTargetOnlyPrefixTokenCount => {
                "speculative_prefill_target_only_prefix_token_count"
            }
            Self::SpeculativePrefillTerminalCaptureChunckCount => {
                "speculative_prefill_terminal_capture_chunck_count"
            }
            Self::SpeculativePrefillTerminalMtpHistoryTokenCount => {
                "speculative_prefill_terminal_mtp_history_token_count"
            }
            Self::SpeculativePrefillDraftScoringCount => "speculative_prefill_draft_scoring_count",
            Self::SpeculativePrefillDraftPrefixStoreHitCount => {
                "speculative_prefill_draft_prefix_store_hit_count"
            }
            Self::SpeculativePrefillDraftPrefixStoreWriteCount => {
                "speculative_prefill_draft_prefix_store_write_count"
            }
            Self::SpeculativePrefillDraftPersistentPrefixHitCount => {
                "speculative_prefill_draft_persistent_prefix_hit_count"
            }
            Self::SpeculativePrefillDraftPersistentPrefixRestoredTokenCount => {
                "speculative_prefill_draft_persistent_prefix_restored_token_count"
            }
            Self::SpeculativePrefillContextTargetExpertReclaimedPayloadBytes => {
                "speculative_prefill_context_target_expert_reclaimed_payload_bytes"
            }
            Self::SpeculativePrefillDraftTargetExpertReclaimedPayloadBytes => {
                "speculative_prefill_draft_target_expert_reclaimed_payload_bytes"
            }
            Self::SpeculativePrefillSelectionStoreHitCount => {
                "speculative_prefill_selection_store_hit_count"
            }
            Self::SpeculativePrefillSelectionPersistentHitCount => {
                "speculative_prefill_selection_persistent_hit_count"
            }
            Self::SpeculativePrefillMandatoryVisualTokenCount => {
                "speculative_prefill_mandatory_visual_token_count"
            }
            Self::SpeculativePrefillSelectedTokenCount => {
                "speculative_prefill_selected_token_count"
            }
            Self::SpeculativePrefillSparseTargetChunckCount => {
                "speculative_prefill_sparse_target_chunck_count"
            }
            Self::SpeculativePrefillDraftScoredSuffixTokenCount => {
                "speculative_prefill_draft_scored_suffix_token_count"
            }
            Self::SpeculativePrefillTargetPersistentStateWriteCount => {
                "speculative_prefill_target_persistent_state_write_count"
            }
            Self::SpeculativePrefillTargetPersistentStateRestoredTokenCount => {
                "speculative_prefill_target_persistent_state_restored_token_count"
            }
            Self::SpeculativePrefillTargetExpertRepopulatedPayloadBytes => {
                "speculative_prefill_target_expert_repopulated_payload_bytes"
            }
            Self::SpeculativePrefillOrdinaryControlSpanTokenCount => {
                "speculative_prefill_ordinary_control_span_token_count"
            }
            Self::SpeculativePrefillFallbackCount => "speculative_prefill_fallback_count",
            Self::MtpPromptHistoryInitializationFallbackCount => {
                "mtp_prompt_history_initialization_fallback_count"
            }
            Self::MtpFeedbackHistoryReseedCount => "mtp_feedback_history_reseed_count",
            Self::MtpAcceptedDraftCount => "mtp_accepted_draft_count",
            Self::MtpRejectedDraftCount => "mtp_rejected_draft_count",
            Self::MtpOperationalFallbackCount => "mtp_operational_fallback_count",
        }
    }
}
