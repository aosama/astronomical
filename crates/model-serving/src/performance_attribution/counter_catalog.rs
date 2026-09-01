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
    ForcedThinkingTransitionTokenCount,
    PrefillChunkCount,
    PrefillCapacityRejectionCount,
    PrefillCapacityRetryCount,
    RustExpertStreamingPayloadByteCount,
    RustStreamedExpertProjectionGraphCount,
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
    ExpertResidencyPlanCompleteLayerCount,
    ExpertResidencyPlanPartialLayerCount,
    ExpertResidencyPlanStreamedLayerCount,
    ExpertResidencyPreexistingCompletePayloadBytes,
    ExpertResidencyPreexistingPartialPayloadBytes,
    ExpertResidencyPreservedCompletePayloadBytes,
    ExpertResidencyPreservedPartialPayloadBytes,
    ExpertResidencyPromotedCompletePayloadBytes,
    ExpertResidencyPromotedPartialPayloadBytes,
    ExpertResidencyRetiredCompletePayloadBytes,
    ExpertResidencyRetiredPartialPayloadBytes,
    ExpertTopologyPreservedPayloadBytes,
    ExpertTopologyRetiredPayloadBytes,
    MandatoryPrefillExpertSourcePayloadBytes,
    MandatoryDecodeExpertSourcePayloadBytes,
    AvoidedCompleteLayerExpertSourcePayloadBytes,
    CompleteLayerPrefetchUsefulPayloadBytes,
    CompleteLayerPrefetchWastedPayloadBytes,
    RetainedRouteAssignmentHitCount,
    RetainedRouteAssignmentMissCount,
    ExpertResidencyCommitRejectionCount,
    MtpDepthSelectionFallbackCount,
    MtpAdmittedAttemptCount,
    SpeculativePrefillTargetOnlyPrefixChunkCount,
    SpeculativePrefillTargetOnlyPrefixTokenCount,
    SpeculativePrefillTerminalCaptureChunkCount,
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
    SpeculativePrefillSparseTargetChunkCount,
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
    MtpRequestedDepthTotal,
    MtpEffectiveDepthTotal,
    MtpProposedDraftCount,
    MtpEosTruncatedPrefixCount,
    MtpOutputDepthDowngradeCount,
    MtpContextDepthDowngradeCount,
    MtpThinkingDepthDowngradeCount,
    MtpMemoryDepthDowngradeCount,
    MtpVerificationWorkspaceByteCount,
    MtpBoundarySnapshotByteCount,
    MtpPersistentGrowthByteCount,
    MtpTargetRepairCount,
    MtpPredictorReplayTokenCount,
    MtpQueuedFrontierRestorationCount,
    MtpCancellationWithQueuedStateCount,
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
        Self::ForcedThinkingTransitionTokenCount,
        Self::PrefillChunkCount,
        Self::PrefillCapacityRejectionCount,
        Self::PrefillCapacityRetryCount,
        Self::RustExpertStreamingPayloadByteCount,
        Self::RustStreamedExpertProjectionGraphCount,
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
        Self::ExpertResidencyPlanCompleteLayerCount,
        Self::ExpertResidencyPlanPartialLayerCount,
        Self::ExpertResidencyPlanStreamedLayerCount,
        Self::ExpertResidencyPreexistingCompletePayloadBytes,
        Self::ExpertResidencyPreexistingPartialPayloadBytes,
        Self::ExpertResidencyPreservedCompletePayloadBytes,
        Self::ExpertResidencyPreservedPartialPayloadBytes,
        Self::ExpertResidencyPromotedCompletePayloadBytes,
        Self::ExpertResidencyPromotedPartialPayloadBytes,
        Self::ExpertResidencyRetiredCompletePayloadBytes,
        Self::ExpertResidencyRetiredPartialPayloadBytes,
        Self::ExpertTopologyPreservedPayloadBytes,
        Self::ExpertTopologyRetiredPayloadBytes,
        Self::MandatoryPrefillExpertSourcePayloadBytes,
        Self::MandatoryDecodeExpertSourcePayloadBytes,
        Self::AvoidedCompleteLayerExpertSourcePayloadBytes,
        Self::CompleteLayerPrefetchUsefulPayloadBytes,
        Self::CompleteLayerPrefetchWastedPayloadBytes,
        Self::RetainedRouteAssignmentHitCount,
        Self::RetainedRouteAssignmentMissCount,
        Self::ExpertResidencyCommitRejectionCount,
        Self::MtpDepthSelectionFallbackCount,
        Self::MtpAdmittedAttemptCount,
        Self::SpeculativePrefillTargetOnlyPrefixChunkCount,
        Self::SpeculativePrefillTargetOnlyPrefixTokenCount,
        Self::SpeculativePrefillTerminalCaptureChunkCount,
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
        Self::SpeculativePrefillSparseTargetChunkCount,
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
        Self::MtpRequestedDepthTotal,
        Self::MtpEffectiveDepthTotal,
        Self::MtpProposedDraftCount,
        Self::MtpEosTruncatedPrefixCount,
        Self::MtpOutputDepthDowngradeCount,
        Self::MtpContextDepthDowngradeCount,
        Self::MtpThinkingDepthDowngradeCount,
        Self::MtpMemoryDepthDowngradeCount,
        Self::MtpVerificationWorkspaceByteCount,
        Self::MtpBoundarySnapshotByteCount,
        Self::MtpPersistentGrowthByteCount,
        Self::MtpTargetRepairCount,
        Self::MtpPredictorReplayTokenCount,
        Self::MtpQueuedFrontierRestorationCount,
        Self::MtpCancellationWithQueuedStateCount,
        Self::MtpOperationalFallbackCount,
    ];

    pub(super) const fn identifier(self) -> &'static str {
        match self {
            Self::PromptTokenCount => "prompt_token_count",
            Self::RestoredPersistentPromptCacheTokenCount => {
                "restored_persistent_prompt_cache_token_count"
            }
            Self::GeneratedTokenCount => "generated_token_count",
            Self::ForcedThinkingTransitionTokenCount => "forced_thinking_transition_token_count",
            Self::PrefillChunkCount => "prefill_chunk_count",
            Self::PrefillCapacityRejectionCount => "prefill_capacity_rejection_count",
            Self::PrefillCapacityRetryCount => "prefill_capacity_retry_count",
            Self::RustExpertStreamingPayloadByteCount => "rust_expert_streaming_payload_byte_count",
            Self::RustStreamedExpertProjectionGraphCount => {
                "rust_streamed_expert_projection_graph_count"
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
            Self::ExpertResidencyPlanCompleteLayerCount => {
                "expert_residency_plan_complete_layer_count"
            }
            Self::ExpertResidencyPlanPartialLayerCount => {
                "expert_residency_plan_partial_layer_count"
            }
            Self::ExpertResidencyPlanStreamedLayerCount => {
                "expert_residency_plan_streamed_layer_count"
            }
            Self::ExpertResidencyPreexistingCompletePayloadBytes => {
                "expert_residency_preexisting_complete_payload_bytes"
            }
            Self::ExpertResidencyPreexistingPartialPayloadBytes => {
                "expert_residency_preexisting_partial_payload_bytes"
            }
            Self::ExpertResidencyPreservedCompletePayloadBytes => {
                "expert_residency_preserved_complete_payload_bytes"
            }
            Self::ExpertResidencyPreservedPartialPayloadBytes => {
                "expert_residency_preserved_partial_payload_bytes"
            }
            Self::ExpertResidencyPromotedCompletePayloadBytes => {
                "mandatory_prefill_complete_layer_promoted_payload_byte_count"
            }
            Self::ExpertResidencyPromotedPartialPayloadBytes => {
                "mandatory_decode_routed_page_promoted_payload_byte_count"
            }
            Self::ExpertResidencyRetiredCompletePayloadBytes => {
                "expert_residency_retired_complete_payload_bytes"
            }
            Self::ExpertResidencyRetiredPartialPayloadBytes => {
                "expert_residency_retired_partial_payload_bytes"
            }
            Self::ExpertTopologyPreservedPayloadBytes => {
                "expert_topology_preserved_payload_byte_count"
            }
            Self::ExpertTopologyRetiredPayloadBytes => "expert_topology_retired_payload_byte_count",
            Self::MandatoryPrefillExpertSourcePayloadBytes => {
                "mandatory_prefill_expert_source_payload_bytes"
            }
            Self::MandatoryDecodeExpertSourcePayloadBytes => {
                "mandatory_decode_expert_source_payload_bytes"
            }
            Self::AvoidedCompleteLayerExpertSourcePayloadBytes => {
                "avoided_complete_layer_expert_source_payload_bytes"
            }
            Self::CompleteLayerPrefetchUsefulPayloadBytes => {
                "complete_layer_prefetch_useful_payload_bytes"
            }
            Self::CompleteLayerPrefetchWastedPayloadBytes => {
                "complete_layer_prefetch_wasted_payload_bytes"
            }
            Self::RetainedRouteAssignmentHitCount => "retained_route_assignment_hit_count",
            Self::RetainedRouteAssignmentMissCount => "retained_route_assignment_miss_count",
            Self::ExpertResidencyCommitRejectionCount => "expert_residency_commit_rejection_count",
            Self::MtpDepthSelectionFallbackCount => "mtp_memory_admission_fallback_count",
            Self::MtpAdmittedAttemptCount => "mtp_admitted_attempt_count",
            Self::SpeculativePrefillTargetOnlyPrefixChunkCount => {
                "speculative_prefill_target_only_prefix_chunk_count"
            }
            Self::SpeculativePrefillTargetOnlyPrefixTokenCount => {
                "speculative_prefill_target_only_prefix_token_count"
            }
            Self::SpeculativePrefillTerminalCaptureChunkCount => {
                "speculative_prefill_terminal_capture_chunk_count"
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
            Self::SpeculativePrefillSparseTargetChunkCount => {
                "speculative_prefill_sparse_target_chunk_count"
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
            Self::MtpRequestedDepthTotal => "mtp_requested_depth_total",
            Self::MtpEffectiveDepthTotal => "mtp_effective_depth_total",
            Self::MtpProposedDraftCount => "mtp_proposed_draft_count",
            Self::MtpEosTruncatedPrefixCount => "mtp_eos_truncated_prefix_count",
            Self::MtpOutputDepthDowngradeCount => "mtp_output_depth_downgrade_count",
            Self::MtpContextDepthDowngradeCount => "mtp_context_depth_downgrade_count",
            Self::MtpThinkingDepthDowngradeCount => "mtp_thinking_depth_downgrade_count",
            Self::MtpMemoryDepthDowngradeCount => "mtp_memory_depth_downgrade_count",
            Self::MtpVerificationWorkspaceByteCount => "mtp_verification_workspace_byte_count",
            Self::MtpBoundarySnapshotByteCount => "mtp_boundary_snapshot_byte_count",
            Self::MtpPersistentGrowthByteCount => "mtp_persistent_growth_byte_count",
            Self::MtpTargetRepairCount => "mtp_target_repair_count",
            Self::MtpPredictorReplayTokenCount => "mtp_predictor_replay_token_count",
            Self::MtpQueuedFrontierRestorationCount => "mtp_queued_frontier_restoration_count",
            Self::MtpCancellationWithQueuedStateCount => "mtp_cancellation_with_queued_state_count",
            Self::MtpOperationalFallbackCount => "mtp_operational_fallback_count",
        }
    }
}
