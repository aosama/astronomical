import Foundation;

extension PerformanceCounter {

    /// Stable report identifier serialized into attribution diagnostics.
    ///
    /// Report records are keyed by identifier, so every counter must keep a
    /// unique, nonempty string here; two counters sharing one identifier would
    /// silently overwrite each other in serialized diagnostics.
    public var identifier: String {
        switch self {
        case .promptTokenCount: return "prompt_token_count";
        case .restoredPersistentPromptCacheTokenCount: return "restored_persistent_prompt_cache_token_count";
        case .generatedTokenCount: return "generated_token_count";
        case .forcedThinkingTransitionTokenCount: return "forced_thinking_transition_token_count";
        case .prefillChunkCount: return "prefill_chunk_count";
        case .prefillCapacityRejectionCount: return "prefill_capacity_rejection_count";
        case .prefillCapacityRetryCount: return "prefill_capacity_retry_count";
        case .rustExpertStreamingPayloadByteCount: return "rust_expert_streaming_payload_byte_count";
        case .rustStreamedExpertProjectionGraphCount: return "rust_streamed_expert_projection_graph_count";
        case .positionalFileReadCallCount: return "positional_file_read_call_count";
        case .positionalFileReadByteCount: return "positional_file_read_byte_count";
        case .positionalFileReadElapsedNanoseconds: return "positional_file_read_elapsed_nanoseconds";
        case .positionalFileReadMaximumElapsedNanoseconds: return "positional_file_read_maximum_elapsed_nanoseconds";
        case .positionalFileReadMaximumConcurrentCount: return "positional_file_read_maximum_concurrent_count";
        case .positionalFileReadFailureCount: return "positional_file_read_failure_count";
        case .expertRoutePredictedExpertCount: return "expert_route_predicted_expert_count";
        case .expertRouteMatchedExpertCount: return "expert_route_matched_expert_count";
        case .expertRouteCompletelyMatchedLayerCount: return "expert_route_completely_matched_layer_count";
        case .expertRouteExaminedLayerCount: return "expert_route_examined_layer_count";
        case .expertResidencyPlanCompleteLayerCount: return "expert_residency_plan_complete_layer_count";
        case .expertResidencyPlanPartialLayerCount: return "expert_residency_plan_partial_layer_count";
        case .expertResidencyPlanStreamedLayerCount: return "expert_residency_plan_streamed_layer_count";
        case .expertResidencyPreexistingCompletePayloadBytes: return "expert_residency_preexisting_complete_payload_bytes";
        case .expertResidencyPreexistingPartialPayloadBytes: return "expert_residency_preexisting_partial_payload_bytes";
        case .expertResidencyPreservedCompletePayloadBytes: return "expert_residency_preserved_complete_payload_bytes";
        case .expertResidencyPreservedPartialPayloadBytes: return "expert_residency_preserved_partial_payload_bytes";
        case .expertResidencyPromotedCompletePayloadBytes: return "mandatory_prefill_complete_layer_promoted_payload_byte_count";
        case .expertResidencyPromotedPartialPayloadBytes: return "mandatory_decode_routed_page_promoted_payload_byte_count";
        case .expertResidencyRetiredCompletePayloadBytes: return "expert_residency_retired_complete_payload_bytes";
        case .expertResidencyRetiredPartialPayloadBytes: return "expert_residency_retired_partial_payload_bytes";
        case .expertTopologyPreservedPayloadBytes: return "expert_topology_preserved_payload_byte_count";
        case .expertTopologyRetiredPayloadBytes: return "expert_topology_retired_payload_byte_count";
        case .mandatoryPrefillExpertSourcePayloadBytes: return "mandatory_prefill_expert_source_payload_bytes";
        case .mandatoryDecodeExpertSourcePayloadBytes: return "mandatory_decode_expert_source_payload_bytes";
        case .avoidedCompleteLayerExpertSourcePayloadBytes: return "avoided_complete_layer_expert_source_payload_bytes";
        case .completeLayerPrefetchUsefulPayloadBytes: return "complete_layer_prefetch_useful_payload_bytes";
        case .completeLayerPrefetchWastedPayloadBytes: return "complete_layer_prefetch_wasted_payload_bytes";
        case .retainedRouteAssignmentHitCount: return "retained_route_assignment_hit_count";
        case .retainedRouteAssignmentMissCount: return "retained_route_assignment_miss_count";
        case .hotExpertPartialRouteHitCount: return "hot_expert_partial_route_hit_count";
        case .hotExpertWarmInsertCount: return "hot_expert_warm_insert_count";
        case .hotExpertRouteFullyCoveredCount: return "hot_expert_route_fully_covered_count";
        case .hotExpertRoutePartiallyCoveredCount: return "hot_expert_route_partially_covered_count";
        case .hotExpertRouteFullyMissedCount: return "hot_expert_route_fully_missed_count";
        case .hotExpertRouteRetainedAssignmentCount: return "hot_expert_route_retained_assignment_count";
        case .hotExpertRouteMissingAssignmentCount: return "hot_expert_route_missing_assignment_count";
        case .hotExpertMixedRouteCount: return "hot_expert_mixed_route_count";
        case .expertResidencyCommitRejectionCount: return "expert_residency_commit_rejection_count";
        case .expertResidencyReadThroughSeatedCompletePayloadBytes: return "expert_residency_read_through_seated_complete_payload_bytes";
        case .expertResidencyDecodeSeatingStreamedCompletePayloadBytes: return "expert_residency_decode_seating_streamed_complete_payload_bytes";
        case .memoryCeilingUtilizationCeilingBytes: return "memory_ceiling_utilization_ceiling_bytes";
        case .memoryCeilingUtilizationActiveBytes: return "memory_ceiling_utilization_active_bytes";
        case .memoryCeilingUtilizationUnusedHeadroomBytes: return "memory_ceiling_utilization_unused_headroom_bytes";
        case .memoryCeilingUtilizationReservedModelCoreSlackBytes: return "memory_ceiling_utilization_reserved_model_core_slack_bytes";
        case .memoryCeilingUtilizationReservedContextGrowthBytes: return "memory_ceiling_utilization_reserved_context_growth_bytes";
        case .memoryCeilingUtilizationReservedActivationAndWorkspaceBytes: return "memory_ceiling_utilization_reserved_activation_and_workspace_bytes";
        case .memoryCeilingUtilizationUnseatedExpertEntitlementBytes: return "memory_ceiling_utilization_unseated_expert_entitlement_bytes";
        case .memoryCeilingUtilizationUnexplainedHeadroomBytes: return "memory_ceiling_utilization_unexplained_headroom_bytes";
        case .memoryCeilingUtilizationOwnerOverrunBytes: return "memory_ceiling_utilization_owner_overrun_bytes";
        case .routeObservationStoredRecordCount: return "route_observation_stored_record_count";
        case .routeObservationEvictedRecordCount: return "route_observation_evicted_record_count";
        case .routeObservationCapturedLayerCount: return "route_observation_captured_layer_count";
        case .previousTokenPrefetchIssueCount: return "previous_token_prefetch_issue_count";
        case .previousTokenPrefetchHitCount: return "previous_token_prefetch_hit_count";
        case .previousTokenPrefetchMissCount: return "previous_token_prefetch_miss_count";
        case .previousTokenPrefetchByteCount: return "previous_token_prefetch_byte_count";
        case .previousTokenPrefetchCapacityDropCount: return "previous_token_prefetch_capacity_drop_count";
        case .admissionReserveExactContextSourceCount: return "admission_reserve_exact_context_source_count";
        case .admissionReservePhaseScaledSourceCount: return "admission_reserve_phase_scaled_source_count";
        case .admissionReserveGlobalMaximumSourceCount: return "admission_reserve_global_maximum_source_count";
        }
    }
}
