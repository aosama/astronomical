import Foundation;

/// One bounded numerical counter attached to a performance-attribution report.
///
/// Raw values index a fixed array on the hot path, and declaration order is
/// the zero-allocation bridge from recording to serialized diagnostics: a new
/// counter must always be appended last, or previously stored report rows
/// mislabel one value and drop another. The hermetic catalog suite guards
/// this invariant, because a counter outside the reserved storage once
/// stopped the inference worker mid-request instead of failing a build.
public enum PerformanceCounter: Int, CaseIterable, Sendable {

    case promptTokenCount
    case restoredPersistentPromptCacheTokenCount
    case generatedTokenCount
    case forcedThinkingTransitionTokenCount
    case prefillChunkCount
    case prefillCapacityRejectionCount
    case prefillCapacityRetryCount
    case rustExpertStreamingPayloadByteCount
    case rustStreamedExpertProjectionGraphCount
    case positionalFileReadCallCount
    case positionalFileReadByteCount
    case positionalFileReadElapsedNanoseconds
    case positionalFileReadMaximumElapsedNanoseconds
    case positionalFileReadMaximumConcurrentCount
    case positionalFileReadFailureCount
    case expertRoutePredictedExpertCount
    case expertRouteMatchedExpertCount
    case expertRouteCompletelyMatchedLayerCount
    case expertRouteExaminedLayerCount
    case expertResidencyPlanCompleteLayerCount
    case expertResidencyPlanPartialLayerCount
    case expertResidencyPlanStreamedLayerCount
    case expertResidencyPreexistingCompletePayloadBytes
    case expertResidencyPreexistingPartialPayloadBytes
    case expertResidencyPreservedCompletePayloadBytes
    case expertResidencyPreservedPartialPayloadBytes
    case expertResidencyPromotedCompletePayloadBytes
    case expertResidencyPromotedPartialPayloadBytes
    case expertResidencyRetiredCompletePayloadBytes
    case expertResidencyRetiredPartialPayloadBytes
    case expertTopologyPreservedPayloadBytes
    case expertTopologyRetiredPayloadBytes
    case mandatoryPrefillExpertSourcePayloadBytes
    case mandatoryDecodeExpertSourcePayloadBytes
    case avoidedCompleteLayerExpertSourcePayloadBytes
    case completeLayerPrefetchUsefulPayloadBytes
    case completeLayerPrefetchWastedPayloadBytes
    case retainedRouteAssignmentHitCount
    case retainedRouteAssignmentMissCount
    case hotExpertPartialRouteHitCount
    case hotExpertWarmInsertCount
    /// Decode tokens whose routed experts were all warm (issue #373 baseline).
    case hotExpertRouteFullyCoveredCount
    /// Decode tokens with some warm and some cold routed experts: the mixed-serving target.
    case hotExpertRoutePartiallyCoveredCount
    /// Decode tokens with no warm routed experts.
    case hotExpertRouteFullyMissedCount
    /// Routed assignments served from retained RAM across classified tokens.
    case hotExpertRouteRetainedAssignmentCount
    /// Routed assignments read from storage across classified tokens.
    case hotExpertRouteMissingAssignmentCount
    /// Decode forwards served partly from RAM and partly from storage (issue #373).
    case hotExpertMixedRouteCount
    case expertResidencyCommitRejectionCount
    /// Complete-layer payload prefill handed to retained ownership while
    /// streaming it, so decode seating does not read the same bytes again.
    case expertResidencyReadThroughSeatedCompletePayloadBytes
    /// Complete-layer payload the decode seating pass streamed from storage.
    /// Issue #339 drives this to zero whenever prefill already seated the
    /// layers the decode plan would otherwise re-read.
    case expertResidencyDecodeSeatingStreamedCompletePayloadBytes
    /// Peak MLX active-memory ceiling observed during decode (issue #507).
    case memoryCeilingUtilizationCeilingBytes
    /// Active memory at the same decode step as the utilization split.
    case memoryCeilingUtilizationActiveBytes
    /// Ceiling minus active at that step: the headroom a reader sees idle.
    case memoryCeilingUtilizationUnusedHeadroomBytes
    /// Model-core reserve the loaded core did not occupy at that step.
    case memoryCeilingUtilizationReservedModelCoreSlackBytes
    /// Context-window reserve persistent request state did not occupy.
    case memoryCeilingUtilizationReservedContextGrowthBytes
    /// Activation workspace, stream slot, and other fixed reserves.
    case memoryCeilingUtilizationReservedActivationAndWorkspaceBytes
    /// Expert entitlement granted but never filled by warming: the recoverable
    /// part of the headroom, and the only part a residency change should target.
    case memoryCeilingUtilizationUnseatedExpertEntitlementBytes
    /// Headroom with no named owner. A non-trivial value means an unexplained
    /// decision point exists, which is why it is reported rather than absorbed.
    case memoryCeilingUtilizationUnexplainedHeadroomBytes
    /// Headroom a named owner consumed beyond its reserve, which clamped terms
    /// would otherwise hide.
    case memoryCeilingUtilizationOwnerOverrunBytes
    /// Decode tokens whose true route was stored in the observation history (#536).
    case routeObservationStoredRecordCount
    /// Observations evicted from the history by its capacity bound (#536).
    case routeObservationEvictedRecordCount
    /// Sparse layers whose route was captured for one finalized token (#536).
    case routeObservationCapturedLayerCount
    /// Previous-token experts written into leftover slots (#537).
    case previousTokenPrefetchIssueCount
    /// Next-token demands served from a previous-token prefetch slot (#537).
    case previousTokenPrefetchHitCount
    /// Prefetched experts that the next token did not demand (#537).
    case previousTokenPrefetchMissCount
    /// Payload bytes written by previous-token prefetch inserts (#537).
    case previousTokenPrefetchByteCount
    /// Previous-token experts dropped because leftover slots were full (#537).
    case previousTokenPrefetchCapacityDropCount
    /// Admission decisions composed from the exact-context transient evidence (#691).
    case admissionReserveExactContextSourceCount
    /// Admission decisions composed from a token-scaled phase estimate (#691).
    case admissionReservePhaseScaledSourceCount
    /// Admission decisions composed from the all-phase maximum (#691).
    case admissionReserveGlobalMaximumSourceCount

    /// Number of distinct counters the enabled report storage reserves.
    public static var count: Int {
        PerformanceCounter.allCases.count;
    }
}
