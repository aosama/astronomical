import Foundation

/// One request-stable answer to pin, stream, and shrink.
public enum RequestStableResidencyPlanning {

    /**
     * Binds leftover packing to the request contract for the current phase.
     *
     * Prefill opens once, then only shrinks. Generation, decode, and idle
     * discard the prefill contract and keep the leftover candidate
     * unchanged.
     *
     * - Parameters:
     *   - phase: Lifecycle position the plan is published for.
     *   - existingRequestResidency: Contract already opened for this
     *     request, if any.
     *   - candidatePlan: Leftover plan computed for this boundary.
     *   - currentResidencies: Ownership currently materialized.
     *   - releasedCompletePayloadBytes: Complete-layer bytes a capacity
     *     failure reclaimed since the last publish.
     *   - layerGeometries: Geometry used to price shrink targets.
     * - Returns: The surviving contract (nil outside prefill) and the plan
     *   to enact.
     */
    public static func publishRequestStableResidencyPlan(
        phase: MemoryPhase,
        existingRequestResidency: RequestExpertResidency?,
        candidatePlan: ExpertResidencyPlan,
        currentResidencies: [CurrentExpertLayerResidency],
        releasedCompletePayloadBytes: UInt64,
        layerGeometries: [ExpertLayerGeometry]
    ) -> (requestResidency: RequestExpertResidency?, plan: ExpertResidencyPlan) {
        switch phase {
        case .prefill:
            var requestResidency: RequestExpertResidency
            if let existingRequestResidency: RequestExpertResidency =
                existingRequestResidency {
                requestResidency = existingRequestResidency
            } else {
                requestResidency = RequestExpertResidency.openPrefill(
                    candidatePlan: candidatePlan)
            }
            if existingRequestResidency != nil && releasedCompletePayloadBytes > 0 {
                requestResidency = requestResidency.shrinkAfterCapacityFailure(
                    requiredReclamationBytes: releasedCompletePayloadBytes,
                    layerGeometries: layerGeometries)
            }
            let stabilizedPlan: ExpertResidencyPlan = requestResidency.stabilizePrefillPlan(
                candidatePlan, currentResidencies: currentResidencies)
            return (requestResidency, stabilizedPlan)
        case .generationPreparation, .decode, .idle:
            return (nil, candidatePlan)
        }
    }

    /**
     * Floors the retained-page ceiling at complete layers already in RAM.
     *
     * Leftover arithmetic can tighten after a chunk or a decode token
     * because a learned context reserve grew. Evicting a seated complete
     * layer to match that smaller number throws away a page this request
     * already paid to read, then every later token streams it again. Real
     * capacity failure still shrinks through
     * `shrinkAfterCapacityFailure` or request-pressure deficit.
     */
    public static func retainedCompleteLayerCeilingAfterPrefillBudgetRefresh(
        leftoverExpertBudgetBytes: UInt64,
        currentCompleteLayerPayloadBytes: UInt64
    ) -> UInt64 {
        if leftoverExpertBudgetBytes > currentCompleteLayerPayloadBytes {
            return leftoverExpertBudgetBytes
        }
        return currentCompleteLayerPayloadBytes
    }

    /**
     * Floors the retained-page ceiling at every resident expert page, warm
     * tables included.
     *
     * A learned budget that tightens between requests must not proactively
     * discard routing knowledge: the hot-expert tables cost disk re-reads
     * to rebuild, and the request-pressure paths already reclaim them
     * exactly when a real forward needs the bytes.
     */
    public static func retainedResidentCeilingAfterBudgetRefresh(
        leftoverExpertBudgetBytes: UInt64,
        currentResidentPayloadBytes: UInt64
    ) -> UInt64 {
        if leftoverExpertBudgetBytes > currentResidentPayloadBytes {
            return leftoverExpertBudgetBytes
        }
        return currentResidentPayloadBytes
    }
}
