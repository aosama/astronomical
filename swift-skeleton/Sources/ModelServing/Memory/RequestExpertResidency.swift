import Foundation

/**
 * Request-scoped expert contract. Execution families enact it; they do
 * not replan it.
 *
 * Leftover RAM arithmetic may be recomputed on every chunk. This contract
 * freezes the prefill decision at request open: pinned complete layers stay
 * pinned, streamed layers stay streamed, and a capacity failure may only
 * shrink the pin set.
 */
public struct RequestExpertResidency: Equatable, Sendable {

    private var layerRoles: [RequestExpertLayerRole]

    private init(layerRoles: [RequestExpertLayerRole]) {
        self.layerRoles = layerRoles
    }

    /**
     * Pins every complete-layer target from the opening leftover plan.
     *
     * - Parameters:
     *   - candidatePlan: The opening prefill plan for this request.
     * - Returns: The request's stable residency contract.
     */
    public static func openPrefill(
        candidatePlan: ExpertResidencyPlan
    ) -> RequestExpertResidency {
        var openedRoles: [RequestExpertLayerRole] = Array(
            repeating: RequestExpertLayerRole.streamed,
            count: candidatePlan.layerTargets.count)
        for layerIndex in candidatePlan.completeLayerTargets {
            if layerIndex < openedRoles.count {
                openedRoles[layerIndex] = RequestExpertLayerRole.pinnedComplete
            }
        }
        return RequestExpertResidency(layerRoles: openedRoles)
    }

    /**
     * - Parameters:
     *   - layerIndex: Decoder layer to look up.
     * - Returns: The layer's role, or nil beyond the contract's layer count.
     */
    public func layerRole(_ layerIndex: Int) -> RequestExpertLayerRole? {
        if layerIndex < 0 || layerIndex >= self.layerRoles.count {
            return nil
        }
        return self.layerRoles[layerIndex]
    }

    /**
     * - Returns: Layer indexes pinned complete, in layer order.
     */
    public func pinnedCompleteLayerIndexes() -> [Int] {
        var pinnedLayerIndexes: [Int] = []
        for roleEntry in self.layerRoles.enumerated() {
            if roleEntry.element == RequestExpertLayerRole.pinnedComplete {
                pinnedLayerIndexes.append(roleEntry.offset)
            }
        }
        return pinnedLayerIndexes
    }

    /**
     * Drops pinned complete layers from the tail until the required
     * reclamation is covered. Those layers become streamed for the rest of
     * prefill; a later leftover plan must not pin them again.
     *
     * - Parameters:
     *   - requiredReclamationBytes: Bytes the capacity failure must reclaim.
     *   - layerGeometries: Geometry used to price each pinned layer.
     * - Returns: The shrunk contract.
     */
    public func shrinkAfterCapacityFailure(
        requiredReclamationBytes: UInt64,
        layerGeometries: [ExpertLayerGeometry]
    ) -> RequestExpertResidency {
        if requiredReclamationBytes == 0 {
            return self
        }
        var shrunkRoles: [RequestExpertLayerRole] = self.layerRoles
        var remainingReclamationBytes: UInt64 = requiredReclamationBytes
        var layerIndex: Int = shrunkRoles.count - 1
        while layerIndex >= 0 {
            if remainingReclamationBytes == 0 {
                break
            }
            if shrunkRoles[layerIndex] != RequestExpertLayerRole.pinnedComplete {
                layerIndex -= 1
                continue
            }
            var completeLayerPayloadBytes: UInt64 = 0
            if layerIndex < layerGeometries.count {
                completeLayerPayloadBytes =
                    layerGeometries[layerIndex].completeLayerPayloadBytes
            }
            shrunkRoles[layerIndex] = RequestExpertLayerRole.streamed
            remainingReclamationBytes = SaturatingArithmetic.subtract(
                remainingReclamationBytes, completeLayerPayloadBytes)
            layerIndex -= 1
        }
        return RequestExpertResidency(layerRoles: shrunkRoles)
    }

    /**
     * Rewrites a leftover prefill plan so it cannot promote or release
     * against this contract.
     *
     * - Parameters:
     *   - candidatePlan: The freshly composed leftover plan.
     *   - currentResidencies: Ownership currently materialized.
     * - Returns: The stabilized plan bound to this contract.
     */
    public func stabilizePrefillPlan(
        _ candidatePlan: ExpertResidencyPlan,
        currentResidencies: [CurrentExpertLayerResidency]
    ) -> ExpertResidencyPlan {
        var stabilizedPlan: ExpertResidencyPlan = candidatePlan
        var layerIsCurrentlyComplete: [Bool] =
            Array(repeating: false, count: self.layerRoles.count)
        for currentResidency in currentResidencies {
            if currentResidency.layerIndex < layerIsCurrentlyComplete.count
                && currentResidency.pageClass == .stableCompleteLayer {
                layerIsCurrentlyComplete[currentResidency.layerIndex] = true
            }
        }
        if stabilizedPlan.layerTargets.count != self.layerRoles.count {
            stabilizedPlan.layerTargets = Array(
                repeating: ExpertLayerResidencyTarget.streamOperationLocal,
                count: self.layerRoles.count)
        }
        var expectedPreservedBytes: UInt64 = 0
        var maximumNewRetainedBytes: UInt64 = 0
        for roleEntry in self.layerRoles.enumerated() {
            let currentResidency: CurrentExpertLayerResidency? =
                currentResidencies.first(where: { (residency: CurrentExpertLayerResidency) -> Bool in
                    return residency.layerIndex == roleEntry.offset
                })
            let layerTarget: ExpertLayerResidencyTarget
            switch roleEntry.element {
            case .pinnedComplete:
                if layerIsCurrentlyComplete[roleEntry.offset] {
                    if let residency: CurrentExpertLayerResidency = currentResidency {
                        expectedPreservedBytes = SaturatingArithmetic.add(
                            expectedPreservedBytes, residency.payloadBytes)
                    }
                    layerTarget = .preserveComplete
                } else {
                    if let residency: CurrentExpertLayerResidency = currentResidency {
                        maximumNewRetainedBytes = SaturatingArithmetic.add(
                            maximumNewRetainedBytes, residency.payloadBytes)
                    }
                    layerTarget = .promoteCompleteOnMandatoryRead
                }
            case .streamed:
                layerTarget = .streamOperationLocal
            }
            stabilizedPlan.layerTargets[roleEntry.offset] = layerTarget
        }
        stabilizedPlan.completeLayerTargets = self.pinnedCompleteLayerIndexes()
        stabilizedPlan.reservedRoutedOverlayBytes = 0
        stabilizedPlan.expectedPreservedBytes = expectedPreservedBytes
        stabilizedPlan.maximumNewRetainedBytes = maximumNewRetainedBytes
        stabilizedPlan.isLowBudgetPartialMode = false
        return stabilizedPlan
    }
}
