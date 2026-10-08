import Foundation

/**
 * Family-neutral expert residency planning: plan, keep, release, stream.
 *
 * This namespace owns no arrays and performs no I/O. Execution families ask
 * it which experts to keep after a mandatory read, whether a planned
 * release may run in the current phase, and how a deterministic topology
 * fits the composed ceiling. They must not invent a second answer to those
 * questions.
 */
public enum ExpertResidencyPlanning {

    /**
     * Produces a deterministic complete-foundation plus routed-overlay plan.
     *
     * - Parameters:
     *   - phase: Lifecycle position the plan is composed for.
     *   - retainedExpertCeilingBytes: Composed ceiling for retained experts.
     *   - layerGeometries: Exact geometry per sparse decoder layer,
     *     contiguously indexed from zero.
     *   - currentResidencies: Current materialized ownership, strictly
     *     ascending by layer.
     * - Throws: `ExpertResidencyPlanError` on invalid inputs or overflow.
     * - Returns: The phase's residency plan.
     */
    public static func planExpertResidency(
        phase: MemoryPhase,
        retainedExpertCeilingBytes: UInt64,
        layerGeometries: [ExpertLayerGeometry],
        currentResidencies: [CurrentExpertLayerResidency]
    ) throws -> ExpertResidencyPlan {
        let currentByLayer: [CurrentExpertLayerResidency?] =
            try ExpertResidencyValidation.validateCurrentResidencies(
                retainedExpertCeilingBytes: retainedExpertCeilingBytes,
                layerGeometries: layerGeometries,
                currentResidencies: currentResidencies)
        // Generation must keep experts prefill already paid to read. Seating
        // new complete layers over those pages would evict them. After an
        // atomic complete-owner demote the topology is empty: leftover budget
        // must still seat complete layers, or generate runs with zero expert
        // RAM while tens of gigabytes of leftover sit unused.
        if (phase == .generationPreparation || phase == .decode) && !currentResidencies.isEmpty {
            return try preserveExistingExpertPagesForGeneration(
                phase: phase,
                ceilingBytes: retainedExpertCeilingBytes,
                geometries: layerGeometries,
                currentByLayer: currentByLayer)
        }
        let completeModelPayloadBytes: UInt64 = try ExpertResidencyValidation.checkedSum(
            of: layerGeometries.map({ (geometry: ExpertLayerGeometry) -> UInt64 in
                geometry.completeLayerPayloadBytes
            }))
        if completeModelPayloadBytes <= retainedExpertCeilingBytes {
            return try completeModelPlan(
                phase: phase,
                ceilingBytes: retainedExpertCeilingBytes,
                geometries: layerGeometries,
                currentByLayer: currentByLayer,
                completeModelPayloadBytes: completeModelPayloadBytes)
        }
        let routedFloorBytes: [UInt64] = try layerGeometries.map(
            { (geometry: ExpertLayerGeometry) -> UInt64 in
                try ExpertResidencyValidation.routedFloorPayloadBytes(for: geometry)
            })
        let allLayerRoutedFloorBytes: UInt64 =
            try ExpertResidencyValidation.checkedSum(of: routedFloorBytes)
        if allLayerRoutedFloorBytes > retainedExpertCeilingBytes {
            return try lowBudgetPartialPlan(
                phase: phase,
                ceilingBytes: retainedExpertCeilingBytes,
                geometries: layerGeometries,
                currentByLayer: currentByLayer)
        }
        return try ExpertFoundationOverlayPlanning.foundationAndOverlayPlan(
            phase: phase,
            ceilingBytes: retainedExpertCeilingBytes,
            geometries: layerGeometries,
            currentByLayer: currentByLayer,
            routedFloorBytes: routedFloorBytes,
            allLayerRoutedFloorBytes: allLayerRoutedFloorBytes)
    }

    /**
     * Orders two partial pages by coverage per payload byte; ties break to
     * the higher layer index so ordering is total and deterministic.
     */
    internal static func comparePartialCoverage(
        _ left: (layerIndex: Int, residency: CurrentExpertLayerResidency),
        _ right: (layerIndex: Int, residency: CurrentExpertLayerResidency)
    ) -> ComparisonResult {
        let leftScore: UInt128 =
            UInt128(left.residency.coveredWeightedDemand)
            * UInt128(right.residency.payloadBytes)
        let rightScore: UInt128 =
            UInt128(right.residency.coveredWeightedDemand)
            * UInt128(left.residency.payloadBytes)
        if leftScore < rightScore {
            return .orderedAscending
        }
        if leftScore > rightScore {
            return .orderedDescending
        }
        if right.layerIndex < left.layerIndex {
            return .orderedAscending
        }
        if right.layerIndex > left.layerIndex {
            return .orderedDescending
        }
        return .orderedSame
    }

    /// Deterministic release order: partials by ascending coverage, then
    /// complete layers from the deepest layer backward.
    internal static func releaseOrder(
        currentByLayer: [CurrentExpertLayerResidency?]
    ) -> [Int] {
        var partialLayers: [(layerIndex: Int, residency: CurrentExpertLayerResidency)] = []
        for layerEntry in currentByLayer.enumerated() {
            guard let residency: CurrentExpertLayerResidency = layerEntry.element else {
                continue
            }
            if residency.pageClass == .elasticRoutedExperts {
                partialLayers.append((layerEntry.offset, residency))
            }
        }
        partialLayers.sort { (left, right) -> Bool in
            return comparePartialCoverage(left, right) == .orderedAscending
        }
        var orderedLayerIndexes: [Int] = partialLayers.map({ (partial: (layerIndex: Int, residency: CurrentExpertLayerResidency)) -> Int in
            return partial.layerIndex
        })
        for layerEntry in currentByLayer.enumerated().reversed() {
            if let residency: CurrentExpertLayerResidency = layerEntry.element,
                residency.pageClass == .stableCompleteLayer {
                orderedLayerIndexes.append(layerEntry.offset)
            }
        }
        return orderedLayerIndexes
    }

    private static func completeModelPlan(
        phase: MemoryPhase,
        ceilingBytes: UInt64,
        geometries: [ExpertLayerGeometry],
        currentByLayer: [CurrentExpertLayerResidency?],
        completeModelPayloadBytes: UInt64
    ) throws -> ExpertResidencyPlan {
        var preservedBytes: UInt64 = 0
        var layerTargets: [ExpertLayerResidencyTarget] = []
        layerTargets.reserveCapacity(geometries.count)
        for currentResidency in currentByLayer {
            if let residency: CurrentExpertLayerResidency = currentResidency,
                residency.pageClass == .stableCompleteLayer {
                layerTargets.append(.preserveComplete)
            } else {
                layerTargets.append(.promoteCompleteOnMandatoryRead)
            }
            if let residency: CurrentExpertLayerResidency = currentResidency {
                let (summedPreserved, preservedOverflowed) =
                    preservedBytes.addingReportingOverflow(residency.payloadBytes)
                if preservedOverflowed {
                    throw ExpertResidencyPlanError.byteCountOverflow
                }
                preservedBytes = summedPreserved
            }
        }
        return ExpertResidencyPlan(
            phase: phase,
            retainedExpertCeilingBytes: ceilingBytes,
            completeLayerTargets: Array(0..<geometries.count),
            layerTargets: layerTargets,
            reservedRoutedOverlayBytes: 0,
            expectedPreservedBytes: preservedBytes,
            maximumNewRetainedBytes: SaturatingArithmetic.subtract(
                completeModelPayloadBytes, preservedBytes),
            deterministicReleaseOrder: releaseOrder(currentByLayer: currentByLayer),
            isLowBudgetPartialMode: false)
    }

    private static func preserveExistingExpertPagesForGeneration(
        phase: MemoryPhase,
        ceilingBytes: UInt64,
        geometries: [ExpertLayerGeometry],
        currentByLayer: [CurrentExpertLayerResidency?]
    ) throws -> ExpertResidencyPlan {
        let selectedLayerIndexes: [Int] = greedilySelectedLayerIndexes(
            ceilingBytes: ceilingBytes,
            layerCount: geometries.count,
            currentByLayer: currentByLayer)
        var preservedBytes: UInt64 = 0
        var layerTargets: [ExpertLayerResidencyTarget] = []
        layerTargets.reserveCapacity(geometries.count)
        for layerEntry in currentByLayer.enumerated() {
            guard let residency: CurrentExpertLayerResidency = layerEntry.element else {
                layerTargets.append(.admitPartialOnMandatoryRouteRead)
                continue
            }
            let isSelected: Bool = selectedLayerIndexes.contains(layerEntry.offset)
            if isSelected {
                if residency.pageClass == .stableCompleteLayer {
                    layerTargets.append(.preserveComplete)
                } else {
                    layerTargets.append(.preservePartial)
                }
                let (summedPreserved, preservedOverflowed) =
                    preservedBytes.addingReportingOverflow(residency.payloadBytes)
                if preservedOverflowed {
                    throw ExpertResidencyPlanError.byteCountOverflow
                }
                preservedBytes = summedPreserved
            } else {
                if residency.pageClass == .stableCompleteLayer {
                    layerTargets.append(.releaseCompleteForExactDeficit)
                } else {
                    layerTargets.append(.releasePartial)
                }
            }
        }
        return ExpertResidencyPlan(
            phase: phase,
            retainedExpertCeilingBytes: ceilingBytes,
            completeLayerTargets: selectedCompleteLayerIndexes(
                currentByLayer: currentByLayer, selectedLayerIndexes: selectedLayerIndexes),
            layerTargets: layerTargets,
            reservedRoutedOverlayBytes: 0,
            expectedPreservedBytes: preservedBytes,
            maximumNewRetainedBytes: SaturatingArithmetic.subtract(ceilingBytes, preservedBytes),
            deterministicReleaseOrder: releaseOrder(currentByLayer: currentByLayer),
            isLowBudgetPartialMode: false)
    }

    private static func lowBudgetPartialPlan(
        phase: MemoryPhase,
        ceilingBytes: UInt64,
        geometries: [ExpertLayerGeometry],
        currentByLayer: [CurrentExpertLayerResidency?]
    ) throws -> ExpertResidencyPlan {
        let selectedLayerIndexes: [Int] = greedilySelectedLayerIndexes(
            ceilingBytes: ceilingBytes,
            layerCount: geometries.count,
            currentByLayer: currentByLayer)
        var preservedBytes: UInt64 = 0
        var layerTargets: [ExpertLayerResidencyTarget] = []
        layerTargets.reserveCapacity(geometries.count)
        for layerEntry in currentByLayer.enumerated() {
            guard let residency: CurrentExpertLayerResidency = layerEntry.element else {
                layerTargets.append(.streamOperationLocal)
                continue
            }
            let isSelected: Bool = selectedLayerIndexes.contains(layerEntry.offset)
            if isSelected {
                if residency.pageClass == .stableCompleteLayer {
                    layerTargets.append(.preserveComplete)
                } else {
                    layerTargets.append(.preservePartial)
                }
                let (summedPreserved, preservedOverflowed) =
                    preservedBytes.addingReportingOverflow(residency.payloadBytes)
                if preservedOverflowed {
                    throw ExpertResidencyPlanError.byteCountOverflow
                }
                preservedBytes = summedPreserved
            } else {
                if residency.pageClass == .stableCompleteLayer {
                    layerTargets.append(.releaseCompleteForExactDeficit)
                } else {
                    layerTargets.append(.releasePartial)
                }
            }
        }
        return ExpertResidencyPlan(
            phase: phase,
            retainedExpertCeilingBytes: ceilingBytes,
            completeLayerTargets: selectedCompleteLayerIndexes(
                currentByLayer: currentByLayer, selectedLayerIndexes: selectedLayerIndexes),
            layerTargets: layerTargets,
            reservedRoutedOverlayBytes: 0,
            expectedPreservedBytes: preservedBytes,
            maximumNewRetainedBytes: 0,
            deterministicReleaseOrder: releaseOrder(currentByLayer: currentByLayer),
            isLowBudgetPartialMode: true)
    }

    /// Greedy preservation: current pages by preservation priority while
    /// each fits the remaining ceiling.
    private static func greedilySelectedLayerIndexes(
        ceilingBytes: UInt64,
        layerCount: Int,
        currentByLayer: [CurrentExpertLayerResidency?]
    ) -> [Int] {
        var candidateLayers: [(layerIndex: Int, residency: CurrentExpertLayerResidency)] = []
        for layerEntry in currentByLayer.enumerated() {
            guard let residency: CurrentExpertLayerResidency = layerEntry.element else {
                continue
            }
            candidateLayers.append((layerEntry.offset, residency))
        }
        candidateLayers.sort { (left, right) -> Bool in
            return preservationPriorityPrecedes(left, right)
        }
        var selectedLayerIndexes: [Int] = []
        var preservedBytes: UInt64 = 0
        for candidate in candidateLayers {
            let remainingCeilingBytes: UInt64 =
                SaturatingArithmetic.subtract(ceilingBytes, preservedBytes)
            if candidate.residency.payloadBytes <= remainingCeilingBytes {
                selectedLayerIndexes.append(candidate.layerIndex)
                preservedBytes = SaturatingArithmetic.add(
                    preservedBytes, candidate.residency.payloadBytes)
            }
        }
        return selectedLayerIndexes
    }

    /// Preservation priority: partial pages precede complete layers, then
    /// higher coverage per payload byte precedes lower.
    private static func preservationPriorityPrecedes(
        _ left: (layerIndex: Int, residency: CurrentExpertLayerResidency),
        _ right: (layerIndex: Int, residency: CurrentExpertLayerResidency)
    ) -> Bool {
        let leftIsComplete: Bool = left.residency.pageClass == .stableCompleteLayer
        let rightIsComplete: Bool = right.residency.pageClass == .stableCompleteLayer
        if leftIsComplete != rightIsComplete {
            return !leftIsComplete
        }
        return comparePartialCoverage(right, left) == .orderedAscending
    }

    private static func selectedCompleteLayerIndexes(
        currentByLayer: [CurrentExpertLayerResidency?],
        selectedLayerIndexes: [Int]
    ) -> [Int] {
        var completeLayerTargets: [Int] = []
        for layerEntry in currentByLayer.enumerated() {
            guard let residency: CurrentExpertLayerResidency = layerEntry.element else {
                continue
            }
            if residency.pageClass == .stableCompleteLayer
                && selectedLayerIndexes.contains(layerEntry.offset) {
                completeLayerTargets.append(layerEntry.offset)
            }
        }
        return completeLayerTargets
    }
}
