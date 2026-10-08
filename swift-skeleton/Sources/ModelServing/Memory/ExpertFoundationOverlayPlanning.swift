import Foundation

/**
 * The foundation-and-overlay expert residency strategy.
 *
 * When the ceiling cannot hold every complete layer, this planner keeps the
 * routed floor for all layers and promotes the cheapest complete layers on
 * top of that foundation.
 */
internal enum ExpertFoundationOverlayPlanning {

    internal static func foundationAndOverlayPlan(
        phase: MemoryPhase,
        ceilingBytes: UInt64,
        geometries: [ExpertLayerGeometry],
        currentByLayer: [CurrentExpertLayerResidency?],
        routedFloorBytes: [UInt64],
        allLayerRoutedFloorBytes: UInt64
    ) throws -> ExpertResidencyPlan {
        var isCompleteTarget: [Bool] = Array(repeating: false, count: geometries.count)
        var foundationAndFloorBytes: UInt64 = allLayerRoutedFloorBytes
        var completeCandidates:
            [(
                layerIndex: Int,
                incrementalBytes: UInt64,
                isCurrentlyComplete: Bool
            )] = geometries.map({ (geometry: ExpertLayerGeometry) -> (layerIndex: Int, incrementalBytes: UInt64, isCurrentlyComplete: Bool) in
                let isCurrentlyComplete: Bool = currentByLayer[geometry.layerIndex]?
                    .pageClass == .stableCompleteLayer
                return (
                    geometry.layerIndex,
                    geometry.completeLayerPayloadBytes - routedFloorBytes[geometry.layerIndex],
                    isCurrentlyComplete
                )
            })
        completeCandidates.sort { (left, right) -> Bool in
            if left.isCurrentlyComplete != right.isCurrentlyComplete {
                return left.isCurrentlyComplete
            }
            if left.incrementalBytes != right.incrementalBytes {
                return left.incrementalBytes < right.incrementalBytes
            }
            return left.layerIndex < right.layerIndex
        }
        for candidate in completeCandidates {
            let remainingCeilingBytes: UInt64 =
                SaturatingArithmetic.subtract(ceilingBytes, foundationAndFloorBytes)
            if candidate.incrementalBytes <= remainingCeilingBytes {
                isCompleteTarget[candidate.layerIndex] = true
                let (summedFoundation, foundationOverflowed) =
                    foundationAndFloorBytes.addingReportingOverflow(candidate.incrementalBytes)
                if foundationOverflowed {
                    throw ExpertResidencyPlanError.byteCountOverflow
                }
                foundationAndFloorBytes = summedFoundation
            }
        }

        var overlayExtraBytes: UInt64 =
            SaturatingArithmetic.subtract(ceilingBytes, foundationAndFloorBytes)
        var shouldPreservePartial: [Bool] = Array(repeating: false, count: geometries.count)
        var partialCandidates:
            [(layerIndex: Int, residency: CurrentExpertLayerResidency)] = []
        for layerEntry in currentByLayer.enumerated() {
            guard let residency: CurrentExpertLayerResidency = layerEntry.element else {
                continue
            }
            if !isCompleteTarget[layerEntry.offset]
                && residency.pageClass == .elasticRoutedExperts {
                partialCandidates.append((layerEntry.offset, residency))
            }
        }
        partialCandidates.sort { (left, right) -> Bool in
            return ExpertResidencyPlanning.comparePartialCoverage(right, left)
                == .orderedAscending
        }
        for candidate in partialCandidates {
            let incrementalBytes: UInt64 = SaturatingArithmetic.subtract(
                candidate.residency.payloadBytes,
                routedFloorBytes[candidate.layerIndex])
            if incrementalBytes <= overlayExtraBytes {
                shouldPreservePartial[candidate.layerIndex] = true
                overlayExtraBytes = SaturatingArithmetic.subtract(
                    overlayExtraBytes, incrementalBytes)
            }
        }

        var preservedBytes: UInt64 = 0
        var layerTargets: [ExpertLayerResidencyTarget] = []
        layerTargets.reserveCapacity(geometries.count)
        for layerIndex in 0..<geometries.count {
            let currentResidency: CurrentExpertLayerResidency? = currentByLayer[layerIndex]
            let layerTarget: ExpertLayerResidencyTarget
            if isCompleteTarget[layerIndex] {
                if let residency: CurrentExpertLayerResidency = currentResidency,
                    residency.pageClass == .stableCompleteLayer {
                    layerTarget = .preserveComplete
                } else {
                    layerTarget = .promoteCompleteOnMandatoryRead
                }
            } else {
                switch currentResidency {
                case .some(let residency)
                where residency.pageClass == .stableCompleteLayer:
                    layerTarget = .releaseCompleteForExactDeficit
                case .some where shouldPreservePartial[layerIndex]:
                    layerTarget = .preservePartial
                case .some:
                    layerTarget = .releasePartial
                case .none:
                    layerTarget = .admitPartialOnMandatoryRouteRead
                }
            }
            let targetPreservesOwnership: Bool =
                layerTarget == .preserveComplete
                || layerTarget == .preservePartial
                || layerTarget == .promoteCompleteOnMandatoryRead
            if targetPreservesOwnership, let residency: CurrentExpertLayerResidency =
                currentResidency {
                let (summedPreserved, preservedOverflowed) =
                    preservedBytes.addingReportingOverflow(residency.payloadBytes)
                if preservedOverflowed {
                    throw ExpertResidencyPlanError.byteCountOverflow
                }
                preservedBytes = summedPreserved
            }
            layerTargets.append(layerTarget)
        }
        let reservedRoutedOverlayBytes: UInt64 = try ExpertResidencyValidation.checkedSum(
            of: routedFloorBytes.enumerated().compactMap({ (floorEntry: EnumeratedSequence<[UInt64]>.Element) -> UInt64? in
                if isCompleteTarget[floorEntry.offset] {
                    return nil
                }
                return floorEntry.element
            }))
        let targetCapacityBytes: UInt64 = try ExpertResidencyValidation.checkedSum(
            of: geometries.map({ (geometry: ExpertLayerGeometry) -> UInt64 in
                if isCompleteTarget[geometry.layerIndex] {
                    return geometry.completeLayerPayloadBytes
                }
                return routedFloorBytes[geometry.layerIndex]
            }))
        if targetCapacityBytes > ceilingBytes {
            throw ExpertResidencyPlanError.plannedResidencyExceedsCeiling
        }
        return ExpertResidencyPlan(
            phase: phase,
            retainedExpertCeilingBytes: ceilingBytes,
            completeLayerTargets: isCompleteTarget.enumerated().compactMap(
                { (targetEntry: EnumeratedSequence<[Bool]>.Element) -> Int? in
                    if targetEntry.element {
                        return targetEntry.offset
                    }
                    return nil
                }),
            layerTargets: layerTargets,
            reservedRoutedOverlayBytes: reservedRoutedOverlayBytes,
            expectedPreservedBytes: preservedBytes,
            maximumNewRetainedBytes: SaturatingArithmetic.subtract(
                targetCapacityBytes, preservedBytes),
            deterministicReleaseOrder: ExpertResidencyPlanning.releaseOrder(
                currentByLayer: currentByLayer),
            isLowBudgetPartialMode: false)
    }
}
