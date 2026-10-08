import Foundation

/// Fail-closed validation and checked byte arithmetic for residency planning.
internal enum ExpertResidencyValidation {

    /**
     * Validates plan inputs and indexes current residencies by layer.
     *
     * - Throws: `ExpertResidencyPlanError` when geometry is inconsistent,
     *   current residency is out of range, unordered, mis-sized, or already
     *   exceeds the composed ceiling.
     * - Returns: One optional residency per layer, in layer order.
     */
    internal static func validateCurrentResidencies(
        retainedExpertCeilingBytes: UInt64,
        layerGeometries: [ExpertLayerGeometry],
        currentResidencies: [CurrentExpertLayerResidency]
    ) throws -> [CurrentExpertLayerResidency?] {
        if layerGeometries.isEmpty {
            throw ExpertResidencyPlanError.emptyGeometry
        }
        for geometryEntry in layerGeometries.enumerated() {
            try validateGeometry(
                expectedLayerIndex: geometryEntry.offset,
                geometry: geometryEntry.element)
        }
        var currentByLayer: [CurrentExpertLayerResidency?] =
            Array(repeating: nil, count: layerGeometries.count)
        var previousLayerIndex: Int? = nil
        var currentPayloadBytes: UInt64 = 0
        for residency in currentResidencies {
            if residency.layerIndex >= layerGeometries.count {
                throw ExpertResidencyPlanError.currentLayerOutOfRange(
                    layerIndex: residency.layerIndex)
            }
            if let previousLayerIndex = previousLayerIndex,
                previousLayerIndex >= residency.layerIndex {
                throw ExpertResidencyPlanError.duplicateOrUnorderedCurrentLayer(
                    layerIndex: residency.layerIndex)
            }
            previousLayerIndex = residency.layerIndex
            try validateCurrentResidency(
                geometry: layerGeometries[residency.layerIndex],
                residency: residency)
            let (summedPayload, payloadOverflowed) =
                currentPayloadBytes.addingReportingOverflow(residency.payloadBytes)
            if payloadOverflowed {
                throw ExpertResidencyPlanError.byteCountOverflow
            }
            currentPayloadBytes = summedPayload
            currentByLayer[residency.layerIndex] = residency
        }
        if currentPayloadBytes > retainedExpertCeilingBytes {
            throw ExpertResidencyPlanError.currentResidencyExceedsCeiling
        }
        return currentByLayer
    }

    /**
     * Payload bytes one layer's routed floor occupies: the experts a single
     * token must read, bounded by the layer's expert capacity.
     *
     * - Throws: `ExpertResidencyPlanError.byteCountOverflow` on arithmetic
     *   overflow.
     */
    internal static func routedFloorPayloadBytes(
        for geometry: ExpertLayerGeometry
    ) throws -> UInt64 {
        let routedExpertCount: Int = min(geometry.expertsPerToken, geometry.expertCapacity)
        guard let routedExpertCountBytes: UInt64 = UInt64(exactly: routedExpertCount) else {
            throw ExpertResidencyPlanError.byteCountOverflow
        }
        let (floorBytes, floorOverflowed) =
            geometry.expertPayloadBytes.multipliedReportingOverflow(by: routedExpertCountBytes)
        if floorOverflowed {
            throw ExpertResidencyPlanError.byteCountOverflow
        }
        return floorBytes
    }

    /**
     * Sums byte counts, rejecting overflow instead of wrapping.
     *
     * - Throws: `ExpertResidencyPlanError.byteCountOverflow` when the total
     *   exceeds `UInt64.max`.
     */
    internal static func checkedSum(of byteCounts: [UInt64]) throws -> UInt64 {
        var totalBytes: UInt64 = 0
        for byteCount in byteCounts {
            let (summedTotal, addOverflowed) = totalBytes.addingReportingOverflow(byteCount)
            if addOverflowed {
                throw ExpertResidencyPlanError.byteCountOverflow
            }
            totalBytes = summedTotal
        }
        return totalBytes
    }

    private static func validateGeometry(
        expectedLayerIndex: Int,
        geometry: ExpertLayerGeometry
    ) throws -> Void {
        if geometry.layerIndex != expectedLayerIndex {
            throw ExpertResidencyPlanError.nonContiguousLayerIndex(
                expectedLayerIndex: expectedLayerIndex,
                actualLayerIndex: geometry.layerIndex)
        }
        if geometry.expertCapacity == 0
            || geometry.expertPayloadBytes == 0
            || geometry.completeLayerPayloadBytes == 0
            || geometry.expertsPerToken == 0 {
            throw ExpertResidencyPlanError.zeroGeometry(layerIndex: geometry.layerIndex)
        }
        guard let expertCapacityBytes: UInt64 = UInt64(exactly: geometry.expertCapacity) else {
            throw ExpertResidencyPlanError.byteCountOverflow
        }
        let (expectedCompleteBytes, completeOverflowed) =
            geometry.expertPayloadBytes.multipliedReportingOverflow(by: expertCapacityBytes)
        if completeOverflowed {
            throw ExpertResidencyPlanError.byteCountOverflow
        }
        if expectedCompleteBytes != geometry.completeLayerPayloadBytes {
            throw ExpertResidencyPlanError.inconsistentCompletePayload(
                layerIndex: geometry.layerIndex)
        }
    }

    private static func validateCurrentResidency(
        geometry: ExpertLayerGeometry,
        residency: CurrentExpertLayerResidency
    ) throws -> Void {
        if !retainedExpertIdsAreValid(
            retainedExpertIds: residency.retainedExpertIds,
            expertCapacity: geometry.expertCapacity) {
            throw ExpertResidencyPlanError.invalidRetainedExpertIds(
                layerIndex: residency.layerIndex)
        }
        guard let retainedCountBytes: UInt64 = UInt64(exactly: residency.retainedExpertIds.count)
        else {
            throw ExpertResidencyPlanError.byteCountOverflow
        }
        let (expectedPayloadBytes, payloadOverflowed) =
            geometry.expertPayloadBytes.multipliedReportingOverflow(by: retainedCountBytes)
        if payloadOverflowed {
            throw ExpertResidencyPlanError.byteCountOverflow
        }
        let classIsConsistent: Bool
        switch residency.pageClass {
        case .stableCompleteLayer:
            classIsConsistent =
                residency.retainedExpertIds.count == geometry.expertCapacity
        case .elasticRoutedExperts:
            classIsConsistent =
                residency.retainedExpertIds.count < geometry.expertCapacity
        }
        if expectedPayloadBytes != residency.payloadBytes || !classIsConsistent {
            throw ExpertResidencyPlanError.inconsistentCurrentPayload(
                layerIndex: residency.layerIndex,
                payloadBytes: residency.payloadBytes,
                geometryExpertPayloadBytes: geometry.expertPayloadBytes,
                retainedCount: residency.retainedExpertIds.count,
                expectedPayloadBytes: expectedPayloadBytes)
        }
    }

    private static func retainedExpertIdsAreValid(
        retainedExpertIds: [Int],
        expertCapacity: Int
    ) -> Bool {
        if retainedExpertIds.isEmpty {
            return false
        }
        var pairIndex: Int = 1
        while pairIndex < retainedExpertIds.count {
            if retainedExpertIds[pairIndex - 1] >= retainedExpertIds[pairIndex] {
                return false
            }
            pairIndex += 1
        }
        for expertId in retainedExpertIds {
            if expertId >= expertCapacity {
                return false
            }
        }
        return true
    }
}
