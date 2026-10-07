import Foundation

/// Structural input defect that must stop planning before ownership changes.
public enum ExpertResidencyPlanError: Error, Equatable, Sendable {

    /// The geometry list held no layers at all.
    case emptyGeometry

    /// Layer indexes must be contiguous from zero.
    case nonContiguousLayerIndex(expectedLayerIndex: Int, actualLayerIndex: Int)

    /// A layer carried a zero capacity, payload, or experts-per-token.
    case zeroGeometry(layerIndex: Int)

    /// The complete payload did not equal expert payload times capacity.
    case inconsistentCompletePayload(layerIndex: Int)

    /// Current residency layer indexes must strictly ascend.
    case duplicateOrUnorderedCurrentLayer(layerIndex: Int)

    /// Current residency referenced a layer the geometry does not contain.
    case currentLayerOutOfRange(layerIndex: Int)

    /// Retained expert identifiers were empty, unsorted, duplicated, or
    /// beyond the layer's expert capacity.
    case invalidRetainedExpertIds(layerIndex: Int)

    /// Retained payload disagreed with geometry or residency class.
    case inconsistentCurrentPayload(
        layerIndex: Int,
        payloadBytes: UInt64,
        geometryExpertPayloadBytes: UInt64,
        retainedCount: Int,
        expectedPayloadBytes: UInt64)

    /// Current retained payload exceeded the composed ceiling.
    case currentResidencyExceedsCeiling

    /// Planned retained payload exceeded the composed ceiling.
    case plannedResidencyExceedsCeiling

    /// Expert residency byte arithmetic overflowed.
    case byteCountOverflow
}
