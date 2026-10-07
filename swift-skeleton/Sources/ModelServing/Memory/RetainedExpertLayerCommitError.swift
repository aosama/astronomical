import Foundation

/// Invalid commit metadata rejected before prior ownership can be mutated.
public enum RetainedExpertLayerCommitError: Error, Equatable {

    /// The commit references a layer the cache does not hold.
    case layerOutOfRange(layerIndex: Int)

    /// The commit declares zero expert capacity for the layer.
    case zeroExpertCapacity(layerIndex: Int)

    /// Routed expert identifiers are empty, not strictly ascending, out of
    /// range, or cover the whole capacity (complete reads use the complete
    /// commit instead).
    case invalidExpertIds(layerIndex: Int)

    /// The offered page materialized zero payload bytes.
    case zeroPayload(layerIndex: Int)

    /// Adding the candidate payload to resident accounting overflowed.
    case payloadByteCountOverflow(layerIndex: Int)

    /// Resident accounting underflowed before the candidate was added,
    /// meaning bookkeeping drifted from owned pages.
    case inconsistentPayloadAccounting(layerIndex: Int)
}
