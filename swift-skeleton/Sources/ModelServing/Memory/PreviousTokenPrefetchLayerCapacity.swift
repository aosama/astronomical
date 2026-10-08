import Foundation

/// How many leftover slots one sparse layer may fill without eviction,
/// port of the Rust `PreviousTokenPrefetchLayerCapacity`.
public struct PreviousTokenPrefetchLayerCapacity: Equatable, Sendable {

    public let layerIndex: Int

    public let freeSlotCount: Int

    public init(layerIndex: Int, freeSlotCount: Int) {
        self.layerIndex = layerIndex
        self.freeSlotCount = freeSlotCount
    }
}
