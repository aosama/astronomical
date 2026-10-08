import Foundation

/// One expert-slot pairing retained by a previous-token prefetch plan,
/// replacing the Rust plan's `(layer_index, expert_id)` tuple so plans
/// stay `Equatable` under Swift Testing assertions.
public struct PreviousTokenPrefetchRetention: Equatable, Sendable {

    public let layerIndex: Int

    public let expertId: Int

    public init(layerIndex: Int, expertId: Int) {
        self.layerIndex = layerIndex
        self.expertId = expertId
    }
}
