import Foundation;

/// The per-request stack of decoder-cache layer states, port of the Rust
/// `RequestDecoderStateStack`. Layer positions and their families are fixed
/// for the lifetime of a request; the state owners inside each slot are
/// mutable while the stack itself is not.
public final class RequestDecoderStateStack {

    private let decoderLayerStates: [DecoderCacheState];

    public init(decoderLayerStates: [DecoderCacheState]) {
        self.decoderLayerStates = decoderLayerStates;
    }

    public var layerCount: Int {
        return self.decoderLayerStates.count;
    }

    /// The state family at one layer position, or nil when the position is
    /// outside the stack.
    public func layer(layerIndex: Int) -> DecoderCacheState? {
        guard layerIndex >= 0 && layerIndex < self.decoderLayerStates.count
        else {
            return nil;
        }
        return self.decoderLayerStates[layerIndex];
    }
}
