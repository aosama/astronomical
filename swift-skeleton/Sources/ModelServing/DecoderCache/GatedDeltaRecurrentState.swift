import Foundation;

import MLX;

/// One hybrid layer's gated-delta recurrent state, port of the Rust
/// `GatedDeltaRecurrentState` owner. Like the convolution, the state is one
/// tensor snapshot at a time.
public final class GatedDeltaRecurrentState {

    private var stateStorage: MLXArray?;

    public init() {
        self.stateStorage = nil;
    }

    public func state() -> MLXArray? {
        return self.stateStorage;
    }

    /// Replaces the recurrent state from a restored persistent prompt-cache
    /// snapshot.
    public func restoreFromSnapshot(_ restoredState: MLXArray) {
        self.stateStorage = restoredState;
    }
}
