import Foundation;

import MLX;

/// One hybrid layer's rolling convolution state, port of the Rust
/// `ConvolutionState` owner. The state is one tensor snapshot at a time;
/// prompts replace it wholesale rather than appending.
public final class ConvolutionState {

    private var stateStorage: MLXArray?;

    public init() {
        self.stateStorage = nil;
    }

    public func state() -> MLXArray? {
        return self.stateStorage;
    }

    /// Replaces the convolution rolling buffer from a restored persistent
    /// prompt-cache snapshot.
    public func restoreFromSnapshot(_ restoredState: MLXArray) {
        self.stateStorage = restoredState;
    }
}
