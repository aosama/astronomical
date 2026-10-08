import Foundation;

import MLX;

import RuntimeIntegration;

/// One full-attention layer's in-memory key/value state, port of the Rust
/// `FullAttentionKeyValueState` owner family. The physical slabs may hold
/// more tokens than the logical offset because of step-bounded
/// over-allocation; the offset is the state's logical length.
public final class FullAttentionKeyValueState {

    private var keysStorage: MLXArray?;

    private var valuesStorage: MLXArray?;

    private var offsetTokensValue: Int;

    public init() {
        self.keysStorage = nil;
        self.valuesStorage = nil;
        self.offsetTokensValue = 0;
    }

    /// The number of tokens written into the K and V slabs so far.
    public var offsetTokens: Int {
        return self.offsetTokensValue;
    }

    public func keysState() -> MLXArray? {
        return self.keysStorage;
    }

    public func valuesState() -> MLXArray? {
        return self.valuesStorage;
    }

    /// Replaces the K and V storage from a restored persistent prompt-cache
    /// prefix. The slabs arrive as one final-length concatenation; the owner
    /// takes them wholesale and advances its offset to their token count.
    public func restoreFromBlocks(
        restoredKeys: MLXArray,
        restoredValues: MLXArray
    ) throws {
        let restoredKeyShape: [Int] = restoredKeys.shape;
        let slabsAreValid: Bool = restoredKeyShape.count == 4
            && restoredKeyShape == restoredValues.shape
            && restoredKeyShape[DecoderCacheStateConstants.TOKEN_AXIS] > 0;
        if slabsAreValid == false {
            throw MlxRuntimeError.runtimeOperation(
                operation: "restore full-attention key/value state",
                description: "restored K and V slabs must have identical rank-four "
                    + "nonempty shapes");
        }
        self.keysStorage = restoredKeys;
        self.valuesStorage = restoredValues;
        self.offsetTokensValue = restoredKeyShape[DecoderCacheStateConstants.TOKEN_AXIS];
    }
}

/// Vocabulary shared by the decoder-cache state owners.
enum DecoderCacheStateConstants {

    /// The sequence-position axis of a rank-four full-attention state slab
    /// shaped [batch, heads, tokens, head dimension].
    static let TOKEN_AXIS: Int = 2;
}
