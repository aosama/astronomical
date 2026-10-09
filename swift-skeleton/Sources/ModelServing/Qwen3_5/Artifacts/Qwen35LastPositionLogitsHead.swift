import Foundation;

import MLX;
import MLXNN;

/**
 * A Linear-compatible head that projects only the final sequence position
 * when the input spans multiple tokens. This mirrors the Rust forward
 * contract (crates/model-serving/src/qwen3_5/model/forward_graph.rs):
 * intermediate prefill chunks need only cache tensors, and even the
 * terminal chunk samples just the last position, so building the full
 * [tokens, vocabulary] projection per chunk was discarded compute on the
 * order of a fifth of a 35B MoE prefill chunk. Single-token decode calls
 * pass through unchanged, and the wrapped head keeps its own quantized
 * execution, so no weights move and no numeric path changes.
 */
final class Qwen35LastPositionLogitsHead: Linear {

    @ModuleInfo var wrappedHead: Linear;

    init(_ wrappedHead: Linear) {
        self._wrappedHead.wrappedValue = wrappedHead;
        // The placeholder weight keeps this instance Linear-typed so it can
        // replace the upstream head in the module tree; the wrapped head
        // owns the real projection and its parameters.
        super.init(weight: MLXArray.zeros([1, 1]));
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        if x.dim(1) > 1 {
            return self.wrappedHead(x[0..., (x.dim(1) - 1)..., 0...]);
        }
        return self.wrappedHead(x);
    }
}
