import Foundation;

import MLX;
import MLXLMCommon;
import MLXNN;

/**
 * The engine-facing paged expert primitive: an upstream SwitchGLU whose
 * every forward first materializes the routed experts the layer is missing
 * through the paging decorator, then delegates to the untouched upstream
 * execution.
 *
 * The repository swaps this primitive into each MoE layer's
 * `switch_mlp` slot at paging setup, so the upstream sparse block keeps
 * its own router math and its own weighted-reduction selection — the
 * decorator intercepts only at the expert-primitive boundary. The upstream
 * fused sorted reduction bypasses `callAsFunction` entirely; it requires
 * exactly eight experts per token plus quantized projections, so it never
 * engages through the paged engine until a real-model interception seam
 * lands, and the engine's prefill chunk sizing stays under the upstream
 * gather/sort threshold meanwhile.
 */
public final class Qwen35MoePagedSwitchGLU: SwitchGLU {

    /// The decorator this primitive consults before every forward.
    private let pagingDecorator: Qwen35MoePagedExpertDecorator;

    /// The decoder layer index this primitive serves.
    private let decoderLayerIndex: Int;

    /// The forward context supplying the attributed input token id.
    private let forwardContext: Qwen35MoePagedForwardContext;

    /// The per-token hidden width, which is also each assignment's output
    /// width — the fallback tensor's last axis on the fault path.
    private let inputDimensionCount: Int;

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        decoderLayerIndex: Int,
        pagingDecorator: Qwen35MoePagedExpertDecorator,
        forwardContext: Qwen35MoePagedForwardContext
    ) {
        self.pagingDecorator = pagingDecorator;
        self.decoderLayerIndex = decoderLayerIndex;
        self.forwardContext = forwardContext;
        self.inputDimensionCount = inputDims;
        super.init(inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts);
    }

    /**
     * Materializes the routed-but-missing experts, then runs the upstream
     * expert execution unchanged. The upstream signature cannot throw, so a
     * failed preparation records its fault on the forward context and
     * returns a well-formed zero tensor — the engine aborts the generation
     * fail-closed before any token reaches the caller. Calling the upstream
     * execution after a failed preparation is forbidden: the shape guard
     * exists because the sorted gather reads out of bounds and the missing
     * experts would execute as zeros.
     */
    public override func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        do {
            try self.pagingDecorator.prepareLayerForForward(
                layerIndex: self.decoderLayerIndex,
                inputTokenCount: x.dim(0),
                routingIndices: indices,
                switchGlu: self,
                inputTokenId: self.forwardContext.currentInputTokenId);
        } catch {
            self.forwardContext.recordForwardFault(error);
            return MLX.zeros(
                [x.dim(0), indices.dim(1), self.inputDimensionCount], dtype: x.dtype);
        }
        return super.callAsFunction(x, indices);
    }
}
