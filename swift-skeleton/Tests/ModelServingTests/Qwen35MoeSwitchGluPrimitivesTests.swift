import Foundation;

import JourneyCategories;
import MLX;
import MLXLMCommon;
import MLXNN;
import ModelServingTestSupport;
import Testing;

@testable import ModelServing;

/**
 * Engine epic #983 track E3 substrate evidence: the upstream mlx-swift-lm
 * MoE expert primitives (SwitchGLU, SwitchLinear, QuantizedSwitchLinear)
 * route token assignments correctly through both the unsorted path below
 * the gather/sort threshold and the sorted path at or above it, survive
 * affine quantization without losing their routes, and evaluate
 * deterministically. Deterministic weights are compared against an
 * independent per-expert reference built from plain matrix multiplies —
 * never against golden-master constants.
 */
@Suite(.serialized, .tags(.hermeticMlxJourney))
final class Qwen35MoeSwitchGluPrimitivesTests {

    init() {
        signal(SIGPIPE, SIG_IGN);
        MLXMetallibLocator.overrideMetallibPathIfNecessary();
    }

    @Test
    func shouldRouteAssignmentsThroughTheUnsortedUpstreamSwitchGluPath() {
        let routingIndices: MLXArray = MLXArray(
            [Int32](repeating: 0, count: 4), [2, 2]);
        let primitiveFixture = DeterministicSwitchGluFixture(
            tokenCount: 2, inputDimensions: 4, hiddenDimensions: 8, expertCount: 3);

        let routedOutputs: MLXArray = primitiveFixture.switchGlu(
            primitiveFixture.tokenInputs, routingIndices);

        #expect(routedOutputs.shape == [2, 2, 4]);
        let maximumDifference: Float = maximumAbsoluteDifference(
            routedOutputs, primitiveFixture.referenceOutputs(routingIndices: routingIndices));
        #expect(
            maximumDifference < 1e-4,
            "the unsorted expert path must match the per-expert reference (max diff \(maximumDifference))");
    }

    @Test
    func shouldRouteAssignmentsThroughTheSortedUpstreamSwitchGluPath() {
        // 32 tokens x 2 assignments reaches the upstream gather/sort
        // threshold of 64 and exercises the sorted gather path.
        let routingIndices: MLXArray = MLXArray(
            (0..<32).map({ (tokenIndex: Int) -> Int32 in
                return Int32((tokenIndex * 2) % 3);
            }) + (0..<32).map({ (tokenIndex: Int) -> Int32 in
                return Int32((tokenIndex * 2 + 1) % 3);
            }), [32, 2]);
        let primitiveFixture = DeterministicSwitchGluFixture(
            tokenCount: 32, inputDimensions: 4, hiddenDimensions: 8, expertCount: 3);

        let routedOutputs: MLXArray = primitiveFixture.switchGlu(
            primitiveFixture.tokenInputs, routingIndices);

        #expect(routedOutputs.shape == [32, 2, 4]);
        let maximumDifference: Float = maximumAbsoluteDifference(
            routedOutputs, primitiveFixture.referenceOutputs(routingIndices: routingIndices));
        #expect(
            maximumDifference < 1e-4,
            "the sorted expert path must match the per-expert reference (max diff \(maximumDifference))");
    }

    @Test
    func shouldQuantizeSwitchGluExpertsWhilePreservingRoutes() {
        let routingIndices: MLXArray = MLXArray(
            [Int32](repeating: 0, count: 16), [8, 2]);
        let primitiveFixture = DeterministicSwitchGluFixture(
            tokenCount: 8, inputDimensions: 32, hiddenDimensions: 64, expertCount: 3);

        let floatOutputs: MLXArray = primitiveFixture.switchGlu(
            primitiveFixture.tokenInputs, routingIndices);
        MLX.eval(floatOutputs);

        MLXNN.quantize(model: primitiveFixture.switchGlu) { (_: String, _: Module) -> (
            groupSize: Int, bits: Int, mode: QuantizationMode
        )? in
            return (groupSize: 32, bits: 4, mode: QuantizationMode.affine);
        };

        let quantizedParameterNames: Array<String> = primitiveFixture.switchGlu.parameters()
            .flattened().map({ (parameterEntry: (String, MLXArray)) -> String in
                return parameterEntry.0;
            });
        #expect(
            quantizedParameterNames.contains(where: { (parameterName: String) -> Bool in
                return parameterName.hasSuffix("gate_proj.scales");
            }),
            "affine quantization must replace the expert projections with quantized modules");

        let quantizedOutputs: MLXArray = primitiveFixture.switchGlu(
            primitiveFixture.tokenInputs, routingIndices);
        let maximumDifference: Float = maximumAbsoluteDifference(
            quantizedOutputs, floatOutputs);
        #expect(
            maximumDifference < 0.75,
            "4-bit group-32 affine quantization must stay numerically near the float route while a corrupted route would be off by orders more (max diff \(maximumDifference))");
    }

    @Test
    func shouldProduceIdenticalOutputsForRepeatedIdenticalRoutes() {
        let routingIndices: MLXArray = MLXArray(
            [Int32](repeating: 0, count: 16), [8, 2]);
        let primitiveFixture = DeterministicSwitchGluFixture(
            tokenCount: 8, inputDimensions: 4, hiddenDimensions: 8, expertCount: 3);

        let firstOutputs: MLXArray = primitiveFixture.switchGlu(
            primitiveFixture.tokenInputs, routingIndices);
        let secondOutputs: MLXArray = primitiveFixture.switchGlu(
            primitiveFixture.tokenInputs, routingIndices);

        #expect(maximumAbsoluteDifference(firstOutputs, secondOutputs) == 0);
    }

    /// Maximum absolute elementwise difference between two equally shaped
    /// arrays, fully evaluated.
    private func maximumAbsoluteDifference(_ leftOutputs: MLXArray, _ rightOutputs: MLXArray) -> Float {
        let differenceArray: MLXArray = MLX.max(MLX.abs(leftOutputs - rightOutputs));
        return differenceArray.asArray(Float.self)[0];
    }
}

/// One SwitchGLU whose expert weights are deterministic constructed values,
/// together with an independent per-expert reference computation built from
/// plain matrix multiplies so the comparison never reuses upstream code.
private final class DeterministicSwitchGluFixture {

    let switchGlu: SwitchGLU;
    let tokenInputs: MLXArray;
    private let gateWeights: MLXArray;
    private let upWeights: MLXArray;
    private let downWeights: MLXArray;

    init(
        tokenCount: Int,
        inputDimensions: Int,
        hiddenDimensions: Int,
        expertCount: Int
    ) {
        self.switchGlu = SwitchGLU(
            inputDims: inputDimensions, hiddenDims: hiddenDimensions,
            numExperts: expertCount, bias: false);
        self.gateWeights = DeterministicSwitchGluFixture.constructedWeights(
            leadingDimensions: [expertCount, hiddenDimensions], trailingDimensions: inputDimensions,
            salt: 1);
        self.upWeights = DeterministicSwitchGluFixture.constructedWeights(
            leadingDimensions: [expertCount, hiddenDimensions], trailingDimensions: inputDimensions,
            salt: 2);
        self.downWeights = DeterministicSwitchGluFixture.constructedWeights(
            leadingDimensions: [expertCount, inputDimensions], trailingDimensions: hiddenDimensions,
            salt: 3);

        let initializedInputs: Array<Float> = (0..<(tokenCount * inputDimensions)).map(
            { (flatIndex: Int) -> Float in
                return Float((flatIndex % 7) - 3) * 0.1;
            });
        self.tokenInputs = MLXArray(initializedInputs, [tokenCount, inputDimensions]);

        let boundWeights: Dictionary<String, MLXArray> = [
            "gate_proj.weight": self.gateWeights,
            "up_proj.weight": self.upWeights,
            "down_proj.weight": self.downWeights,
        ];
        try! self.switchGlu.update(
            parameters: ModuleParameters.unflattened(boundWeights), verify: [.all]);
        MLX.eval(self.switchGlu);
    }

    /// Recomputes the routed expert outputs with plain per-expert matrix
    /// multiplies: output[token, slot] = down[e] @ (silu(gate[e] @ x) * up[e] @ x).
    func referenceOutputs(routingIndices: MLXArray) -> MLXArray {
        let tokenCount: Int = tokenInputs.dim(0);
        let topKPerToken: Int = routingIndices.dim(1);
        let stackedTokenOutputs: Array<MLXArray> = (0..<tokenCount).map(
            { (tokenIndex: Int) -> MLXArray in
                let tokenInput: MLXArray = tokenInputs[tokenIndex];
                let assignmentOutputs: Array<MLXArray> = (0..<topKPerToken).map(
                    { (assignmentIndex: Int) -> MLXArray in
                        let expertIndex: Int = Int(
                            routingIndices[tokenIndex, assignmentIndex].asArray(Int32.self)[0]);
                        let gateProjection: MLXArray = matmul(tokenInput, gateWeights[expertIndex].T);
                        let upProjection: MLXArray = matmul(tokenInput, upWeights[expertIndex].T);
                        let activatedHidden: MLXArray = MLXNN.silu(gateProjection) * upProjection;
                        return matmul(activatedHidden, downWeights[expertIndex].T);
                    });
                return stacked(assignmentOutputs, axis: 0);
            });
        return stacked(stackedTokenOutputs, axis: 0);
    }

    /// Fills [e1, e2, trailing] with small deterministic values spanning
    /// negative and positive magnitudes.
    private static func constructedWeights(
        leadingDimensions: Array<Int>,
        trailingDimensions: Int,
        salt: Int
    ) -> MLXArray {
        let flatCount: Int = leadingDimensions.reduce(trailingDimensions, { (partial: Int, dimension: Int) -> Int in
            return partial * dimension;
        });
        let constructedValues: Array<Float> = (0..<flatCount).map(
            { (flatIndex: Int) -> Float in
                let columnIndex: Int = flatIndex % trailingDimensions;
                let leadingIndex: Int = flatIndex / trailingDimensions;
                return Float(((leadingIndex * 7 + columnIndex * 3 + salt) % 9) - 4) * 0.1;
            });
        return MLXArray(constructedValues, leadingDimensions + [trailingDimensions]);
    }
}
