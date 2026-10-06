import XCTest;

import Foundation;

import MLX;
import MLXNN;
import MLXLMCommon;
import MLXLLM;

@testable import ModelServing;

/**
 * Engine epic #983 track E0 bring-up evidence: the pinned mlx-swift and
 * mlx-swift-lm dependencies compile, the MLX array runtime executes on this
 * machine, MLXNN modules evaluate, and the upstream language-model toolkit
 * links into the serving module. No model downloads and no GPU journeys —
 * fully hermetic.
 */
final class MlxDependencyBringUpTests: XCTestCase {

    func testMlxArrayMathExecutesAndRoundTripsThroughEval() {
        let leftMatrix: MLXArray = MLXArray(Array<Float>(repeating: 1, count: 8), [2, 4]);
        let rightMatrix: MLXArray = MLXArray(Array<Float>(repeating: 2, count: 8), [4, 2]);
        let product: MLXArray = matmul(leftMatrix, rightMatrix);
        let evaluatedProduct: Array<Float> = product.asArray(Float.self);
        XCTAssertEqual(evaluatedProduct, Array<Float>(repeating: 8, count: 4));
    }

    func testMlxnnModuleEvaluatesAForwardPass() {
        // Deterministic weights: row r filled with value r.
        let weightMatrix: Array<Float> = (0..<12).map { (weightIndex: Int) -> Float in
            return Float(weightIndex / 4);
        };
        let projectionLayer: Linear = Linear(weight: MLXArray(weightMatrix, [3, 4]), bias: nil);
        let inputBatch: MLXArray = MLXArray(Array<Float>(repeating: 1, count: 8), [2, 4]);
        let outputBatch: MLXArray = projectionLayer(inputBatch);
        let evaluatedOutput: Array<Float> = outputBatch.asArray(Float.self);
        // Row j of the weight holds the value j, so every all-ones input
        // row projects to the per-row sums [0, 4, 8].
        XCTAssertEqual(evaluatedOutput, [0, 4, 8, 0, 4, 8]);
        XCTAssertEqual(outputBatch.shape, [2, 3]);
    }

    func testMlxlmCommonToolkitLinksIntoTheServingModule() throws {
        // The upstream types the engine adapter will build on resolve and
        // construct without any network or checkpoint access.
        let cacheConfiguration = KVCacheConfiguration();
        let rotaryLayer = RoPE(dimensions: 4, traditional: false, base: 10_000, scale: 1);
        let rotaryOutput: MLXArray = rotaryLayer(MLXArray(Array<Float>(repeating: 0.5, count: 16), [1, 2, 2, 4]));
        XCTAssertEqual(rotaryOutput.shape, [1, 2, 2, 4]);
        _ = cacheConfiguration;
    }
}
