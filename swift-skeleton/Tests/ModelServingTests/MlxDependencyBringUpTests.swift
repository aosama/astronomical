import Foundation;

import MLX;
import MLXNN;
import MLXLMCommon;
import MLXLLM;
import ModelServingTestSupport;
import Testing;
import JourneyCategories;

@testable import ModelServing;

/**
 * Engine epic #983 track E0 bring-up evidence: the pinned mlx-swift and
 * mlx-swift-lm dependencies compile, the MLX array runtime executes on this
 * machine, MLXNN modules evaluate, and the upstream language-model toolkit
 * links into the serving module. No model downloads and no long GPU
 * journeys — tiny arrays only.
 */
@Suite(.serialized, .tags(.hermeticMlxJourney))
final class MlxDependencyBringUpTests {

    init() {
        signal(SIGPIPE, SIG_IGN);
        MLXMetallibLocator.overrideMetallibPathIfNecessary();
    }

    @Test
    func should_evaluate_mlx_array_math_and_round_trip_through_eval() {
        let leftMatrix: MLXArray = MLXArray(Array<Float>(repeating: 1, count: 8), [2, 4]);
        let rightMatrix: MLXArray = MLXArray(Array<Float>(repeating: 2, count: 8), [4, 2]);
        let product: MLXArray = matmul(leftMatrix, rightMatrix);
        let evaluatedProduct: Array<Float> = product.asArray(Float.self);
        #expect(evaluatedProduct == Array<Float>(repeating: 8, count: 4));
    }

    @Test
    func should_evaluate_an_mlxnn_module_forward_pass() {
        // Deterministic weights: row r filled with value r.
        let weightMatrix: Array<Float> = (0..<12).map({ (weightIndex: Int) -> Float in
            return Float(weightIndex / 4);
        });
        let projectionLayer: Linear = Linear(weight: MLXArray(weightMatrix, [3, 4]), bias: nil);
        let inputBatch: MLXArray = MLXArray(Array<Float>(repeating: 1, count: 8), [2, 4]);
        let outputBatch: MLXArray = projectionLayer(inputBatch);
        let evaluatedOutput: Array<Float> = outputBatch.asArray(Float.self);
        // Row j of the weight holds the value j, so every all-ones input
        // row projects to the per-row sums [0, 4, 8].
        #expect(evaluatedOutput == [0, 4, 8, 0, 4, 8]);
        #expect(outputBatch.shape == [2, 3]);
    }

    @Test
    func should_link_the_mlxlm_common_toolkit_into_the_serving_module() {
        // The upstream types the engine adapter will build on resolve and
        // construct without any network or checkpoint access.
        let cacheConfiguration: KVCacheConfiguration = KVCacheConfiguration();
        let rotaryLayer: RoPE = RoPE(dimensions: 4, traditional: false, base: 10_000, scale: 1);
        let rotaryOutput: MLXArray = rotaryLayer(MLXArray(Array<Float>(repeating: 0.5, count: 16), [1, 2, 2, 4]));
        #expect(rotaryOutput.shape == [1, 2, 2, 4]);
        _ = cacheConfiguration;
    }
}
