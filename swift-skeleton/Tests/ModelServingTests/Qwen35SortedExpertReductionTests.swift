import JourneyCategories
import MLX
import MLXLMCommon
import ModelServingTestSupport
import Testing

@testable import ModelServing

extension MlxGpuJourneyContainer {
    @Suite(.tags(.hermeticMlxJourney))
    internal final class Qwen35SortedExpertReductionTests {
        internal init() {
            MLXMetallibLocator.overrideMetallibPathIfNecessary()
        }

        @Test(.timeLimit(.minutes(2)))
        internal func should_pass_the_bounded_execution_probe() throws -> Void {
            let reduction: Qwen35SortedExpertReduction = Qwen35SortedExpertReduction.probed(
                attributionEnabled: false)
            switch reduction.probe() {
            case .success:
                break
            case .failure(let probeError):
                Issue.record("the installed Metal device failed the probe: \(probeError)")
            }
        }

        @Test(.timeLimit(.minutes(2)), arguments: [2, 8])
        internal func should_preserve_permutation_and_weights_in_supported_and_fallback_shapes(
            expertsPerToken: Int
        ) throws -> Void {
            let reduction: Qwen35SortedExpertReduction = Qwen35SortedExpertReduction.probed(
                attributionEnabled: false)
            let assignmentCount: Int = 16 * expertsPerToken
            let sortedOutputs: MLXArray = MLX.sin(MLXArray(0..<(assignmentCount * 64))
                .asType(.float32)).reshaped(assignmentCount, 64).asType(.bfloat16)
            let inverseOrder: MLXArray = MLXArray((0..<assignmentCount).reversed().map({
                (assignmentIndex: Int) -> UInt32 in
                return UInt32(assignmentIndex)
            }))
            let scores: MLXArray = MLXArray.ones([16, expertsPerToken], dtype: .bfloat16)
                / Float(expertsPerToken)
            let actual: MLXArray = reduction(
                sortedOutputs: sortedOutputs, inverseOrder: inverseOrder, scores: scores)
            let reference: MLXArray = reduction(
                sortedOutputs: sortedOutputs, inverseOrder: inverseOrder, scores: scores,
                useCustomKernel: false)
            #expect(MLX.allClose(actual, reference, rtol: 0.03, atol: 0.01).item(Bool.self))
            #expect(actual.shape == [16, 64])
        }
    }
}
