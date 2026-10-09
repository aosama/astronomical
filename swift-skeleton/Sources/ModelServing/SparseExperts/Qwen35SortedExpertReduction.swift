import Foundation
import MLX
import MLXFast
import MLXLMCommon

internal final class Qwen35SortedExpertReduction: CustomMetalKernelProbe {
    private let kernel: MLXFast.MLXFastKernel
    private var isSupported: Bool
    internal var family: CustomMetalKernelFamily {
        return .sortedExpertWeightedSum
    }

    private init() {
        self.kernel = MLXFast.metalKernel(
            name: "qwen35_sorted_expert_reduction",
            inputNames: ["sorted_outputs", "inverse_order", "weights"],
            outputNames: ["output"],
            source: """
                uint feature = thread_position_in_grid.x;
                uint token = thread_position_in_grid.y;
                T accumulator = (T)0;
                const uint assignment_base = token * (uint)K;
                for (uint slot = 0; slot < (uint)K; ++slot) {
                    const uint assignment = assignment_base + slot;
                    const uint sorted_row = (uint)inverse_order[assignment];
                    const T weighted = (T)(
                        (float)sorted_outputs[sorted_row * threads_per_grid.x + feature]
                        * (float)weights[assignment]);
                    accumulator = accumulator + weighted;
                }
                output[token * threads_per_grid.x + feature] = accumulator;
                """,
            ensureRowContiguous: true)
        self.isSupported = false
    }

    internal static func probed(attributionEnabled: Bool) -> Qwen35SortedExpertReduction {
        let reduction = Qwen35SortedExpertReduction()
        let capabilities: WorkerKernelCapabilities = WorkerKernelCapabilities.probeCustomKernels(
            [reduction], attributionEnabled ? PerformanceAttribution.enabled() : PerformanceAttribution.disabled())
        reduction.isSupported = capabilities.isCustomKernelSupported(.sortedExpertWeightedSum)
        if reduction.isSupported == false {
            FileHandle.standardError.write(Data(
                "sorted expert reduction uses public MLX fallback: \(capabilities.verdict(.sortedExpertWeightedSum))\n".utf8))
        }
        return reduction
    }

    internal func probe() -> Result<Void, KernelCapabilityError> {
        do {
            try MLX.withError({
                () throws -> Void in
                let sortedOutputs: MLXArray = MLXArray((0..<(64 * 64)).map({
                    (flatIndex: Int) -> Float in
                    return Float(((flatIndex / 64) * 3 + flatIndex % 64) % 17 - 8)
                }), [64, 64]).asType(.bfloat16)
                let inverseOrder: MLXArray = MLXArray((0..<64).reversed().map({
                    (assignmentIndex: Int) -> UInt32 in
                    return UInt32(assignmentIndex)
                }))
                let scores: MLXArray = MLXArray((0..<64).map({
                    (assignmentIndex: Int) -> Float in
                    return Float(assignmentIndex % 3 + 1) * 0.125
                }), [8, 8]).asType(.bfloat16)
                let actual: MLXArray = self.direct(
                    sortedOutputs: sortedOutputs, inverseOrder: inverseOrder, scores: scores)
                let expected: MLXArray = Self.reference(
                    sortedOutputs: sortedOutputs, inverseOrder: inverseOrder, scores: scores)
                try MLX.checkedEval(actual, expected)
                if MLX.arrayEqual(actual, expected).item(Bool.self) == false {
                    throw KernelCapabilityError.outputMismatch(
                        description: "sorted expert reduction disagrees with the public MLX reference")
                }
            })
            return .success(())
        } catch let capabilityError as KernelCapabilityError {
            return .failure(capabilityError)
        } catch {
            return .failure(.execution(description: String(describing: error)))
        }
    }

    internal func callAsFunction(
        sortedOutputs: MLXArray, inverseOrder: MLXArray, scores: MLXArray,
        useCustomKernel: Bool = true
    ) -> MLXArray {
        guard useCustomKernel, self.isSupported,
            sortedOutputs.ndim == 2, sortedOutputs.dtype == .bfloat16,
            sortedOutputs.dim(1).isMultiple(of: 64),
            scores.ndim == 2, scores.dim(1) == 8,
            scores.dtype == .bfloat16, scores.size >= 64,
            inverseOrder.ndim == 1, inverseOrder.dtype == .uint32,
            inverseOrder.size == scores.size, sortedOutputs.dim(0) == scores.size
        else {
            return Self.reference(
                sortedOutputs: sortedOutputs, inverseOrder: inverseOrder, scores: scores)
        }
        return self.direct(sortedOutputs: sortedOutputs, inverseOrder: inverseOrder, scores: scores)
    }

    private func direct(
        sortedOutputs: MLXArray, inverseOrder: MLXArray, scores: MLXArray
    ) -> MLXArray {
        let hiddenSize: Int = sortedOutputs.dim(1)
        let tokenCount: Int = scores.dim(0)
        return self.kernel(
            [sortedOutputs, inverseOrder, scores],
            template: [("T", sortedOutputs.dtype), ("K", 8)],
            grid: (hiddenSize, tokenCount, 1), threadGroup: (64, 4, 1),
            outputShapes: [[tokenCount, hiddenSize]], outputDTypes: [.bfloat16])[0]
    }

    private static func reference(
        sortedOutputs: MLXArray, inverseOrder: MLXArray, scores: MLXArray
    ) -> MLXArray {
        let restored: MLXArray = sortedOutputs[inverseOrder].reshaped(
            scores.dim(0), scores.dim(1), sortedOutputs.dim(1))
        return MLXLMCommon.weightedExpertSum(restored, scores)
    }
}
