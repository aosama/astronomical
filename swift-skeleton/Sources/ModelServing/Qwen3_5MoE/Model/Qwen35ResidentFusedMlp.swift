import Foundation
import MLX
import MLXNN

internal final class Qwen35ResidentFusedMlp: Module, UnaryLayer {
    @ModuleInfo private var originalMlp: Module
    @ModuleInfo private var fusedExperts: Qwen35ResidentGateUpFusion
    private let originalForward: any UnaryLayer
    private let router: Linear
    private let sharedExpert: any UnaryLayer
    private let sharedExpertGate: Linear
    private let topK: Int
    private let normalize: Bool
    private let attributionEnabled: Bool

    internal init(
        originalMlp: Module, originalForward: any UnaryLayer,
        fusedExperts: Qwen35ResidentGateUpFusion, router: Linear,
        sharedExpert: any UnaryLayer, sharedExpertGate: Linear, topK: Int, normalize: Bool,
        attributionEnabled: Bool
    ) {
        self._originalMlp.wrappedValue = originalMlp
        self._fusedExperts.wrappedValue = fusedExperts
        self.originalForward = originalForward
        self.router = router
        self.sharedExpert = sharedExpert
        self.sharedExpertGate = sharedExpertGate
        self.topK = topK
        self.normalize = normalize
        self.attributionEnabled = attributionEnabled
        super.init()
        self.train(false)
    }

    internal func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        if hiddenStates.dim(1) == 1 {
            return self.originalForward(hiddenStates)
        }
        let startedAt: ContinuousClock.Instant? = ServingPerformanceAttribution.startedOperation(
            operationName: "qwen35_fused_mlp_prefill_graph", attributionEnabled: self.attributionEnabled)
        defer {
            ServingPerformanceAttribution.endedOperation(
                operationName: "qwen35_fused_mlp_prefill_graph", operationStart: startedAt,
                attributionEnabled: self.attributionEnabled)
        }
        let probabilities: MLXArray = MLX.softmax(
            self.router(hiddenStates), axis: -1, precise: true)
        let partitionIndex: Int = probabilities.dim(-1) - self.topK
        let routes: MLXArray = MLX.argPartition(
            probabilities, kth: partitionIndex, axis: -1)[.ellipsis, partitionIndex...]
        var scores: MLXArray = MLX.takeAlong(probabilities, routes, axis: -1)
        if self.normalize {
            scores = scores / scores.sum(axis: -1, keepDims: true)
        }
        let tokenCount: Int = hiddenStates.size / hiddenStates.dim(-1)
        let combined: MLXArray = self.fusedExperts(
            hiddenStates.reshaped(tokenCount, hiddenStates.dim(-1)),
            routes.reshaped(tokenCount, self.topK),
            scores: scores.reshaped(tokenCount, self.topK)).reshaped(hiddenStates.shape)
        let shared: MLXArray = MLX.sigmoid(self.sharedExpertGate(hiddenStates))
            * self.sharedExpert(hiddenStates)
        return combined + shared
    }
}
