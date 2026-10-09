import Foundation
import MLX
import MLXLMCommon
import MLXNN

internal final class Qwen35ResidentGateUpFusion: Module {
    @ModuleInfo private var weight: MLXArray
    @ModuleInfo private var scales: MLXArray
    @ModuleInfo private var biases: MLXArray
    @ModuleInfo private var downProjection: SwitchLinear
    private let groupSize: Int
    private let bits: Int
    private let attributionEnabled: Bool
    private let sortedReduction: Qwen35SortedExpertReduction?

    private init(
        weight: MLXArray, scales: MLXArray, biases: MLXArray,
        downProjection: SwitchLinear, groupSize: Int, bits: Int, attributionEnabled: Bool,
        sortedReduction: Qwen35SortedExpertReduction?
    ) {
        self._weight.wrappedValue = weight
        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases
        self._downProjection.wrappedValue = downProjection
        self.groupSize = groupSize
        self.bits = bits
        self.attributionEnabled = attributionEnabled
        self.sortedReduction = sortedReduction
        super.init()
        self.freeze()
        self.train(false)
    }

    internal static func make(
        experts: SwitchGLU, attributionEnabled: Bool = false,
        sortedReduction: Qwen35SortedExpertReduction? = nil
    ) throws -> Qwen35ResidentGateUpFusion? {
        let projections: [String: Module] = Dictionary(uniqueKeysWithValues: experts.namedModules())
        guard let gate: QuantizedSwitchLinear = projections["gate_proj"] as? QuantizedSwitchLinear,
            let up: QuantizedSwitchLinear = projections["up_proj"] as? QuantizedSwitchLinear,
            let down: SwitchLinear = projections["down_proj"] as? SwitchLinear,
            type(of: gate) == QuantizedSwitchLinear.self,
            type(of: up) == QuantizedSwitchLinear.self,
            gate.groupSize == up.groupSize, gate.bits == up.bits,
            gate.mode == .affine, up.mode == .affine
        else {
            return nil
        }
        let gateParameters: [String: MLXArray] = Dictionary(
            uniqueKeysWithValues: gate.parameters().flattened())
        let upParameters: [String: MLXArray] = Dictionary(
            uniqueKeysWithValues: up.parameters().flattened())
        guard gateParameters["bias"] == nil, upParameters["bias"] == nil,
            let gateWeight: MLXArray = gateParameters["weight"],
            let upWeight: MLXArray = upParameters["weight"],
            let gateScales: MLXArray = gateParameters["scales"],
            let upScales: MLXArray = upParameters["scales"],
            let gateBiases: MLXArray = gateParameters["biases"],
            let upBiases: MLXArray = upParameters["biases"],
            gateWeight.shape == upWeight.shape,
            gateScales.shape == upScales.shape,
            gateBiases.shape == upBiases.shape
        else {
            return nil
        }
        let fusedWeight: MLXArray = MLX.concatenated([gateWeight, upWeight], axis: 1)
        let fusedScales: MLXArray = MLX.concatenated([gateScales, upScales], axis: 1)
        let fusedBiases: MLXArray = MLX.concatenated([gateBiases, upBiases], axis: 1)
        MLX.eval(fusedWeight, fusedScales, fusedBiases)
        // Decode retains the upstream router and reduction. Views keep that
        // graph on the same payload rather than retaining a second expert copy.
        let intermediateSize: Int = gateWeight.dim(1)
        for (projectionName, projectionRange) in [
            ("gate_proj", 0..<intermediateSize),
            ("up_proj", intermediateSize..<(2 * intermediateSize))
        ] {
            try experts.update(parameters: ModuleParameters.unflattened([
                "\(projectionName).weight": fusedWeight[0..., projectionRange, 0...],
                "\(projectionName).scales": fusedScales[0..., projectionRange, 0...],
                "\(projectionName).biases": fusedBiases[0..., projectionRange, 0...]
            ]), verify: [.noUnusedKeys, .shapeMismatch])
        }
        return Qwen35ResidentGateUpFusion(
            weight: fusedWeight, scales: fusedScales, biases: fusedBiases,
            downProjection: down, groupSize: gate.groupSize, bits: gate.bits,
            attributionEnabled: attributionEnabled, sortedReduction: sortedReduction)
    }

    internal func callAsFunction(
        _ tokenRows: MLXArray, _ routes: MLXArray, scores: MLXArray
    ) -> MLXArray {
        let startedAt: ContinuousClock.Instant? = ServingPerformanceAttribution.startedOperation(
            operationName: "qwen35_fused_expert_graph", attributionEnabled: self.attributionEnabled)
        defer {
            ServingPerformanceAttribution.endedOperation(
                operationName: "qwen35_fused_expert_graph", operationStart: startedAt,
                attributionEnabled: self.attributionEnabled)
        }
        var expandedRows: MLXArray = MLX.expandedDimensions(tokenRows, axes: [-2, -3])
        var sortedRoutes: MLXArray = routes
        var inverseOrder: MLXArray = MLXArray()
        let shouldSort: Bool = routes.size >= 64
        if shouldSort {
            (expandedRows, sortedRoutes, inverseOrder) = MLXLMCommon.gatherSort(
                x: expandedRows, indices: routes)
        }
        let gateUp: MLXArray = MLX.gatherQuantizedMM(
            expandedRows, self.weight, scales: self.scales, biases: self.biases,
            rhsIndices: sortedRoutes, transpose: true, groupSize: self.groupSize,
            bits: self.bits, mode: .affine, sortedIndices: shouldSort)
        let gateAndUp: [MLXArray] = MLX.split(gateUp, parts: 2, axis: -1)
        let activated: MLXArray = MLXLMCommon.compiledSiluProduct(gateAndUp[0], gateAndUp[1])
        var projected: MLXArray = self.downProjection(
            activated, sortedRoutes, sortedIndices: shouldSort)
        if shouldSort, let sortedReduction: Qwen35SortedExpertReduction = self.sortedReduction {
            return sortedReduction(
                sortedOutputs: MLX.squeezed(projected, axis: -2),
                inverseOrder: inverseOrder, scores: scores)
        }
        if shouldSort {
            projected = MLXLMCommon.scatterUnsort(
                x: projected, invOrder: inverseOrder, shape: routes.shape)
        }
        return MLXLMCommon.weightedExpertSum(MLX.squeezed(projected, axis: -2), scores)
    }
}
