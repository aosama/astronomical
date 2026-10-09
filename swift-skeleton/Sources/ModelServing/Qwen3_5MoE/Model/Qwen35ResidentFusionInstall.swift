import Foundation
import MLXLLM
import MLXLMCommon
import MLXNN

internal enum Qwen35ResidentFusionInstall {
    internal static func install(
        model: Qwen35Model, configuration: Qwen3_5Config, attributionEnabled: Bool
    ) throws -> Void {
        let startedAt: ContinuousClock.Instant? = ServingPerformanceAttribution.startedOperation(
            operationName: "qwen35_resident_gate_up_fusion", attributionEnabled: attributionEnabled)
        defer {
            ServingPerformanceAttribution.endedOperation(
                operationName: "qwen35_resident_gate_up_fusion", operationStart: startedAt,
                attributionEnabled: attributionEnabled)
        }
        var replacements: [(String, Module)] = []
        var sortedReduction: Qwen35SortedExpertReduction?
        for (modulePath, module) in model.namedModules() {
            guard modulePath.hasSuffix(".mlp"),
                let originalForward: any UnaryLayer = module as? any UnaryLayer
            else {
                continue
            }
            let children: [String: Module] = Dictionary(uniqueKeysWithValues: module.namedModules())
            guard let experts: SwitchGLU = children["switch_mlp"] as? SwitchGLU else {
                continue
            }
            if sortedReduction == nil {
                sortedReduction = Qwen35SortedExpertReduction.probed(
                    attributionEnabled: attributionEnabled)
            }
            guard let router: Linear = children["gate"] as? Linear,
                let sharedExpert: any UnaryLayer = children["shared_expert"] as? any UnaryLayer,
                let sharedExpertGate: Linear = children["shared_expert_gate"] as? Linear
            else {
                throw InferenceEngineError.modelLoad(reason: "the resident MoE block has incomplete routing modules")
            }
            if let fusedExperts: Qwen35ResidentGateUpFusion = try Qwen35ResidentGateUpFusion.make(
                experts: experts, attributionEnabled: attributionEnabled,
                sortedReduction: sortedReduction) {
                replacements.append((modulePath, Qwen35ResidentFusedMlp(
                    originalMlp: module, originalForward: originalForward, fusedExperts: fusedExperts,
                    router: router, sharedExpert: sharedExpert, sharedExpertGate: sharedExpertGate,
                    topK: Int(configuration.expertsPerToken()),
                    normalize: configuration.normalizesTopKProbabilities(),
                    attributionEnabled: attributionEnabled)))
            } else {
                // Array traversal needs every sibling slot, including layers
                // whose declared projection profiles remain separate.
                replacements.append((modulePath, module))
            }
        }
        if replacements.isEmpty == false {
            do {
                try model.update(modules: ModuleChildren.unflattened(replacements), verify: [.all])
            } catch {
                throw InferenceEngineError.modelLoad(
                    reason: "resident expert fusion installation failed: \(error)")
            }
        }
    }
}
