import Foundation
import JourneyCategories
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import ModelServingTestSupport
import Testing

@testable import ModelServing

extension MlxGpuJourneyContainer {
    @Suite(.tags(.hermeticMlxJourney))
    internal final class Qwen35ResidentGateUpFusionTests {
        internal init() {
            MLXMetallibLocator.overrideMetallibPathIfNecessary()
        }

        @Test(.timeLimit(.minutes(2)), arguments: [1, 32])
        internal func should_preserve_sorted_and_unsorted_quantized_expert_outputs(
            tokenCount: Int
        ) throws -> Void {
            let experts = SwitchGLU(inputDims: 64, hiddenDims: 64, numExperts: 4)
            MLXNN.quantize(model: experts, groupSize: 32, bits: 4)
            let tokenRows: MLXArray = MLX.sin(MLXArray(0..<(tokenCount * 64))
                .asType(.float32)).reshaped(tokenCount, 64).asType(.bfloat16)
            let routes: MLXArray = MLXArray((0..<(tokenCount * 2)).map({
                (assignmentIndex: Int) -> UInt32 in
                return UInt32(assignmentIndex % 4)
            }), [tokenCount, 2])
            let scores: MLXArray = MLXArray.ones([tokenCount, 2], dtype: .bfloat16) * 0.5
            let expected: MLXArray = MLXLMCommon.weightedExpertSum(
                experts(tokenRows, routes), scores)
            MLX.eval(expected)
            let fused: Qwen35ResidentGateUpFusion = try #require(
                try Qwen35ResidentGateUpFusion.make(experts: experts))
            let actual: MLXArray = fused(tokenRows, routes, scores: scores)
            MLX.eval(actual)
            #expect(actual.shape == expected.shape)
            #expect(MLX.allClose(actual, expected, rtol: 0.01, atol: 0.002).item(Bool.self))
            let retainedOriginal: MLXArray = MLXLMCommon.weightedExpertSum(
                experts(tokenRows, routes), scores)
            #expect(MLX.allClose(retainedOriginal, expected, rtol: 0, atol: 0).item(Bool.self))
        }

        @Test(.timeLimit(.minutes(2)))
        internal func should_install_fusion_without_changing_the_moe_block_output() throws -> Void {
            let (fixtureDirectory, layout): (URL, TinyMoeArtifactFixture.SynthesizedLayout) =
                try TinyMoeArtifactFixture.writeModelDirectory()
            defer { try? FileManager.default.removeItem(at: fixtureDirectory) }
            let configuration: Qwen35Configuration = try JSONDecoder().decode(
                Qwen35Configuration.self, from: Data(layout.configBytes))
            let repositoryConfiguration: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(
                configBytes: layout.configBytes)
            let model = Qwen35MoEModel(configuration)
            MLXNN.quantize(model: model, groupSize: 32, bits: 4)
            model.train(false)
            let originalMlp: any UnaryLayer = try #require(model.namedModules().first(where: {
                (modulePath: String, _: Module) -> Bool in
                return modulePath.hasSuffix(".mlp")
            })?.1 as? any UnaryLayer)
            let hiddenStates: MLXArray = MLX.sin(MLXArray(0..<(32 * 64))
                .asType(.float32)).reshaped(1, 32, 64).asType(.bfloat16)
            let expected: MLXArray = originalMlp(hiddenStates)
            MLX.eval(expected)
            try Qwen35ResidentFusionInstall.install(
                model: model, configuration: repositoryConfiguration, attributionEnabled: false)
            let installed: Qwen35ResidentFusedMlp = try #require(model.namedModules()
                .compactMap({ (_: String, module: Module) -> Qwen35ResidentFusedMlp? in
                    return module as? Qwen35ResidentFusedMlp
                }).first)
            let actual: MLXArray = installed(hiddenStates)
            #expect(MLX.allClose(actual, expected, rtol: 0.01, atol: 0.002).item(Bool.self))
            let decodeStates: MLXArray = hiddenStates[0..., 0..<1, 0...]
            #expect(MLX.arrayEqual(installed(decodeStates), originalMlp(decodeStates)).item(Bool.self))
        }

        @Test(.timeLimit(.minutes(2)))
        internal func should_preserve_incompatible_siblings_during_partial_layer_fusion() throws -> Void {
            let (fixtureDirectory, layout): (URL, TinyMoeArtifactFixture.SynthesizedLayout) =
                try TinyMoeArtifactFixture.writeModelDirectory()
            defer { try? FileManager.default.removeItem(at: fixtureDirectory) }
            let configurationText: String = String(decoding: layout.configBytes, as: UTF8.self)
                .replacingOccurrences(of: "\"num_hidden_layers\": 1", with: "\"num_hidden_layers\": 3")
                .replacingOccurrences(
                    of: "\"layer_types\": [\"full_attention\"]",
                    with: "\"layer_types\": [\"full_attention\", \"full_attention\", \"full_attention\"]")
            let configurationBytes: [UInt8] = Array(configurationText.utf8)
            let configuration: Qwen35Configuration = try JSONDecoder().decode(
                Qwen35Configuration.self, from: Data(configurationBytes))
            let repositoryConfiguration: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(
                configBytes: configurationBytes)
            let model = Qwen35MoEModel(configuration)
            MLXNN.quantize(model: model, filter: {
                (modulePath: String, _: Module) -> (groupSize: Int, bits: Int, mode: QuantizationMode)? in
                if modulePath.contains("layers.0.mlp.switch_mlp.gate_proj") {
                    return nil
                }
                return (32, modulePath.contains("layers.2.mlp.switch_mlp.up_proj") ? 8 : 4, .affine)
            })
            model.train(false)
            let originalMlps: [(String, Module)] = model.namedModules().filter({
                (modulePath: String, _: Module) -> Bool in
                return modulePath.hasSuffix(".mlp")
            })
            #expect(originalMlps.count == 3)
            try Qwen35ResidentFusionInstall.install(
                model: model, configuration: repositoryConfiguration, attributionEnabled: false)
            let installedMlps: [String: Module] = Dictionary(uniqueKeysWithValues: model.namedModules())
            for (modulePath, originalMlp) in originalMlps {
                let installedMlp: Module = try #require(installedMlps[modulePath])
                if modulePath.contains("layers.1.") {
                    #expect(installedMlp is Qwen35ResidentFusedMlp)
                } else {
                    #expect(installedMlp === originalMlp)
                }
            }
        }

        @Test(.timeLimit(.minutes(2)))
        internal func should_keep_mixed_projection_profiles_on_the_original_path() throws -> Void {
            let experts = SwitchGLU(inputDims: 64, hiddenDims: 64, numExperts: 4)
            MLXNN.quantize(model: experts, filter: {
                (modulePath: String, _: Module) -> (groupSize: Int, bits: Int, mode: QuantizationMode)? in
                return (32, modulePath == "gate_proj" ? 4 : 8, .affine)
            })
            let originalParameters: [String: MLXArray] = Dictionary(
                uniqueKeysWithValues: experts.parameters().flattened())
            #expect(try Qwen35ResidentGateUpFusion.make(experts: experts) == nil)
            let retainedParameters: [String: MLXArray] = Dictionary(
                uniqueKeysWithValues: experts.parameters().flattened())
            for (parameterName, originalParameter) in originalParameters {
                let retainedParameter: MLXArray = try #require(retainedParameters[parameterName])
                #expect(MLX.arrayEqual(originalParameter, retainedParameter).item(Bool.self))
            }
        }
    }
}
