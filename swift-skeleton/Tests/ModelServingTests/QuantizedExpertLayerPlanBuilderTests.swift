import Foundation

import Testing

import ModelServing

/// Mixed native and affine Qwen expert-page construction, port of
/// crates/model-serving/tests/qwen3_5_moe_hermetic/mixed_expert_storage.rs:
/// an affine OptiQ document does not make every sparse expert affine, and
/// some modules stay native floating-point (`weight` only). These tests
/// prove paging follows each module profile instead of rejecting native
/// experts as zero-bit affine.
@Suite
final class QuantizedExpertLayerPlanBuilderTests {

    private static let EXPERT_CAPACITY: Int = 2
    private static let OUTPUT_DIMENSION: Int = 4
    private static let INPUT_DIMENSION: Int = 64
    private static let AFFINE_BITS: Int = 6
    private static let AFFINE_GROUP_SIZE: Int = 64
    private static let NATIVE_LAYER_PREFIX: String = "language_model.model.layers.0.mlp"
    private static let AFFINE_LAYER_PREFIX: String = "language_model.model.layers.1.mlp"
    private static let MIXED_PROJECTION_LAYER_PREFIX: String = "language_model.model.layers.2.mlp"
    private static let SHARD_FILE_NAME: String = "model.safetensors"
    private static let PROJECTION_NAMES: [String] = ["gate_proj", "up_proj", "down_proj"]

    @Test
    func should_plan_native_expert_layers_from_unquantized_module_profiles() throws {
        let mixedExpertArtifact: MixedExpertArtifact = try Self.writeMixedNativeAndAffineLayers()
        defer { mixedExpertArtifact.cleanUp() }

        let nativeLayerPlan: QuantizedExpertLayerPlan = try QuantizedExpertLayerPlanBuilder.buildLayerPlan(
            modelDirectory: mixedExpertArtifact.modelDirectoryUrl,
            weightMap: mixedExpertArtifact.weightMap,
            layerPrefix: Self.NATIVE_LAYER_PREFIX,
            config: mixedExpertArtifact.config)

        #expect(nativeLayerPlan.quantizationMode == .nativeBfloat16)
        for projectionName: String in Self.PROJECTION_NAMES {
            Self.assertNativeProjection(layerPlan: nativeLayerPlan, projectionName: projectionName)
        }
    }

    @Test
    func should_plan_affine_expert_layers_from_affine_module_profiles() throws {
        let mixedExpertArtifact: MixedExpertArtifact = try Self.writeMixedNativeAndAffineLayers()
        defer { mixedExpertArtifact.cleanUp() }

        let affineLayerPlan: QuantizedExpertLayerPlan = try QuantizedExpertLayerPlanBuilder.buildLayerPlan(
            modelDirectory: mixedExpertArtifact.modelDirectoryUrl,
            weightMap: mixedExpertArtifact.weightMap,
            layerPrefix: Self.AFFINE_LAYER_PREFIX,
            config: mixedExpertArtifact.config)

        #expect(affineLayerPlan.quantizationMode == .affine)
        for projectionName: String in Self.PROJECTION_NAMES {
            Self.assertAffineProjection(layerPlan: affineLayerPlan, projectionName: projectionName)
        }
    }

    @Test
    func should_plan_mixed_native_and_affine_expert_layers_in_one_artifact() throws {
        let mixedExpertArtifact: MixedExpertArtifact = try Self.writeMixedNativeAndAffineLayers()
        defer { mixedExpertArtifact.cleanUp() }

        let nativeLayerPlan: QuantizedExpertLayerPlan = try QuantizedExpertLayerPlanBuilder.buildLayerPlan(
            modelDirectory: mixedExpertArtifact.modelDirectoryUrl,
            weightMap: mixedExpertArtifact.weightMap,
            layerPrefix: Self.NATIVE_LAYER_PREFIX,
            config: mixedExpertArtifact.config)
        let affineLayerPlan: QuantizedExpertLayerPlan = try QuantizedExpertLayerPlanBuilder.buildLayerPlan(
            modelDirectory: mixedExpertArtifact.modelDirectoryUrl,
            weightMap: mixedExpertArtifact.weightMap,
            layerPrefix: Self.AFFINE_LAYER_PREFIX,
            config: mixedExpertArtifact.config)

        for projectionName: String in Self.PROJECTION_NAMES {
            Self.assertNativeProjection(layerPlan: nativeLayerPlan, projectionName: projectionName)
            Self.assertAffineProjection(layerPlan: affineLayerPlan, projectionName: projectionName)
        }
    }

    @Test
    func should_plan_mixed_native_and_affine_projections_in_one_layer() throws {
        let mixedProjectionArtifact: MixedExpertArtifact = try Self.writeMixedProjectionsInOneLayer()
        defer { mixedProjectionArtifact.cleanUp() }

        let mixedLayerPlan: QuantizedExpertLayerPlan = try QuantizedExpertLayerPlanBuilder.buildLayerPlan(
            modelDirectory: mixedProjectionArtifact.modelDirectoryUrl,
            weightMap: mixedProjectionArtifact.weightMap,
            layerPrefix: Self.MIXED_PROJECTION_LAYER_PREFIX,
            config: mixedProjectionArtifact.config)

        Self.assertNativeProjection(layerPlan: mixedLayerPlan, projectionName: "gate_proj")
        Self.assertAffineProjection(layerPlan: mixedLayerPlan, projectionName: "up_proj")
        Self.assertAffineProjection(layerPlan: mixedLayerPlan, projectionName: "down_proj")
    }

    @Test
    func should_reject_affine_profiles_when_companion_tensors_are_absent() throws {
        let affineProfileWithoutCompanions: MixedExpertArtifact = try Self.writeWeightOnlyTensorsWithoutUnquantizedResolve()
        defer { affineProfileWithoutCompanions.cleanUp() }

        do {
            _ = try QuantizedExpertLayerPlanBuilder.buildLayerPlan(
                modelDirectory: affineProfileWithoutCompanions.modelDirectoryUrl,
                weightMap: affineProfileWithoutCompanions.weightMap,
                layerPrefix: Self.NATIVE_LAYER_PREFIX,
                config: affineProfileWithoutCompanions.config)
            Issue.record("an affine profile without scales and biases must fail closed")
        } catch let pagingError as ExpertPagingError {
            guard case .manifestValidationFailure(let failureDescription) = pagingError else {
                Issue.record("expected a manifest validation failure, found \(pagingError)")
                return
            }
            #expect(failureDescription.contains("missing shard-index entry"))
        }
    }

    private static func assertNativeProjection(
        layerPlan: QuantizedExpertLayerPlan, projectionName: String
    ) -> Void {
        #expect(layerPlan.quantizationModeForProjection(projectionName: projectionName) == .nativeBfloat16)
        #expect(projectionParameterNames(layerPlan: layerPlan, projectionName: projectionName) == ["weight"])
    }

    private static func assertAffineProjection(
        layerPlan: QuantizedExpertLayerPlan, projectionName: String
    ) -> Void {
        #expect(layerPlan.quantizationModeForProjection(projectionName: projectionName) == .affine)
        #expect(
            projectionParameterNames(layerPlan: layerPlan, projectionName: projectionName)
                == ["weight", "scales", "biases"])
    }

    private static func projectionParameterNames(
        layerPlan: QuantizedExpertLayerPlan, projectionName: String
    ) -> [String] {
        return layerPlan.tensorSources
            .filter({ (tensorSource: QuantizedTensorSource) -> Bool in
                return tensorSource.projectionName == projectionName
            })
            .map({ (tensorSource: QuantizedTensorSource) -> String in tensorSource.parameterName })
    }

    private enum LayerStorageSpec {
        case nativeLayer(layerPrefix: String)
        case affineLayer(layerPrefix: String)
        case affineWeightWithoutCompanions(layerPrefix: String)
        case mixedProjections(layerPrefix: String, nativeProjectionNames: [String])
    }

    private final class MixedExpertArtifact {
        let modelDirectoryUrl: URL
        let config: Qwen3_5Config
        let weightMap: [String: String]

        init(modelDirectoryUrl: URL, config: Qwen3_5Config, weightMap: [String: String]) {
            self.modelDirectoryUrl = modelDirectoryUrl
            self.config = config
            self.weightMap = weightMap
        }

        func cleanUp() -> Void {
            try? FileManager.default.removeItem(at: self.modelDirectoryUrl)
        }
    }

    private struct ShardTensor {
        let tensorName: String
        let dtypeName: String
        let shape: [Int]
        let payloadByteCount: Int
    }

    private static func writeMixedNativeAndAffineLayers() throws -> MixedExpertArtifact {
        return try writeExpertArtifact(
            layers: [
                .nativeLayer(layerPrefix: NATIVE_LAYER_PREFIX),
                .affineLayer(layerPrefix: AFFINE_LAYER_PREFIX),
            ],
            resolveUnquantizedModules: true)
    }

    private static func writeMixedProjectionsInOneLayer() throws -> MixedExpertArtifact {
        return try writeExpertArtifact(
            layers: [
                .mixedProjections(
                    layerPrefix: MIXED_PROJECTION_LAYER_PREFIX,
                    nativeProjectionNames: ["gate_proj"]),
            ],
            resolveUnquantizedModules: true)
    }

    private static func writeWeightOnlyTensorsWithoutUnquantizedResolve() throws -> MixedExpertArtifact {
        return try writeExpertArtifact(
            layers: [
                .affineWeightWithoutCompanions(layerPrefix: NATIVE_LAYER_PREFIX),
            ],
            resolveUnquantizedModules: false)
    }

    private static func writeExpertArtifact(
        layers: [LayerStorageSpec], resolveUnquantizedModules: Bool
    ) throws -> MixedExpertArtifact {
        let modelDirectoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("layer-plan-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: modelDirectoryUrl, withIntermediateDirectories: true)
        var tensors: [ShardTensor] = []
        var weightMap: [String: String] = [:]
        for layer: LayerStorageSpec in layers {
            switch layer {
            case .nativeLayer(let layerPrefix):
                appendNativeLayer(tensors: &tensors, weightMap: &weightMap, layerPrefix: layerPrefix)
            case .affineLayer(let layerPrefix):
                appendAffineLayer(tensors: &tensors, weightMap: &weightMap, layerPrefix: layerPrefix)
            case .affineWeightWithoutCompanions(let layerPrefix):
                appendAffineWeightWithoutCompanions(
                    tensors: &tensors, weightMap: &weightMap, layerPrefix: layerPrefix)
            case .mixedProjections(let layerPrefix, let nativeProjectionNames):
                for projectionName: String in PROJECTION_NAMES {
                    if nativeProjectionNames.contains(projectionName) {
                        appendNativeProjection(
                            tensors: &tensors, weightMap: &weightMap,
                            layerPrefix: layerPrefix, projectionName: projectionName)
                    } else {
                        appendAffineProjection(
                            tensors: &tensors, weightMap: &weightMap,
                            layerPrefix: layerPrefix, projectionName: projectionName)
                    }
                }
            }
        }
        try writeSafetensorsFile(
            fileUrl: modelDirectoryUrl.appendingPathComponent(SHARD_FILE_NAME), tensors: tensors)
        var config: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(
            configBytes: try Qwen3_5MoeConfigFixtures.frozenOrnith10ConfigBytes())
        if resolveUnquantizedModules {
            config.resolveUnquantizedModulesFromShardIndex(shardTensorNames: Set(weightMap.keys))
        }
        return MixedExpertArtifact(
            modelDirectoryUrl: modelDirectoryUrl, config: config, weightMap: weightMap)
    }

    private static func appendNativeLayer(
        tensors: inout [ShardTensor], weightMap: inout [String: String], layerPrefix: String
    ) -> Void {
        for projectionName: String in PROJECTION_NAMES {
            appendNativeProjection(
                tensors: &tensors, weightMap: &weightMap,
                layerPrefix: layerPrefix, projectionName: projectionName)
        }
    }

    private static func appendAffineLayer(
        tensors: inout [ShardTensor], weightMap: inout [String: String], layerPrefix: String
    ) -> Void {
        for projectionName: String in PROJECTION_NAMES {
            appendAffineProjection(
                tensors: &tensors, weightMap: &weightMap,
                layerPrefix: layerPrefix, projectionName: projectionName)
        }
    }

    private static func appendAffineWeightWithoutCompanions(
        tensors: inout [ShardTensor], weightMap: inout [String: String], layerPrefix: String
    ) -> Void {
        let packedWidth: Int = INPUT_DIMENSION * AFFINE_BITS / 32
        for projectionName: String in PROJECTION_NAMES {
            let weightName: String = "\(layerPrefix).switch_mlp.\(projectionName).weight"
            weightMap[weightName] = SHARD_FILE_NAME
            tensors.append(ShardTensor(
                tensorName: weightName,
                dtypeName: "U32",
                shape: [EXPERT_CAPACITY, OUTPUT_DIMENSION, packedWidth],
                payloadByteCount: EXPERT_CAPACITY * OUTPUT_DIMENSION * packedWidth * 4))
        }
    }

    private static func appendNativeProjection(
        tensors: inout [ShardTensor], weightMap: inout [String: String],
        layerPrefix: String, projectionName: String
    ) -> Void {
        let tensorName: String = "\(layerPrefix).switch_mlp.\(projectionName).weight"
        weightMap[tensorName] = SHARD_FILE_NAME
        tensors.append(ShardTensor(
            tensorName: tensorName,
            dtypeName: "BF16",
            shape: [EXPERT_CAPACITY, OUTPUT_DIMENSION, INPUT_DIMENSION],
            payloadByteCount: EXPERT_CAPACITY * OUTPUT_DIMENSION * INPUT_DIMENSION * 2))
    }

    private static func appendAffineProjection(
        tensors: inout [ShardTensor], weightMap: inout [String: String],
        layerPrefix: String, projectionName: String
    ) -> Void {
        let packedWidth: Int = INPUT_DIMENSION * AFFINE_BITS / 32
        let scaleWidth: Int = INPUT_DIMENSION / AFFINE_GROUP_SIZE
        let weightName: String = "\(layerPrefix).switch_mlp.\(projectionName).weight"
        let scalesName: String = "\(layerPrefix).switch_mlp.\(projectionName).scales"
        let biasesName: String = "\(layerPrefix).switch_mlp.\(projectionName).biases"
        weightMap[weightName] = SHARD_FILE_NAME
        weightMap[scalesName] = SHARD_FILE_NAME
        weightMap[biasesName] = SHARD_FILE_NAME
        tensors.append(ShardTensor(
            tensorName: weightName,
            dtypeName: "U32",
            shape: [EXPERT_CAPACITY, OUTPUT_DIMENSION, packedWidth],
            payloadByteCount: EXPERT_CAPACITY * OUTPUT_DIMENSION * packedWidth * 4))
        let companionShape: [Int] = [EXPERT_CAPACITY, OUTPUT_DIMENSION, scaleWidth]
        let companionPayloadByteCount: Int = EXPERT_CAPACITY * OUTPUT_DIMENSION * scaleWidth * 2
        tensors.append(ShardTensor(
            tensorName: scalesName,
            dtypeName: "BF16",
            shape: companionShape,
            payloadByteCount: companionPayloadByteCount))
        tensors.append(ShardTensor(
            tensorName: biasesName,
            dtypeName: "BF16",
            shape: companionShape,
            payloadByteCount: companionPayloadByteCount))
    }

    private static func writeSafetensorsFile(fileUrl: URL, tensors: [ShardTensor]) throws -> Void {
        var payloadBytes: Data = Data()
        var headerEntries: [String] = []
        for tensor: ShardTensor in tensors {
            let payloadStartByte: Int = payloadBytes.count
            payloadBytes.append(Data(count: tensor.payloadByteCount))
            let payloadEndByte: Int = payloadBytes.count
            let shapeText: String = tensor.shape
                .map({ (dimension: Int) -> String in String(dimension) })
                .joined(separator: ",")
            headerEntries.append(
                "\"\(tensor.tensorName)\": {\"dtype\": \"\(tensor.dtypeName)\", "
                    + "\"shape\": [\(shapeText)], "
                    + "\"data_offsets\": [\(payloadStartByte), \(payloadEndByte)]}")
        }
        let headerJsonText: String = "{" + headerEntries.joined(separator: ",") + "}"
        let headerBytes: Data = Data(headerJsonText.utf8)
        var fileBytes: Data = Data()
        withUnsafeBytes(of: UInt64(headerBytes.count).littleEndian, { (lengthBuffer: UnsafeRawBufferPointer) -> Void in
            fileBytes.append(contentsOf: lengthBuffer)
        })
        fileBytes.append(headerBytes)
        fileBytes.append(payloadBytes)
        try fileBytes.write(to: fileUrl)
    }
}
