import Foundation;
import IpcProtocol;

/// The Qwen3.5 vision configuration accepted for Qwen3.5 execution. Port of
/// crates/model-serving/src/qwen3_5/vision/vision_config.rs.
///
/// The accepted architecture values are not tuning knobs. They determine
/// tensor shapes and model math: head_dimension=1152/16=72, position table
/// side=sqrt(2304)=48, flattened patch width=3*2*16*16=1536, and merger
/// width=1152*2*2=4608. Validating them at artifact load keeps later graph
/// assembly simple and avoids accidentally running checkpoint tensors under a
/// merely similar architecture.
public struct Qwen3_5VisionConfig: Equatable, Sendable {
    private static let ACCEPTED_VISION_MODEL_TYPES: Set<String> = [
        "qwen3_5", "qwen3_5_vision", "qwen3_5_moe_vision", "qwen3_5_moe",
    ];

    private let depthValue: UInt32;
    private let hiddenSizeValue: UInt32;
    private let inChannelsValue: UInt32;
    private let intermediateSizeValue: UInt32;
    private let headCountValue: UInt32;
    private let positionEmbeddingCountValue: UInt32;
    private let patchSizeValue: UInt32;
    private let spatialMergeSizeValue: UInt32;
    private let temporalPatchSizeValue: UInt32;
    private let outHiddenSizeValue: UInt32;
    private let hiddenActivationValue: String;

    private init(
        depth: UInt32, hiddenSize: UInt32, inChannels: UInt32, intermediateSize: UInt32,
        headCount: UInt32, positionEmbeddingCount: UInt32, patchSize: UInt32,
        spatialMergeSize: UInt32, temporalPatchSize: UInt32, outHiddenSize: UInt32,
        hiddenActivation: String) {
        self.depthValue = depth;
        self.hiddenSizeValue = hiddenSize;
        self.inChannelsValue = inChannels;
        self.intermediateSizeValue = intermediateSize;
        self.headCountValue = headCount;
        self.positionEmbeddingCountValue = positionEmbeddingCount;
        self.patchSizeValue = patchSize;
        self.spatialMergeSizeValue = spatialMergeSize;
        self.temporalPatchSizeValue = temporalPatchSize;
        self.outHiddenSizeValue = outHiddenSize;
        self.hiddenActivationValue = hiddenActivation;
    }

    /// Parses the vision_config section from the retained config bytes.
    public static func fromJsonBytes(configBytes: Data) throws -> Qwen3_5VisionConfig {
        guard let visionConfig: Qwen3_5VisionConfig = try Qwen3_5VisionConfig.fromOptionalJsonBytes(
            configBytes: configBytes) else {
            throw Qwen3_5ConfigError.missingVisionConfig;
        }
        return visionConfig;
    }

    /// Parses an optional vision_config section from the retained config
    /// bytes. Text-only Qwen checkpoints legitimately omit this section.
    public static func fromOptionalJsonBytes(configBytes: Data) throws -> Qwen3_5VisionConfig? {
        let configDocument: JsonWireValue;
        do {
            configDocument = try JsonWireParser.parseDocument(documentBytes: configBytes);
        } catch let jsonWireProblem as JsonWireProblem {
            throw Qwen3_5ConfigError.deserializeVisionConfig(problem: jsonWireProblem.description);
        }
        let configObject: JsonWireObject;
        do {
            configObject = try JsonWireValue.extractObject(configDocument);
        } catch let jsonWireProblem as JsonWireProblem {
            throw Qwen3_5ConfigError.deserializeVisionConfig(problem: jsonWireProblem.description);
        }
        guard let visionFieldsValue: JsonWireValue = configObject.value(forKey: "vision_config") else {
            return nil;
        }
        let visionFieldsObject: JsonWireObject;
        do {
            visionFieldsObject = try JsonWireValue.extractObject(visionFieldsValue);
        } catch let jsonWireProblem as JsonWireProblem {
            throw Qwen3_5ConfigError.deserializeVisionConfig(problem: jsonWireProblem.description);
        }
        return try Qwen3_5VisionConfig.decodeValidated(visionFieldsObject: visionFieldsObject);
    }

    private static func decodeValidated(
        visionFieldsObject: JsonWireObject) throws -> Qwen3_5VisionConfig {
        func decodeUInt32(_ fieldName: String) throws -> UInt32 {
            do {
                return try visionFieldsObject.decodeUInt32(fieldName: fieldName);
            } catch let jsonWireProblem as JsonWireProblem {
                throw Qwen3_5ConfigError.deserializeVisionConfig(
                    problem: jsonWireProblem.description);
            }
        }
        func decodeString(_ fieldName: String) throws -> String {
            do {
                return try visionFieldsObject.decodeString(fieldName: fieldName);
            } catch let jsonWireProblem as JsonWireProblem {
                throw Qwen3_5ConfigError.deserializeVisionConfig(
                    problem: jsonWireProblem.description);
            }
        }
        let modelType: String = try decodeString("model_type");
        let hiddenActivation: String = try decodeString("hidden_act");
        let depth: UInt32 = try decodeUInt32("depth");
        let hiddenSize: UInt32 = try decodeUInt32("hidden_size");
        let inChannels: UInt32 = try decodeUInt32("in_channels");
        let intermediateSize: UInt32 = try decodeUInt32("intermediate_size");
        let headCount: UInt32 = try decodeUInt32("num_heads");
        let positionEmbeddingCount: UInt32 = try decodeUInt32("num_position_embeddings");
        let patchSize: UInt32 = try decodeUInt32("patch_size");
        let spatialMergeSize: UInt32 = try decodeUInt32("spatial_merge_size");
        let temporalPatchSize: UInt32 = try decodeUInt32("temporal_patch_size");
        let outHiddenSize: UInt32 = try decodeUInt32("out_hidden_size");
        let deepstackVisualIndexes: Array<UInt32>;
        do {
            deepstackVisualIndexes = try visionFieldsObject.decodeArrayAllowingAbsent(
                fieldName: "deepstack_visual_indexes",
                mappedElement: { (indexWireValue: JsonWireValue) throws -> UInt32 in
                    let indexValue: UInt64 = try JsonWireValue.extractUInt64(indexWireValue);
                    guard let indexUInt32: UInt32 = UInt32(exactly: indexValue) else {
                        throw Qwen3_5ConfigError.deserializeVisionConfig(
                            problem: "invalid type: deepstack_visual_indexes entry exceeds u32");
                    }
                    return indexUInt32;
                });
        } catch let jsonWireProblem as JsonWireProblem {
            throw Qwen3_5ConfigError.deserializeVisionConfig(
                problem: jsonWireProblem.description);
        }
        // deepstack_visual_indexes is schema-validated but not carried into
        // the validated config, mirroring the Rust field that serde parses
        // and the constructor drops.
        _ = deepstackVisualIndexes;

        if Qwen3_5VisionConfig.ACCEPTED_VISION_MODEL_TYPES.contains(modelType) == false {
            throw Qwen3_5ConfigError.unexpectedStringValue(
                fieldName: "vision_config.model_type",
                expectedValue: "qwen3_5, qwen3_5_vision, qwen3_5_moe_vision, or qwen3_5_moe",
                actualValue: modelType);
        }
        // Structural sanity: values must be positive. Any valid Qwen3.5 vision
        // config is accepted; the values are not hardcoded to one model.
        if depth == 0 || hiddenSize == 0 || inChannels == 0 || intermediateSize == 0
            || headCount == 0 || positionEmbeddingCount == 0 || patchSize == 0
            || spatialMergeSize == 0 || temporalPatchSize == 0 || outHiddenSize == 0 {
            throw Qwen3_5ConfigError.invalidConfigValue(
                description: "vision_config numeric fields must be positive");
        }
        if hiddenSize % headCount != 0 {
            throw Qwen3_5ConfigError.invalidConfigValue(
                description: "vision_config.hidden_size must divide evenly by num_heads");
        }
        let headDimension: UInt32 = hiddenSize / headCount;
        if headDimension % 4 != 0 {
            throw Qwen3_5ConfigError.invalidConfigValue(
                description:
                    "vision attention head dimension must divide evenly across two rotary axes");
        }
        if hiddenActivation != "gelu_pytorch_tanh" {
            throw Qwen3_5ConfigError.unexpectedStringValue(
                fieldName: "vision_config.hidden_act",
                expectedValue: "gelu_pytorch_tanh",
                actualValue: hiddenActivation);
        }
        return Qwen3_5VisionConfig(
            depth: depth, hiddenSize: hiddenSize, inChannels: inChannels,
            intermediateSize: intermediateSize, headCount: headCount,
            positionEmbeddingCount: positionEmbeddingCount, patchSize: patchSize,
            spatialMergeSize: spatialMergeSize, temporalPatchSize: temporalPatchSize,
            outHiddenSize: outHiddenSize, hiddenActivation: hiddenActivation);
    }

    /// Returns the vision transformer block count.
    public var depth: UInt32 {
        return self.depthValue;
    }

    /// Returns the vision hidden dimension.
    public var hiddenSize: UInt32 {
        return self.hiddenSizeValue;
    }

    /// Returns the input channel count (3 for RGB).
    public var inChannels: UInt32 {
        return self.inChannelsValue;
    }

    /// Returns the feed-forward intermediate size.
    public var intermediateSize: UInt32 {
        return self.intermediateSizeValue;
    }

    /// Returns the attention head count.
    public var headCount: UInt32 {
        return self.headCountValue;
    }

    /// Returns the positional embedding count.
    public var positionEmbeddingCount: UInt32 {
        return self.positionEmbeddingCountValue;
    }

    /// Returns the patch size in pixels.
    public var patchSize: UInt32 {
        return self.patchSizeValue;
    }

    /// Returns the spatial merge size.
    public var spatialMergeSize: UInt32 {
        return self.spatialMergeSizeValue;
    }

    /// Returns the temporal patch size.
    public var temporalPatchSize: UInt32 {
        return self.temporalPatchSizeValue;
    }

    /// Returns the output hidden size (projects into the text model).
    public var outHiddenSize: UInt32 {
        return self.outHiddenSizeValue;
    }

    /// Returns the vision hidden activation function name.
    public var hiddenActivation: String {
        return self.hiddenActivationValue;
    }
}
