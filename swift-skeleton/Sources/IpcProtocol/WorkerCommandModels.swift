import Foundation;

/// A command sent from the HTTP process to its one inference worker.
///
/// Wire shape is internally tagged with `kind` in snake_case. Newtype-variant
/// payloads (one wrapped struct) flatten their fields into the same wire
/// object beside the tag, matching serde's internally-tagged serialization.
public enum WorkerCommand: Equatable {
    /// Supplies immutable startup configuration before any worker operation.
    case initializeWorker(WorkerStartupConfiguration);
    /// Starts one bounded structured-chat generation.
    case generate(ChatGenerationCommand);
    /// Starts one bounded text-to-image generation.
    case generateImage(ImageGenerationCommand);
    /// Starts one bounded text-embedding request.
    case generateEmbeddings(EmbeddingsCommand);
    /// Stops the active generation with this request identifier.
    case cancel(requestId: RequestId);
    /// Swaps the loaded model to a different model directory.
    /// The worker unloads the current model, validates and loads the new one,
    /// then emits a ModelSwapped event with the new model_id and capabilities.
    case swapModel(modelDirectory: String, modelConfiguration: WorkerModelConfiguration);
    /// Requests one MLX memory observation from a ready idle worker.
    case sampleMlxMemory;
    /// Replaces the worker's effective MLX process memory ceiling while idle.
    case updateMlxMemoryLimit(effectiveMlxMemoryCeilingBytes: UInt64, configurationGeneration: String);
    /// Requests the worker delete the persistent prompt-cache footprint on SSD
    /// (solid-state drive). A `nil` model id clears the entire global cache
    /// root; a concrete id removes only that model's tree.
    case clearPromptCache(modelId: String?);

    private static let expectedVariantNames: Array<String> = [
        "initialize_worker", "generate", "generate_image", "generate_embeddings", "cancel",
        "swap_model", "sample_mlx_memory", "update_mlx_memory_limit", "clear_prompt_cache",
    ];

    internal func wireValue() -> JsonWireValue {
        switch (self) {
        case let .initializeWorker(startupConfiguration):
            return WorkerCommand.flattenedTaggedWireObject(variantName: "initialize_worker", payloadWireValue: startupConfiguration.wireValue());
        case let .generate(chatGenerationCommand):
            return WorkerCommand.flattenedTaggedWireObject(variantName: "generate", payloadWireValue: chatGenerationCommand.wireValue());
        case let .generateImage(imageGenerationCommand):
            return WorkerCommand.flattenedTaggedWireObject(variantName: "generate_image", payloadWireValue: imageGenerationCommand.wireValue());
        case let .generateEmbeddings(embeddingsCommand):
            return WorkerCommand.flattenedTaggedWireObject(variantName: "generate_embeddings", payloadWireValue: embeddingsCommand.wireValue());
        case let .cancel(requestId):
            var wireObject = WorkerCommand.emptyWireObject();
            wireObject.appendEntry(key: "kind", value: .string("cancel"));
            wireObject.appendEntry(key: "request_id", value: requestId.wireValue());
            return .object(wireObject);
        case let .swapModel(modelDirectory, modelConfiguration):
            var wireObject = WorkerCommand.emptyWireObject();
            wireObject.appendEntry(key: "kind", value: .string("swap_model"));
            wireObject.appendEntry(key: "model_directory", value: .string(modelDirectory));
            wireObject.appendEntry(key: "model_configuration", value: modelConfiguration.wireValue());
            return .object(wireObject);
        case .sampleMlxMemory:
            var wireObject = WorkerCommand.emptyWireObject();
            wireObject.appendEntry(key: "kind", value: .string("sample_mlx_memory"));
            return .object(wireObject);
        case let .updateMlxMemoryLimit(effectiveMlxMemoryCeilingBytes, configurationGeneration):
            var wireObject = WorkerCommand.emptyWireObject();
            wireObject.appendEntry(key: "kind", value: .string("update_mlx_memory_limit"));
            wireObject.appendEntry(key: "effective_mlx_memory_ceiling_bytes", value: .unsignedInteger(effectiveMlxMemoryCeilingBytes));
            wireObject.appendEntry(key: "configuration_generation", value: .string(configurationGeneration));
            return .object(wireObject);
        case let .clearPromptCache(modelId):
            var wireObject = WorkerCommand.emptyWireObject();
            wireObject.appendEntry(key: "kind", value: .string("clear_prompt_cache"));
            wireObject.appendEntry(key: "model_id", value: WorkerCommand.optionalStringWireValue(modelId));
            return .object(wireObject);
        }
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> WorkerCommand {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        switch try wireObject.decodeTaggedVariantName(tagFieldName: "kind", expectedVariantNames: WorkerCommand.expectedVariantNames) {
        case "initialize_worker":
            let payloadObject = try WorkerCommand.payloadObjectStrippingVariantTag(wireObject: wireObject);
            return .initializeWorker(try WorkerStartupConfiguration.fromWireValue(.object(payloadObject)));
        case "generate":
            let payloadObject = try WorkerCommand.payloadObjectStrippingVariantTag(wireObject: wireObject);
            return .generate(try ChatGenerationCommand.fromWireValue(.object(payloadObject)));
        case "generate_image":
            let payloadObject = try WorkerCommand.payloadObjectStrippingVariantTag(wireObject: wireObject);
            return .generateImage(try ImageGenerationCommand.fromWireValue(.object(payloadObject)));
        case "generate_embeddings":
            let payloadObject = try WorkerCommand.payloadObjectStrippingVariantTag(wireObject: wireObject);
            return .generateEmbeddings(try EmbeddingsCommand.fromWireValue(.object(payloadObject)));
        case "cancel":
            let parsedCommand = WorkerCommand.cancel(requestId: try RequestId.fromWireValue(try wireObject.requireObjectValue(fieldName: "request_id")));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["request_id"]);
            return parsedCommand;
        case "swap_model":
            let parsedCommand = WorkerCommand.swapModel(
                modelDirectory: try wireObject.decodeString(fieldName: "model_directory"),
                modelConfiguration: try WorkerModelConfiguration.fromWireValue(try wireObject.requireObjectValue(fieldName: "model_configuration")));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["model_directory", "model_configuration"]);
            return parsedCommand;
        case "sample_mlx_memory":
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: []);
            return .sampleMlxMemory;
        case "update_mlx_memory_limit":
            let parsedCommand = WorkerCommand.updateMlxMemoryLimit(
                effectiveMlxMemoryCeilingBytes: try wireObject.decodeUInt64(fieldName: "effective_mlx_memory_ceiling_bytes"),
                configurationGeneration: try wireObject.decodeString(fieldName: "configuration_generation"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["effective_mlx_memory_ceiling_bytes", "configuration_generation"]);
            return parsedCommand;
        default:
            let parsedCommand = WorkerCommand.clearPromptCache(modelId: try wireObject.decodeOptionalString(fieldName: "model_id"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["model_id"]);
            return parsedCommand;
        }
    }

    private static func emptyWireObject() -> JsonWireObject {
        return JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
    }

    /// Serializes one wrapped-struct variant: the tag first, then every payload
    /// field flattened into the same object, exactly as serde's internally
    /// tagged newtype variants do.
    private static func flattenedTaggedWireObject(variantName: String, payloadWireValue: JsonWireValue) -> JsonWireValue {
        var wireObject = WorkerCommand.emptyWireObject();
        wireObject.appendEntry(key: "kind", value: .string(variantName));
        for payloadEntry in WorkerCommand.payloadEntries(payloadWireValue: payloadWireValue) {
            wireObject.appendEntry(key: payloadEntry.key, value: payloadEntry.value);
        }
        return .object(wireObject);
    }

    /// Wrapped-struct payload encoders always produce `.object`; the guard
    /// merely satisfies the pattern match.
    private static func payloadEntries(payloadWireValue: JsonWireValue) -> Array<(key: String, value: JsonWireValue)> {
        guard case let .object(payloadWireObject) = payloadWireValue else {
            return Array<(key: String, value: JsonWireValue)>();
        }
        return payloadWireObject.entries;
    }

    /// Rebuilds the wrapped struct's own wire object (without the variant tag)
    /// so the payload type's existing decoder and unknown-field rejection
    /// apply unchanged, mirroring serde stripping the tag before payload
    /// deserialization.
    private static func payloadObjectStrippingVariantTag(wireObject: JsonWireObject) throws -> JsonWireObject {
        var payloadEntries = Array<(key: String, value: JsonWireValue)>();
        for wireEntry in wireObject.entries {
            if wireEntry.key != "kind" {
                payloadEntries.append(wireEntry);
            }
        }
        return JsonWireObject(entries: payloadEntries);
    }

    private static func optionalStringWireValue(_ optionalValue: String?) -> JsonWireValue {
        guard let unwrappedValue = optionalValue else {
            return .null;
        }
        return .string(unwrappedValue);
    }
}
