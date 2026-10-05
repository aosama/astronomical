import Foundation;

/// One reply the daemon sends back to the local CLI process.
/// Wire shape is internally tagged with `kind` in snake_case.
public enum DaemonResponse: Equatable {
    case handshakeAccepted(protocolVersion: UInt32, applicationName: String);
    /// Answer to `DaemonRequest.status`.
    case status(workerStatus: DaemonWorkerStatus, readyModelId: String?, defaultModelId: String?);
    /// Answer to `DaemonRequest.modelsList`.
    case modelsList(models: Array<DaemonListedModel>);
    /// Answer to `DaemonRequest.catalog`.
    case catalog(entries: Array<DaemonCatalogEntry>);
    /// Terminal frame for an admitted `DaemonRequest.downloadStart`.
    case downloadStarted(huggingfaceId: String);
    /// Answer to `DaemonRequest.downloadStatus`.
    case downloadStatus(job: DaemonDownloadJob?);
    /// Terminal frame for an admitted `DaemonRequest.defaultModelSet`.
    case defaultModelSet(defaultModelId: String);
    /// Terminal refusal when the daemon declines a model-management request.
    case requestRejected(reason: String);
    /// One visible answer fragment, streamed in generation order.
    case chatGenerationText(text: String);
    /// One reasoning-channel fragment.
    case chatGenerationReasoning(text: String);
    /// One model-requested function call.
    case chatGenerationToolCall(toolCallIndex: UInt16, functionName: String, argumentsJson: String);
    /// Terminal frame for a finished generation.
    case chatGenerationCompleted(promptTokenCount: UInt32, generatedTokenCount: UInt16, reasoningTokenCount: UInt16, cachedTokenCount: UInt32, reason: ChatGenerationCompletionReason);
    /// Terminal frame for a failed generation.
    case chatGenerationFailed(reason: ChatGenerationFailureReason);
    /// Terminal refusal when the daemon declines the request before inference.
    case generationRejected(reason: String);
    /// Terminal frame for a finished embedding batch.
    case embeddingsCompleted(model: String, vectors: Array<Array<Float>>, inputTokenCounts: Array<UInt32>);
    /// Terminal frame for a failed embedding batch.
    case embeddingsFailed(reason: EmbeddingsFailureReason);

    private static let expectedVariantNames: Array<String> = [
        "handshake_accepted", "status", "models_list", "catalog", "download_started",
        "download_status", "default_model_set", "request_rejected", "chat_generation_text",
        "chat_generation_reasoning", "chat_generation_tool_call", "chat_generation_completed",
        "chat_generation_failed", "generation_rejected", "embeddings_completed", "embeddings_failed",
    ];

    internal func wireValue() -> JsonWireValue {
        switch self {
        case let .handshakeAccepted(protocolVersion, applicationName):
            var wireObject = DaemonResponse.emptyWireObject();
            wireObject.appendEntry(key: "kind", value: .string("handshake_accepted"));
            wireObject.appendEntry(key: "protocol_version", value: .unsignedInteger(UInt64(protocolVersion)));
            wireObject.appendEntry(key: "application_name", value: .string(applicationName));
            return .object(wireObject);
        case let .status(workerStatus, readyModelId, defaultModelId):
            var wireObject = DaemonResponse.emptyWireObject();
            wireObject.appendEntry(key: "kind", value: .string("status"));
            wireObject.appendEntry(key: "worker_status", value: workerStatus.wireValue());
            wireObject.appendEntry(key: "ready_model_id", value: DaemonResponse.optionalStringWireValue(readyModelId));
            wireObject.appendEntry(key: "default_model_id", value: DaemonResponse.optionalStringWireValue(defaultModelId));
            return .object(wireObject);
        case let .modelsList(models):
            var wireObject = DaemonResponse.emptyWireObject();
            wireObject.appendEntry(key: "kind", value: .string("models_list"));
            wireObject.appendEntry(key: "models", value: JsonWireValue.mappedArray(models, mappedWireValue: { (listedModel: DaemonListedModel) -> JsonWireValue in listedModel.wireValue() }));
            return .object(wireObject);
        case let .catalog(entries):
            var wireObject = DaemonResponse.emptyWireObject();
            wireObject.appendEntry(key: "kind", value: .string("catalog"));
            wireObject.appendEntry(key: "entries", value: JsonWireValue.mappedArray(entries, mappedWireValue: { (catalogEntry: DaemonCatalogEntry) -> JsonWireValue in catalogEntry.wireValue() }));
            return .object(wireObject);
        case let .downloadStarted(huggingfaceId):
            var wireObject = DaemonResponse.emptyWireObject();
            wireObject.appendEntry(key: "kind", value: .string("download_started"));
            wireObject.appendEntry(key: "huggingface_id", value: .string(huggingfaceId));
            return .object(wireObject);
        case let .downloadStatus(job):
            var wireObject = DaemonResponse.emptyWireObject();
            wireObject.appendEntry(key: "kind", value: .string("download_status"));
            wireObject.appendEntry(key: "job", value: DaemonResponse.optionalDownloadJobWireValue(job));
            return .object(wireObject);
        case let .defaultModelSet(defaultModelId):
            var wireObject = DaemonResponse.emptyWireObject();
            wireObject.appendEntry(key: "kind", value: .string("default_model_set"));
            wireObject.appendEntry(key: "default_model_id", value: .string(defaultModelId));
            return .object(wireObject);
        case let .requestRejected(reason):
            var wireObject = DaemonResponse.emptyWireObject();
            wireObject.appendEntry(key: "kind", value: .string("request_rejected"));
            wireObject.appendEntry(key: "reason", value: .string(reason));
            return .object(wireObject);
        case let .chatGenerationText(text):
            var wireObject = DaemonResponse.emptyWireObject();
            wireObject.appendEntry(key: "kind", value: .string("chat_generation_text"));
            wireObject.appendEntry(key: "text", value: .string(text));
            return .object(wireObject);
        case let .chatGenerationReasoning(text):
            var wireObject = DaemonResponse.emptyWireObject();
            wireObject.appendEntry(key: "kind", value: .string("chat_generation_reasoning"));
            wireObject.appendEntry(key: "text", value: .string(text));
            return .object(wireObject);
        case let .chatGenerationToolCall(toolCallIndex, functionName, argumentsJson):
            var wireObject = DaemonResponse.emptyWireObject();
            wireObject.appendEntry(key: "kind", value: .string("chat_generation_tool_call"));
            wireObject.appendEntry(key: "tool_call_index", value: .unsignedInteger(UInt64(toolCallIndex)));
            wireObject.appendEntry(key: "function_name", value: .string(functionName));
            wireObject.appendEntry(key: "arguments_json", value: .string(argumentsJson));
            return .object(wireObject);
        case let .chatGenerationCompleted(promptTokenCount, generatedTokenCount, reasoningTokenCount, cachedTokenCount, reason):
            var wireObject = DaemonResponse.emptyWireObject();
            wireObject.appendEntry(key: "kind", value: .string("chat_generation_completed"));
            wireObject.appendEntry(key: "prompt_token_count", value: .unsignedInteger(UInt64(promptTokenCount)));
            wireObject.appendEntry(key: "generated_token_count", value: .unsignedInteger(UInt64(generatedTokenCount)));
            wireObject.appendEntry(key: "reasoning_token_count", value: .unsignedInteger(UInt64(reasoningTokenCount)));
            wireObject.appendEntry(key: "cached_token_count", value: .unsignedInteger(UInt64(cachedTokenCount)));
            wireObject.appendEntry(key: "reason", value: reason.wireValue());
            return .object(wireObject);
        case let .chatGenerationFailed(reason):
            var wireObject = DaemonResponse.emptyWireObject();
            wireObject.appendEntry(key: "kind", value: .string("chat_generation_failed"));
            wireObject.appendEntry(key: "reason", value: reason.wireValue());
            return .object(wireObject);
        case let .generationRejected(reason):
            var wireObject = DaemonResponse.emptyWireObject();
            wireObject.appendEntry(key: "kind", value: .string("generation_rejected"));
            wireObject.appendEntry(key: "reason", value: .string(reason));
            return .object(wireObject);
        case let .embeddingsCompleted(model, vectors, inputTokenCounts):
            var wireObject = DaemonResponse.emptyWireObject();
            wireObject.appendEntry(key: "kind", value: .string("embeddings_completed"));
            wireObject.appendEntry(key: "model", value: .string(model));
            wireObject.appendEntry(key: "vectors", value: DaemonResponse.nestedFloat32ArrayWireValue(vectors));
            wireObject.appendEntry(key: "input_token_counts", value: DaemonResponse.uint32ArrayWireValue(inputTokenCounts));
            return .object(wireObject);
        case let .embeddingsFailed(reason):
            var wireObject = DaemonResponse.emptyWireObject();
            wireObject.appendEntry(key: "kind", value: .string("embeddings_failed"));
            wireObject.appendEntry(key: "reason", value: reason.wireValue());
            return .object(wireObject);
        }
    }

    internal static func fromWireValue(_ wireValue: JsonWireValue) throws -> DaemonResponse {
        let wireObject = try JsonWireValue.extractObject(wireValue);
        switch try wireObject.decodeTaggedVariantName(tagFieldName: "kind", expectedVariantNames: DaemonResponse.expectedVariantNames) {
        case "handshake_accepted":
            let parsedResponse = DaemonResponse.handshakeAccepted(
                protocolVersion: try wireObject.decodeUInt32(fieldName: "protocol_version"),
                applicationName: try wireObject.decodeString(fieldName: "application_name"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["protocol_version", "application_name"]);
            return parsedResponse;
        case "status":
            let parsedResponse = DaemonResponse.status(
                workerStatus: try DaemonWorkerStatus.fromWireValue(try wireObject.requireObjectValue(fieldName: "worker_status")),
                readyModelId: try wireObject.decodeOptionalString(fieldName: "ready_model_id"),
                defaultModelId: try wireObject.decodeOptionalString(fieldName: "default_model_id"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["worker_status", "ready_model_id", "default_model_id"]);
            return parsedResponse;
        case "models_list":
            let parsedResponse = DaemonResponse.modelsList(
                models: try wireObject.decodeArray(fieldName: "models", mappedElement: { (elementWireValue: JsonWireValue) throws -> DaemonListedModel in
                    try DaemonListedModel.fromWireValue(elementWireValue)
                }));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["models"]);
            return parsedResponse;
        case "catalog":
            let parsedResponse = DaemonResponse.catalog(
                entries: try wireObject.decodeArray(fieldName: "entries", mappedElement: { (elementWireValue: JsonWireValue) throws -> DaemonCatalogEntry in
                    try DaemonCatalogEntry.fromWireValue(elementWireValue)
                }));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["entries"]);
            return parsedResponse;
        case "download_started":
            let parsedResponse = DaemonResponse.downloadStarted(huggingfaceId: try wireObject.decodeString(fieldName: "huggingface_id"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["huggingface_id"]);
            return parsedResponse;
        case "download_status":
            let parsedResponse = DaemonResponse.downloadStatus(job: try DaemonResponse.decodeOptionalDownloadJob(wireObject: wireObject));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["job"]);
            return parsedResponse;
        case "default_model_set":
            let parsedResponse = DaemonResponse.defaultModelSet(defaultModelId: try wireObject.decodeString(fieldName: "default_model_id"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["default_model_id"]);
            return parsedResponse;
        case "request_rejected":
            let parsedResponse = DaemonResponse.requestRejected(reason: try wireObject.decodeString(fieldName: "reason"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["reason"]);
            return parsedResponse;
        case "chat_generation_text":
            let parsedResponse = DaemonResponse.chatGenerationText(text: try wireObject.decodeString(fieldName: "text"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["text"]);
            return parsedResponse;
        case "chat_generation_reasoning":
            let parsedResponse = DaemonResponse.chatGenerationReasoning(text: try wireObject.decodeString(fieldName: "text"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["text"]);
            return parsedResponse;
        case "chat_generation_tool_call":
            let parsedResponse = DaemonResponse.chatGenerationToolCall(
                toolCallIndex: try wireObject.decodeUInt16(fieldName: "tool_call_index"),
                functionName: try wireObject.decodeString(fieldName: "function_name"),
                argumentsJson: try wireObject.decodeString(fieldName: "arguments_json"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["tool_call_index", "function_name", "arguments_json"]);
            return parsedResponse;
        case "chat_generation_completed":
            let parsedResponse = DaemonResponse.chatGenerationCompleted(
                promptTokenCount: try wireObject.decodeUInt32(fieldName: "prompt_token_count"),
                generatedTokenCount: try wireObject.decodeUInt16(fieldName: "generated_token_count"),
                reasoningTokenCount: try wireObject.decodeUInt16(fieldName: "reasoning_token_count"),
                cachedTokenCount: try wireObject.decodeUInt32(fieldName: "cached_token_count"),
                reason: try ChatGenerationCompletionReason.fromWireValue(try wireObject.requireObjectValue(fieldName: "reason")));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: [
                "prompt_token_count", "generated_token_count", "reasoning_token_count", "cached_token_count", "reason",
            ]);
            return parsedResponse;
        case "chat_generation_failed":
            let parsedResponse = DaemonResponse.chatGenerationFailed(
                reason: try ChatGenerationFailureReason.fromWireValue(try wireObject.requireObjectValue(fieldName: "reason")));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["reason"]);
            return parsedResponse;
        case "generation_rejected":
            let parsedResponse = DaemonResponse.generationRejected(reason: try wireObject.decodeString(fieldName: "reason"));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["reason"]);
            return parsedResponse;
        case "embeddings_completed":
            let parsedResponse = DaemonResponse.embeddingsCompleted(
                model: try wireObject.decodeString(fieldName: "model"),
                vectors: try wireObject.decodeArray(fieldName: "vectors", mappedElement: { (vectorWireValue: JsonWireValue) throws -> Array<Float> in
                    try JsonWireValue.extractArray(vectorWireValue, mappedElement: { (elementWireValue: JsonWireValue) throws -> Float in
                        try DaemonResponse.float32FromWireValue(elementWireValue)
                    })
                }),
                inputTokenCounts: try wireObject.decodeArray(fieldName: "input_token_counts", mappedElement: { (elementWireValue: JsonWireValue) throws -> UInt32 in
                    try JsonWireValue.clampToUInt32(try JsonWireValue.extractUInt64(elementWireValue))
                }));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["model", "vectors", "input_token_counts"]);
            return parsedResponse;
        default:
            let parsedResponse = DaemonResponse.embeddingsFailed(
                reason: try EmbeddingsFailureReason.fromWireValue(try wireObject.requireObjectValue(fieldName: "reason")));
            try wireObject.rejectUnknownFieldsBesidesTag(tagFieldName: "kind", allowedFieldNames: ["reason"]);
            return parsedResponse;
        }
    }

    private static func emptyWireObject() -> JsonWireObject {
        return JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
    }

    private static func optionalStringWireValue(_ optionalValue: String?) -> JsonWireValue {
        guard let unwrappedValue = optionalValue else {
            return .null;
        }
        return .string(unwrappedValue);
    }

    private static func optionalDownloadJobWireValue(_ optionalJob: DaemonDownloadJob?) -> JsonWireValue {
        guard let unwrappedJob = optionalJob else {
            return .null;
        }
        return unwrappedJob.wireValue();
    }

    private static func decodeOptionalDownloadJob(wireObject: JsonWireObject) throws -> DaemonDownloadJob? {
        guard let jobObject = try wireObject.decodeOptionalObject(fieldName: "job") else {
            return nil;
        }
        return try DaemonDownloadJob.fromWireValue(.object(jobObject));
    }

    private static func nestedFloat32ArrayWireValue(_ vectors: Array<Array<Float>>) -> JsonWireValue {
        return JsonWireValue.mappedArray(vectors, mappedWireValue: { (vector: Array<Float>) -> JsonWireValue in
            JsonWireValue.mappedArray(vector, mappedWireValue: { (vectorElement: Float) -> JsonWireValue in .float32(vectorElement) })
        });
    }

    private static func uint32ArrayWireValue(_ numericValues: Array<UInt32>) -> JsonWireValue {
        return JsonWireValue.mappedArray(numericValues, mappedWireValue: { (numericValue: UInt32) -> JsonWireValue in .unsignedInteger(UInt64(numericValue)) });
    }

    private static func float32FromWireValue(_ elementWireValue: JsonWireValue) throws -> Float {
        switch elementWireValue {
        case let .float32(numericValue): return numericValue;
        case let .double(numericValue): return Float(numericValue);
        case let .unsignedInteger(numericValue): return Float(numericValue);
        case let .signedInteger(numericValue): return Float(numericValue);
        default: throw JsonWireProblem.invalidType(expectedTypeName: "f32", found: elementWireValue.foundDescription);
        }
    }
}
