import Foundation;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;

/// Advertises the models the supervisor can currently serve safely in the
/// OpenAI list shape.
///
/// Migrates the mapping half of apps/supervisor/src/openai_models_endpoint.rs:
/// family discovery owns behavior; this layer owns only the common API shape.
/// When discovery yields nothing but a worker is ready, the ready model is
/// advertised from its acknowledged worker capabilities — the same fallback
/// order as the Rust endpoint.
public enum RestAdvertisedModels {

    private static let ASTRONOMICAL_MODEL_OWNER: String = "astronomical";
    private static let TEXT_INPUT_MODALITY: String = "text";
    private static let IMAGE_INPUT_MODALITY: String = "image";
    private static let TEXT_OUTPUT_MODALITY: String = "text";
    private static let IMAGE_OUTPUT_MODALITY: String = "image";
    private static let OPENAI_CHAT_REASONING_FORMAT: String =
        "openai_chat_reasoning_content_and_responses_reasoning_summary_text";
    private static let OPENAI_FUNCTION_CALL_FORMAT: String = "openai_function_call";
    private static let CHAT_COMPLETIONS_ENDPOINT_PATH: String = "/v1/chat/completions";
    private static let RESPONSES_ENDPOINT_PATH: String = "/v1/responses";
    private static let IMAGE_GENERATIONS_ENDPOINT_PATH: String = "/v1/images/generations";

    /// Advertises the discovered models, falling back to the ready worker's
    /// acknowledged capabilities when discovery yields nothing.
    public static func advertise(
        discoveredModels: Array<DiscoveryDiscoveredModel>,
        workerHealthSnapshot: WorkerHealthSnapshot,
        createdAtIndexedSeconds: UInt64
    ) throws -> Array<OpenAiModel> {
        if discoveredModels.isEmpty {
            guard workerHealthSnapshot.status.isReady(),
                let readyModelId: String = workerHealthSnapshot.readyModelId,
                let readyModelCapabilities: WorkerModelCapabilities = workerHealthSnapshot.readyModelCapabilities
            else {
                return Array();
            }
            return [try self.fromReadyWorkerCapabilities(
                readyModelId: readyModelId,
                readyModelCapabilities: readyModelCapabilities,
                createdAtIndexedSeconds: createdAtIndexedSeconds)];
        }
        var advertisedModels: Array<OpenAiModel> = Array();
        for discoveredModel: DiscoveryDiscoveredModel in discoveredModels {
            advertisedModels.append(try self.fromDiscoveredModel(
                discoveredModel: discoveredModel,
                createdAtIndexedSeconds: createdAtIndexedSeconds));
        }
        return advertisedModels;
    }

    /// Resolves an optional provider prefix in a requested model identifier.
    public static func resolveRequestedModelId(
        requestedModelId: String,
        advertisedModels: Array<OpenAiModel>
    ) -> String {
        let knownModelIds: Array<String> = advertisedModels.map({ (advertisedModel: OpenAiModel) -> String in
            return advertisedModel.id();
        });
        return ModelIdentity.resolveModelId(requestedModelId: requestedModelId, knownModelIds: knownModelIds);
    }

    private static func fromDiscoveredModel(
        discoveredModel: DiscoveryDiscoveredModel,
        createdAtIndexedSeconds: UInt64
    ) throws -> OpenAiModel {
        switch (discoveredModel.capabilities) {
        case let .chat(chatCapabilities):
            return try OpenAiModel.fromParts(modelParts: OpenAiModelParts(
                modelId: discoveredModel.modelId,
                created: createdAtIndexedSeconds,
                ownedBy: RestAdvertisedModels.ASTRONOMICAL_MODEL_OWNER,
                contextWindow: chatCapabilities.contextWindowTokens,
                maxInputTokens: chatCapabilities.maximumInputTokens,
                maxOutputTokens: chatCapabilities.maximumOutputTokens,
                inputModalities: RestAdvertisedModels.inputModalities(supportsVision: chatCapabilities.supportsVision),
                outputModalities: [RestAdvertisedModels.TEXT_OUTPUT_MODALITY],
                supportsStreaming: true,
                supportsReasoning: chatCapabilities.supportsReasoning,
                reasoningFormat: RestAdvertisedModels.reasoningFormat(supportsReasoning: chatCapabilities.supportsReasoning),
                supportsToolCalls: chatCapabilities.supportsToolCalls,
                toolCallFormat: RestAdvertisedModels.toolCallFormat(supportsToolCalls: chatCapabilities.supportsToolCalls),
                supportedEndpoints: RestAdvertisedModels.supportedGenerationEndpointPaths(),
                supportsStructuredOutputs: true,
                structuredOutputEnforcement: ResponseFormatConstants.STRUCTURED_OUTPUT_ENFORCEMENT_LOGITS_MASK));
        case let .imageGeneration(imageCapabilities):
            return try OpenAiModel.fromImageParts(imageModelParts: OpenAiImageModelParts(
                modelId: discoveredModel.modelId,
                created: createdAtIndexedSeconds,
                ownedBy: RestAdvertisedModels.ASTRONOMICAL_MODEL_OWNER,
                inputModalities: [RestAdvertisedModels.TEXT_INPUT_MODALITY],
                outputModalities: [RestAdvertisedModels.IMAGE_OUTPUT_MODALITY],
                supportedEndpoints: imageCapabilities.supportsTextToImage
                    ? [RestAdvertisedModels.IMAGE_GENERATIONS_ENDPOINT_PATH]
                    : Array<String>()));
        case let .embeddings(embeddingCapabilities):
            return try OpenAiModel.fromEmbeddingParts(embeddingModelParts: OpenAiEmbeddingModelParts(
                modelId: discoveredModel.modelId,
                created: createdAtIndexedSeconds,
                ownedBy: RestAdvertisedModels.ASTRONOMICAL_MODEL_OWNER,
                vectorWidth: embeddingCapabilities.vectorWidth,
                maxInputTokens: embeddingCapabilities.maximumInputTokens));
        }
    }

    private static func fromReadyWorkerCapabilities(
        readyModelId: String,
        readyModelCapabilities: WorkerModelCapabilities,
        createdAtIndexedSeconds: UInt64
    ) throws -> OpenAiModel {
        if let chatCapabilities: ChatModelCapabilities = readyModelCapabilities.chat {
            return try OpenAiModel.fromParts(modelParts: OpenAiModelParts(
                modelId: readyModelId,
                created: createdAtIndexedSeconds,
                ownedBy: RestAdvertisedModels.ASTRONOMICAL_MODEL_OWNER,
                contextWindow: chatCapabilities.contextWindow,
                maxInputTokens: chatCapabilities.maxInputTokens,
                maxOutputTokens: chatCapabilities.maxOutputTokens,
                inputModalities: RestAdvertisedModels.inputModalities(supportsVision: chatCapabilities.hasVision),
                outputModalities: [RestAdvertisedModels.TEXT_OUTPUT_MODALITY],
                supportsStreaming: true,
                supportsReasoning: chatCapabilities.supportsReasoning,
                reasoningFormat: RestAdvertisedModels.reasoningFormat(supportsReasoning: chatCapabilities.supportsReasoning),
                supportsToolCalls: chatCapabilities.supportsToolCalls,
                toolCallFormat: RestAdvertisedModels.toolCallFormat(supportsToolCalls: chatCapabilities.supportsToolCalls),
                supportedEndpoints: RestAdvertisedModels.supportedGenerationEndpointPaths(),
                supportsStructuredOutputs: true,
                structuredOutputEnforcement: ResponseFormatConstants.STRUCTURED_OUTPUT_ENFORCEMENT_LOGITS_MASK));
        }
        if let embeddingCapabilities: WorkerEmbeddingCapabilities = readyModelCapabilities.embeddings {
            return try OpenAiModel.fromEmbeddingParts(embeddingModelParts: OpenAiEmbeddingModelParts(
                modelId: readyModelId,
                created: createdAtIndexedSeconds,
                ownedBy: RestAdvertisedModels.ASTRONOMICAL_MODEL_OWNER,
                vectorWidth: embeddingCapabilities.vectorWidth,
                maxInputTokens: embeddingCapabilities.maxInputTokens));
        }
        return try OpenAiModel.fromImageParts(imageModelParts: OpenAiImageModelParts(
            modelId: readyModelId,
            created: createdAtIndexedSeconds,
            ownedBy: RestAdvertisedModels.ASTRONOMICAL_MODEL_OWNER,
            inputModalities: [RestAdvertisedModels.TEXT_INPUT_MODALITY],
            outputModalities: [RestAdvertisedModels.IMAGE_OUTPUT_MODALITY],
            supportedEndpoints: readyModelCapabilities.imageGeneration != nil
                ? [RestAdvertisedModels.IMAGE_GENERATIONS_ENDPOINT_PATH]
                : Array<String>()));
    }

    private static func inputModalities(supportsVision: Bool) -> Array<String> {
        if supportsVision {
            return [RestAdvertisedModels.TEXT_INPUT_MODALITY, RestAdvertisedModels.IMAGE_INPUT_MODALITY];
        }
        return [RestAdvertisedModels.TEXT_INPUT_MODALITY];
    }

    private static func reasoningFormat(supportsReasoning: Bool) -> String? {
        if supportsReasoning {
            return RestAdvertisedModels.OPENAI_CHAT_REASONING_FORMAT;
        }
        return nil;
    }

    private static func toolCallFormat(supportsToolCalls: Bool) -> String? {
        if supportsToolCalls {
            return RestAdvertisedModels.OPENAI_FUNCTION_CALL_FORMAT;
        }
        return nil;
    }

    private static func supportedGenerationEndpointPaths() -> Array<String> {
        return [
            RestAdvertisedModels.CHAT_COMPLETIONS_ENDPOINT_PATH,
            RestAdvertisedModels.RESPONSES_ENDPOINT_PATH,
        ];
    }
}
