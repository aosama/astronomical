import Foundation;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;

/// Everything the REST embeddings route needs: the embeddings executor seam,
/// the supervisor-local request-id space, and the live resolved
/// configuration the discovery and capability checks read from.
public struct RestEmbeddingsRouteContext: @unchecked Sendable {

    let embeddingsExecutor: any EmbeddingsExecuting;
    let requestIdAllocator: ChatRequestIdAllocator;
    let resolvedRuntimeConfig: ResolvedRuntimeConfig;

    public init(
        embeddingsExecutor: any EmbeddingsExecuting,
        requestIdAllocator: ChatRequestIdAllocator,
        resolvedRuntimeConfig: ResolvedRuntimeConfig
    ) {
        self.embeddingsExecutor = embeddingsExecutor;
        self.requestIdAllocator = requestIdAllocator;
        self.resolvedRuntimeConfig = resolvedRuntimeConfig;
    }
}

/// The POST /v1/embeddings endpoint.
///
/// Port of apps/supervisor/src/openai_embeddings_endpoint.rs: complete
/// validation before admission — JSON decode, public validation, worker
/// health, model resolution and capability match — then one embeddings
/// command whose output renders in the requested encoding format.
enum RestEmbeddingsEndpoint {

    static let routeMethod: String = "POST";
    static let routePath: String = "/v1/embeddings";

    static func handle(
        _ request: RestHttpRequest,
        embeddingsContext: RestEmbeddingsRouteContext
    ) -> RestHttpResponse {
        let embeddingsRequest: OpenAiEmbeddingsRequest;
        do {
            let requestWireValue: JsonWireValue = try JsonWireParser.parseDocument(
                documentBytes: request.bodyBytes);
            embeddingsRequest = try OpenAiEmbeddingsRequest.decoded(wireValue: requestWireValue);
        } catch {
            return RestEmbeddingsEndpoint.invalidRequestResponse(
                message: "request body is not valid JSON: \(error)",
                parameter: nil,
                code: "invalid_json");
        }
        let requestParts: OpenAiEmbeddingsRequestParts;
        do {
            requestParts = try embeddingsRequest.intoParts();
        } catch let validationRejection as OpenAiEmbeddingsValidationError {
            return RestEmbeddingsEndpoint.invalidRequestResponse(
                message: validationRejection.errorDescription
                    ?? String(describing: validationRejection),
                parameter: RestEmbeddingsEndpoint.validationParameter(validationRejection),
                code: "invalid_request");
        } catch {
            return RestEmbeddingsEndpoint.invalidRequestResponse(
                message: String(describing: error),
                parameter: nil,
                code: "invalid_request");
        }
        let workerHealthSnapshot: WorkerHealthSnapshot =
            embeddingsContext.embeddingsExecutor.workerHealthSnapshot();
        if workerHealthSnapshot.status.isReady() == false {
            return RestEmbeddingsEndpoint.workerUnavailableResponse();
        }
        let knownModelIds: Array<String> = embeddingsContext.resolvedRuntimeConfig.discoveredModels
            .map { (discoveredModel: DiscoveryDiscoveredModel) -> String in
                return discoveredModel.modelId;
            };
        let resolvedModelId: String = ModelIdentity.resolveModelId(
            requestedModelId: requestParts.model,
            knownModelIds: knownModelIds);
        guard let discoveredModel: DiscoveryDiscoveredModel =
            embeddingsContext.resolvedRuntimeConfig.discoveredModels.first(
                where: { (candidateModel: DiscoveryDiscoveredModel) -> Bool in
                    return candidateModel.modelId == resolvedModelId;
                }) else {
            return RestEmbeddingsEndpoint.invalidRequestResponse(
                message: "model is not available to the local worker",
                parameter: "model",
                code: "model_not_found");
        }
        guard case .embeddings = discoveredModel.capabilities else {
            return RestEmbeddingsEndpoint.invalidRequestResponse(
                message: "the requested model does not support embeddings",
                parameter: "model",
                code: "model_capability_mismatch");
        }
        guard let requestIdentifier: UInt64 = embeddingsContext.requestIdAllocator.allocate() else {
            return RestEmbeddingsEndpoint.serviceUnavailableResponse(
                message: "the local request identifier space is exhausted",
                code: "request_id_exhausted");
        }
        let embeddingsCommand: EmbeddingsCommand = EmbeddingsCommand(
            requestId: RequestId(rawRequestId: requestIdentifier),
            model: resolvedModelId,
            inputs: requestParts.inputs,
            encodingFormat: requestParts.encodingFormat == .float ? .float : .base64,
            dimensions: requestParts.dimensions);
        let embeddingsOutput: EmbeddingsOutput;
        do {
            embeddingsOutput = try embeddingsContext.embeddingsExecutor
                .startEmbeddingsGeneration(embeddingsCommand);
        } catch let startError as GenerationStartError {
            return RestEmbeddingsEndpoint.generationStartFailureResponse(startError);
        } catch let executionError as EmbeddingsExecutionError {
            return RestEmbeddingsEndpoint.executionFailureResponse(
                requestIdentifier,
                executionError);
        } catch {
            return RestEmbeddingsEndpoint.workerUnavailableResponse();
        }
        var aggregateInputTokenCount: UInt32 = 0;
        for inputTokenCount: UInt32 in embeddingsOutput.inputTokenCounts {
            aggregateInputTokenCount =
                aggregateInputTokenCount.addingReportingOverflow(inputTokenCount).overflow
                    ? UInt32.max
                    : aggregateInputTokenCount + inputTokenCount;
        }
        guard let usage: OpenAiTokenUsage = OpenAiTokenUsage.new(
            promptTokens: aggregateInputTokenCount,
            completionTokens: 0) else {
            return RestEmbeddingsEndpoint.workerUnavailableResponse();
        }
        let embeddingRows: Array<OpenAiEmbedding> = embeddingsOutput.embeddings.enumerated()
            .map { (embeddingRow: (offset: Int, element: Array<Float>)) -> OpenAiEmbedding in
                let vector: OpenAiEmbeddingVector;
                if requestParts.encodingFormat == .base64 {
                    vector = .base64(
                        RestEmbeddingsEndpoint.base64Text(embeddingRow.element));
                } else {
                    vector = .float(embeddingRow.element);
                }
                return OpenAiEmbedding(
                    index: UInt32(embeddingRow.offset),
                    embedding: vector);
            };
        let embeddingsResponse: OpenAiEmbeddingsResponse = OpenAiEmbeddingsResponse(
            embeddings: embeddingRows,
            model: resolvedModelId,
            usage: usage);
        return (try? RestHttpResponse.json(
            statusCode: 200,
            wireValue: embeddingsResponse.wireValue()))
            ?? RestHttpResponse.text(statusCode: 500, body: "the response could not be serialized");
    }

    private static func validationParameter(
        _ validationError: OpenAiEmbeddingsValidationError
    ) -> String {
        switch (validationError) {
        case .unknownField:
            return "request";
        case .emptyModel:
            return "model";
        case .emptyInput, .inputCountExceeded, .inputTextTooLarge, .totalInputBytesExceeded:
            return "input";
        case .unsupportedEncodingFormat:
            return "encoding_format";
        case .invalidDimensions:
            return "dimensions";
        }
    }

    private static func base64Text(_ components: Array<Float>) -> String {
        var littleEndianBytes: Array<UInt8> = Array<UInt8>();
        littleEndianBytes.reserveCapacity(components.count * 4);
        for component: Float in components {
            withUnsafeBytes(of: component.bitPattern.littleEndian) { (componentBytes) -> Void in
                littleEndianBytes.append(contentsOf: componentBytes);
            };
        }
        return Data(littleEndianBytes).base64EncodedString();
    }

    private static func generationStartFailureResponse(
        _ startError: GenerationStartError
    ) -> RestHttpResponse {
        switch (startError) {
        case .capacityUnavailable:
            return RestEmbeddingsEndpoint.capacityResponse(message: "the generation queue is full");
        case let .modelLoadFailed(modelLoadFailureReason):
            return RestEmbeddingsEndpoint.jsonResponse(
                statusCode: 503,
                errorResponse: OpenAiErrorResponse.modelLoadFailed(
                    modelLoadFailureReason: modelLoadFailureReason));
        case .requestTooLarge:
            return RestEmbeddingsEndpoint.invalidRequestResponse(
                message: "the request exceeded the local IPC transport limit",
                parameter: nil,
                code: "request_too_large",
                statusCodeOverride: 413);
        case .workerUnavailable:
            return RestEmbeddingsEndpoint.workerUnavailableResponse();
        }
    }

    private static func executionFailureResponse(
        _ requestIdentifier: UInt64,
        _ executionError: EmbeddingsExecutionError
    ) -> RestHttpResponse {
        switch (executionError) {
        case let .workerFailure(failureReason):
            switch (failureReason) {
            case .invalidRequest:
                return RestEmbeddingsEndpoint.invalidRequestResponse(
                    message: "the embeddings request was rejected by the local worker",
                    parameter: nil,
                    code: "invalid_request");
            case let .contextLengthExceeded(actualTotalContextTokens, maximumContextTokens):
                return RestEmbeddingsEndpoint.invalidRequestResponse(
                    message: "embedding input has \(actualTotalContextTokens) tokens, exceeding the "
                        + "\(maximumContextTokens)-token encoder context",
                    parameter: "input",
                    code: "context_length_exceeded");
            case .engineBusy:
                return RestEmbeddingsEndpoint.capacityResponse(
                    message: "the embedding engine is busy");
            case .fatalExecution, .malformedModelOutput:
                return RestEmbeddingsEndpoint.embeddingsWorkerFailureResponse();
            }
        case .workerUnavailable:
            return RestEmbeddingsEndpoint.workerUnavailableResponse();
        }
    }

    private static func embeddingsWorkerFailureResponse() -> RestHttpResponse {
        return RestEmbeddingsEndpoint.jsonResponse(
            statusCode: 500,
            errorResponse: OpenAiErrorResponse.serviceUnavailable(
                message: "embedding generation failed in the local worker",
                code: "embeddings_failed"));
    }

    private static func capacityResponse(message: String) -> RestHttpResponse {
        return RestEmbeddingsEndpoint.jsonResponse(
            statusCode: 429,
            errorResponse: OpenAiErrorResponse.capacityUnavailable(message: message));
    }

    private static func workerUnavailableResponse() -> RestHttpResponse {
        return RestEmbeddingsEndpoint.serviceUnavailableResponse(
            message: "the local worker is unavailable",
            code: "worker_unavailable");
    }

    private static func serviceUnavailableResponse(
        message: String,
        code: String
    ) -> RestHttpResponse {
        return RestEmbeddingsEndpoint.jsonResponse(
            statusCode: 503,
            errorResponse: OpenAiErrorResponse.serviceUnavailable(message: message, code: code));
    }

    private static func invalidRequestResponse(
        message: String,
        parameter: String?,
        code: String,
        statusCodeOverride: Int? = nil
    ) -> RestHttpResponse {
        return RestEmbeddingsEndpoint.jsonResponse(
            statusCode: statusCodeOverride ?? 400,
            errorResponse: OpenAiErrorResponse.invalidRequest(
                message: message, parameter: parameter, code: code));
    }

    private static func jsonResponse(
        statusCode: Int,
        errorResponse: OpenAiErrorResponse
    ) -> RestHttpResponse {
        return (try? RestHttpResponse.json(
            statusCode: statusCode,
            wireValue: errorResponse.wireValue()))
            ?? RestHttpResponse.text(statusCode: 500, body: "the response could not be serialized");
    }
}
