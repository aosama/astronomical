import Foundation;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;

/// Everything the REST Responses route needs: the generation executor seam,
/// the supervisor-local request-id space, the live resolved configuration,
/// and the instance paths the optional thinking-channel seed reads from.
public struct RestResponsesRouteContext: @unchecked Sendable {

    let responsesExecutor: any ChatGenerationExecuting;
    let requestIdAllocator: ChatRequestIdAllocator;
    let resolvedRuntimeConfig: ResolvedRuntimeConfig;
    let instancePaths: AstronomicalInstancePaths;

    public init(
        responsesExecutor: any ChatGenerationExecuting,
        requestIdAllocator: ChatRequestIdAllocator,
        resolvedRuntimeConfig: ResolvedRuntimeConfig,
        instancePaths: AstronomicalInstancePaths
    ) {
        self.responsesExecutor = responsesExecutor;
        self.requestIdAllocator = requestIdAllocator;
        self.resolvedRuntimeConfig = resolvedRuntimeConfig;
        self.instancePaths = instancePaths;
    }
}

/// The POST /v1/responses endpoint.
///
/// Port of apps/supervisor/src/openai_responses_endpoint.rs: decode and
/// validate the public request, gate on worker health, resolve the requested
/// model, translate into the IPC chat command, fill defaults, admit the
/// generation, and answer with either one JSON Responses object or the
/// buffered semantic SSE frame sequence.
enum RestResponsesEndpoint {

    static let routeMethod: String = "POST";
    static let routePath: String = "/v1/responses";
    private static let eventStreamContentType: String = "text/event-stream";

    static func handle(
        _ request: RestHttpRequest,
        responsesContext: RestResponsesRouteContext
    ) -> RestHttpResponse {
        let attributionStart: ContinuousClock.Instant? = DaemonIpcPerformanceAttribution.startedOperation(
            operationName: "rest_responses",
            performanceAttributionEnabled: responsesContext.resolvedRuntimeConfig.performanceAttributionEnabled);
        defer {
            DaemonIpcPerformanceAttribution.finishedOperation(
                operationName: "rest_responses",
                operationStart: attributionStart,
                operationOutcome: "served",
                performanceAttributionEnabled: responsesContext.resolvedRuntimeConfig.performanceAttributionEnabled);
        }
        let responsesWireValue: JsonWireValue;
        do {
            responsesWireValue = try JsonWireParser.parseDocument(documentBytes: request.bodyBytes);
        } catch {
            return RestResponsesEndpoint.invalidRequestResponse(
                message: "request body is not valid JSON: \(error)",
                code: "invalid_json");
        }
        let responsesRequest: OpenAiResponsesRequest;
        do {
            responsesRequest = try OpenAiResponsesRequest.decoded(wireValue: responsesWireValue);
        } catch {
            return RestResponsesEndpoint.invalidRequestResponse(
                message: "request body is not valid JSON: \(error)",
                code: "invalid_json");
        }
        let requestParts: OpenAiResponsesRequestParts;
        do {
            requestParts = try responsesRequest.intoParts();
        } catch {
            return RestResponsesEndpoint.invalidRequestResponse(
                message: String(describing: error),
                code: "invalid_request");
        }
        let workerHealthSnapshot: WorkerHealthSnapshot =
            responsesContext.responsesExecutor.workerHealthSnapshot();
        if workerHealthSnapshot.status.isReady() == false {
            return RestResponsesEndpoint.workerUnavailableResponse();
        }
        guard let resolvedModelId: String = RestResponsesEndpoint.resolveAvailableGenerationModelId(
            requestedModelId: requestParts.model,
            workerHealthSnapshot: workerHealthSnapshot,
            resolvedRuntimeConfig: responsesContext.resolvedRuntimeConfig) else {
            return RestResponsesEndpoint.invalidRequestParameterResponse(
                message: "model is not loaded by the local worker",
                parameter: "model",
                code: "model_not_found");
        }
        if RestResponsesEndpoint.discoveredModelSupportsResponses(
            responsesContext, resolvedModelId: resolvedModelId) == false {
            return RestResponsesEndpoint.invalidRequestParameterResponse(
                message: "the requested model does not support Responses generation",
                parameter: "model",
                code: "model_capability_mismatch");
        }
        guard let requestIdentifier: UInt64 = responsesContext.requestIdAllocator.allocate() else {
            return RestResponsesEndpoint.serviceUnavailableResponse(
                message: "the local request identifier space is exhausted",
                code: "request_id_exhausted");
        }
        let chatGenerationCommand: ChatGenerationCommand;
        do {
            chatGenerationCommand = try OpenAiResponsesTranslation.translateRequestParts(
                requestParts,
                requestId: RequestId(rawRequestId: requestIdentifier));
        } catch {
            return RestResponsesEndpoint.invalidRequestResponse(
                message: String(describing: error),
                code: "invalid_request");
        }
        // Policy defaults fill omissions only, so presence comes from the
        // public parts before translation normalizes the budget fields.
        let generationSettings: ChatGenerationSettings = RequestGenerationDefaults.apply(
            resolvedRuntimeConfig: responsesContext.resolvedRuntimeConfig,
            modelId: resolvedModelId,
            settingsPresence: RequestGenerationSettingsPresence(
                maximumOutputTokensRequested: requestParts.requestedMaximumOutputTokens != nil,
                temperatureRequested: requestParts.temperature != nil,
                topPRequested: requestParts.topP != nil),
            generationSettings: chatGenerationCommand.settings);
        let admittedCommand: ChatGenerationCommand = ChatGenerationCommand(
            requestId: chatGenerationCommand.requestId,
            model: resolvedModelId,
            messages: chatGenerationCommand.messages,
            tools: chatGenerationCommand.tools,
            toolChoice: chatGenerationCommand.toolChoice,
            settings: generationSettings,
            qwenThinkingChannelSeed: QwenThinkingChannelSeed.load(
                resolvedRuntimeConfig: responsesContext.resolvedRuntimeConfig,
                instancePaths: responsesContext.instancePaths,
                modelId: resolvedModelId),
            structuredGeneration: chatGenerationCommand.structuredGeneration);
        let streamEvents: Array<ChatGenerationStreamEvent>;
        do {
            streamEvents = try responsesContext.responsesExecutor.startChatGeneration(admittedCommand);
        } catch let startError as GenerationStartError {
            return RestResponsesEndpoint.generationStartFailureResponse(startError);
        } catch {
            return RestResponsesEndpoint.workerUnavailableResponse();
        }
        guard let createdAtUnixSeconds: UInt64 = RestResponsesTimestamp.currentUnixSeconds() else {
            return RestResponsesEndpoint.serviceUnavailableResponse(
                message: "the local server could not timestamp the response",
                code: "response_timestamp_failed");
        }
        let responseId: String =
            "resp_\(CompletionIdNamespace.shared.rawValue)-\(requestIdentifier)";
        let responsesResponse: RestHttpResponse;
        if requestParts.stream {
            responsesResponse = RestResponsesEndpoint.streamingResponse(
                streamEvents,
                responseId: responseId,
                createdAtUnixSeconds: createdAtUnixSeconds,
                modelId: resolvedModelId,
                instructions: requestParts.instructions,
                requestConfiguration: requestParts.responseConfiguration(),
                reasoningExcluded: requestParts.reasoningExcluded);
        } else {
            responsesResponse = RestResponsesEndpoint.nonStreamingResponse(
                streamEvents,
                responseId: responseId,
                createdAtUnixSeconds: createdAtUnixSeconds,
                modelId: resolvedModelId,
                instructions: requestParts.instructions,
                requestConfiguration: requestParts.responseConfiguration(),
                reasoningExcluded: requestParts.reasoningExcluded,
                structuredOutput: requestParts.structuredOutput);
        }
        return RestResponsesEndpoint.attachingUnenforcedWarning(
            responsesResponse,
            structuredOutput: requestParts.structuredOutput);
    }

    /// Resolves the requested model id against the discovered catalog and the
    /// resident worker model: a provider-prefixed alias canonicalizes to the
    /// discovered id, and either a resident or a discovered model is
    /// servable (the worker loads on demand).
    private static func resolveAvailableGenerationModelId(
        requestedModelId: String,
        workerHealthSnapshot: WorkerHealthSnapshot,
        resolvedRuntimeConfig: ResolvedRuntimeConfig
    ) -> String? {
        let knownModelIds: Array<String> = resolvedRuntimeConfig.discoveredModels.map(
            { (discoveredModel: DiscoveryDiscoveredModel) -> String in
                return discoveredModel.modelId;
            });
        let resolvedModelId: String = ModelIdentity.resolveModelId(
            requestedModelId: requestedModelId,
            knownModelIds: knownModelIds);
        let isReadyModel: Bool = workerHealthSnapshot.readyModelId == resolvedModelId;
        let isDiscoveredModel: Bool = knownModelIds.contains(resolvedModelId);
        return (isReadyModel || isDiscoveredModel) ? resolvedModelId : nil;
    }

    private static func discoveredModelSupportsResponses(
        _ responsesContext: RestResponsesRouteContext,
        resolvedModelId: String
    ) -> Bool {
        guard let discoveredModel: DiscoveryDiscoveredModel = responsesContext.resolvedRuntimeConfig.discoveredModels
            .first(where: { (candidateModel: DiscoveryDiscoveredModel) -> Bool in
                return candidateModel.modelId == resolvedModelId;
            }) else {
            return true;
        }
        if case .chat = discoveredModel.capabilities {
            return true;
        }
        return false;
    }

    private static func nonStreamingResponse(
        _ streamEvents: Array<ChatGenerationStreamEvent>,
        responseId: String,
        createdAtUnixSeconds: UInt64,
        modelId: String,
        instructions: String?,
        requestConfiguration: OpenAiResponseRequestConfiguration,
        reasoningExcluded: Bool,
        structuredOutput: OpenAiStructuredOutput?
    ) -> RestHttpResponse {
        var responseCollector: OpenAiResponsesCollector = OpenAiResponsesCollector(
            responseId: responseId,
            createdAtUnixSeconds: createdAtUnixSeconds,
            modelId: modelId,
            instructions: instructions,
            requestConfiguration: requestConfiguration);
        for streamEvent: ChatGenerationStreamEvent in streamEvents {
            if case let .failed(failureReason) = streamEvent,
               case let .contextLengthExceeded(actualTotalContextTokens, maximumContextTokens) = failureReason {
                // A context overrun is the caller's prompt shape, so it is a
                // 400 attributed to the input parameter.
                return RestResponsesEndpoint.invalidRequestParameterResponse(
                    message: "requested context uses \(actualTotalContextTokens) tokens, "
                        + "exceeding the \(maximumContextTokens)-token model context window",
                    parameter: "input",
                    code: "context_length_exceeded");
            }
            if case let .completed(
                promptTokenCount, generatedTokenCount, reasoningTokenCount, cachedTokenCount,
                completionReason) = streamEvent {
                if structuredOutput != nil {
                    responseCollector.replaceOutputTextWithExtractedJson();
                }
                do {
                    let assembledResponse: OpenAiResponse = try responseCollector.intoResponse(
                        inputTokenCount: promptTokenCount,
                        outputTokenCount: generatedTokenCount,
                        cachedInputTokenCount: cachedTokenCount,
                        reasoningTokenCount: reasoningTokenCount,
                        completionReason: completionReason);
                    return (try? RestHttpResponse.json(
                        statusCode: 200, wireValue: assembledResponse.wireValue()))
                        ?? RestResponsesEndpoint.assemblyFailureResponse(
                            "the local server could not assemble the chat completion");
                } catch {
                    return RestResponsesEndpoint.assemblyFailureResponse(String(describing: error));
                }
            }
            if case .reasoningFragment = streamEvent, reasoningExcluded {
                // The model still thought; the token count still lands in
                // usage via the Completed event. Withhold only the
                // client-visible reasoning item.
                continue;
            }
            do {
                try responseCollector.ingestEvent(streamEvent);
            } catch {
                return RestResponsesEndpoint.assemblyFailureResponse(String(describing: error));
            }
        }
        // The stream ended before the terminal frame: the worker is gone.
        return RestResponsesEndpoint.workerUnavailableResponse();
    }

    private static func streamingResponse(
        _ streamEvents: Array<ChatGenerationStreamEvent>,
        responseId: String,
        createdAtUnixSeconds: UInt64,
        modelId: String,
        instructions: String?,
        requestConfiguration: OpenAiResponseRequestConfiguration,
        reasoningExcluded: Bool
    ) -> RestHttpResponse {
        var streamEncoder: OpenAiResponsesStreamEncoder = OpenAiResponsesStreamEncoder(
            responseId: responseId,
            createdAtUnixSeconds: createdAtUnixSeconds,
            modelId: modelId,
            instructions: instructions,
            requestConfiguration: requestConfiguration,
            reasoningExcluded: reasoningExcluded);
        do {
            var streamBodyText: String = String();
            for encodedEvent: OpenAiResponseStreamEvent in streamEncoder.initialEvents() {
                streamBodyText += try RestResponsesEndpoint.encodedFrame(encodedEvent);
            }
            for streamEvent: ChatGenerationStreamEvent in streamEvents {
                for encodedEvent: OpenAiResponseStreamEvent in try streamEncoder.encode(streamEvent) {
                    streamBodyText += try RestResponsesEndpoint.encodedFrame(encodedEvent);
                }
            }
            // A worker channel that closes before the terminal frame becomes
            // the synthetic unavailable error, mirroring the Rust recv-None
            // stream unfold branch.
            if streamEncoder.isTerminal() == false {
                for encodedEvent: OpenAiResponseStreamEvent in try streamEncoder.encode(
                    .streamError(.workerUnavailable)) {
                    streamBodyText += try RestResponsesEndpoint.encodedFrame(encodedEvent);
                }
            }
            return RestHttpResponse(
                statusCode: 200,
                contentType: RestResponsesEndpoint.eventStreamContentType,
                bodyBytes: Data(streamBodyText.utf8));
        } catch {
            // Our own wire types are the only thing being serialized; a
            // failure here is the server's fault, mirroring the Rust
            // mid-stream serialization failure path.
            return RestResponsesEndpoint.serviceUnavailableResponse(
                message: "the local server could not start the Responses stream",
                code: "responses_stream_encoding_failed");
        }
    }

    private static func encodedFrame(
        _ encodedEvent: OpenAiResponseStreamEvent
    ) throws -> String {
        let serializedPayload: String = try encodedEvent.wireValue().serializedText;
        return "event: \(encodedEvent.eventType())\ndata: \(serializedPayload)\n\n";
    }

    /// Discloses prompt-injected JSON on every structured-output answer:
    /// success there is prompt cooperation, not grammar enforcement.
    private static func attachingUnenforcedWarning(
        _ responsesResponse: RestHttpResponse,
        structuredOutput: OpenAiStructuredOutput?
    ) -> RestHttpResponse {
        guard let structuredOutput = structuredOutput else {
            return responsesResponse;
        }
        return RestHttpResponse(
            statusCode: responsesResponse.statusCode,
            contentType: responsesResponse.contentType,
            bodyBytes: responsesResponse.bodyBytes,
            additionalHeaderLines: responsesResponse.additionalHeaderLines
                + ["Warning: \(structuredOutput.unenforcedWarningHeader())"]);
    }

    private static func generationStartFailureResponse(
        _ startError: GenerationStartError
    ) -> RestHttpResponse {
        switch (startError) {
        case .capacityUnavailable:
            return RestResponsesEndpoint.jsonResponse(
                statusCode: 429,
                errorResponse: OpenAiErrorResponse.capacityUnavailable(
                    message: "the generation queue is full"));
        case let .modelLoadFailed(modelLoadFailureReason):
            return RestResponsesEndpoint.jsonResponse(
                statusCode: 503,
                errorResponse: OpenAiErrorResponse.modelLoadFailed(
                    modelLoadFailureReason: modelLoadFailureReason));
        case let .requestTooLarge(actualIpcMessageBytes, maximumIpcMessageBytes):
            return RestResponsesEndpoint.invalidRequestResponse(
                message: "the request expands to \(actualIpcMessageBytes) bytes for local processing, "
                    + "exceeding the \(maximumIpcMessageBytes)-byte limit; reduce image sizes or "
                    + "conversation history",
                code: "request_too_large",
                statusCodeOverride: 413);
        case .workerUnavailable:
            return RestResponsesEndpoint.workerUnavailableResponse();
        }
    }

    private static func assemblyFailureResponse(_ assemblyMessage: String) -> RestHttpResponse {
        return RestResponsesEndpoint.serviceUnavailableResponse(
            message: assemblyMessage,
            code: "response_generation_failed");
    }

    private static func workerUnavailableResponse() -> RestHttpResponse {
        return RestResponsesEndpoint.serviceUnavailableResponse(
            message: "the local worker is unavailable",
            code: "worker_unavailable");
    }

    private static func serviceUnavailableResponse(
        message: String,
        code: String
    ) -> RestHttpResponse {
        return RestResponsesEndpoint.jsonResponse(
            statusCode: 503,
            errorResponse: OpenAiErrorResponse.serviceUnavailable(message: message, code: code));
    }

    private static func invalidRequestResponse(
        message: String,
        code: String,
        statusCodeOverride: Int? = nil
    ) -> RestHttpResponse {
        return RestResponsesEndpoint.jsonResponse(
            statusCode: statusCodeOverride ?? 400,
            errorResponse: OpenAiErrorResponse.invalidRequest(
                message: message, parameter: nil, code: code));
    }

    private static func invalidRequestParameterResponse(
        message: String,
        parameter: String,
        code: String
    ) -> RestHttpResponse {
        return RestResponsesEndpoint.jsonResponse(
            statusCode: 400,
            errorResponse: OpenAiErrorResponse.invalidRequest(
                message: message, parameter: parameter, code: code));
    }

    private static func jsonResponse(
        statusCode: Int,
        errorResponse: OpenAiErrorResponse
    ) -> RestHttpResponse {
        return (try? RestHttpResponse.json(statusCode: statusCode, wireValue: errorResponse.wireValue()))
            ?? RestHttpResponse.text(statusCode: 500, body: "the response could not be serialized");
    }
}
