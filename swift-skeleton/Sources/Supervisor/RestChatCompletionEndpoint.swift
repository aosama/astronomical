import Foundation;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;

/// Everything the REST chat completion route needs: the generation executor
/// seam, the supervisor-local request-id space, the live resolved
/// configuration, and the instance paths the optional thinking-channel seed
/// reads from.
public struct RestChatRouteContext: @unchecked Sendable {

    let chatExecutor: any ChatGenerationExecuting;
    let requestIdAllocator: ChatRequestIdAllocator;
    let resolvedRuntimeConfig: ResolvedRuntimeConfig;
    let instancePaths: AstronomicalInstancePaths;

    public init(
        chatExecutor: any ChatGenerationExecuting,
        requestIdAllocator: ChatRequestIdAllocator,
        resolvedRuntimeConfig: ResolvedRuntimeConfig,
        instancePaths: AstronomicalInstancePaths
    ) {
        self.chatExecutor = chatExecutor;
        self.requestIdAllocator = requestIdAllocator;
        self.resolvedRuntimeConfig = resolvedRuntimeConfig;
        self.instancePaths = instancePaths;
    }
}

/// The POST /v1/chat/completions endpoint.
///
/// Port of apps/supervisor/src/openai_chat_endpoint.rs: decode and validate
/// the public request, gate on worker health, resolve the requested model,
/// translate into the IPC command, fill defaults, admit the generation, and
/// answer with either one JSON body or the buffered OpenAI SSE frame
/// sequence.
enum RestChatCompletionEndpoint {

    static let routeMethod: String = "POST";
    static let routePath: String = "/v1/chat/completions";
    private static let eventStreamContentType: String = "text/event-stream";

    static func handle(
        _ request: RestHttpRequest,
        chatContext: RestChatRouteContext
    ) -> RestHttpResponse {
        let attributionStart: ContinuousClock.Instant? = DaemonIpcPerformanceAttribution.startedOperation(
            operationName: "rest_chat_completion",
            performanceAttributionEnabled: chatContext.resolvedRuntimeConfig.performanceAttributionEnabled);
        defer {
            DaemonIpcPerformanceAttribution.finishedOperation(
                operationName: "rest_chat_completion",
                operationStart: attributionStart,
                operationOutcome: "served",
                performanceAttributionEnabled: chatContext.resolvedRuntimeConfig.performanceAttributionEnabled);
        }
        let requestDiagnosticSnapshot: OpenAiChatRequestDiagnosticSnapshot =
            OpenAiChatRequestDiagnostics.buildRequestDiagnosticSnapshot(
                requestBodyBytes: request.bodyBytes);
        let requestInfoDiagnosticSnapshot: OpenAiChatRequestInfoDiagnosticSnapshot =
            OpenAiChatRequestDiagnostics.buildRequestInfoDiagnosticSnapshot(
                requestBodyBytes: request.bodyBytes);
        OpenAiChatRequestDiagnostics.logRequestCapture(requestDiagnosticSnapshot);
        let chatCompletionRequest: OpenAiChatCompletionRequest;
        do {
            let requestWireValue: JsonWireValue = try JsonWireParser.parseDocument(
                documentBytes: request.bodyBytes);
            chatCompletionRequest = try OpenAiChatCompletionRequest.decoded(wireValue: requestWireValue);
        } catch {
            OpenAiChatRequestDiagnostics.logRequestRejection(
                reason: "request body is not valid JSON",
                diagnosticSnapshot: requestDiagnosticSnapshot,
                infoDiagnosticSnapshot: requestInfoDiagnosticSnapshot);
            return RestChatCompletionEndpoint.invalidRequestResponse(
                message: "request body is not valid JSON: \(error)",
                code: "invalid_json");
        }
        do {
            try chatCompletionRequest.validate();
        } catch let validationRejection as OpenAiChatCompletionValidationError {
            OpenAiChatRequestDiagnostics.logRequestRejection(
                reason: validationRejection.errorDescription ?? String(describing: validationRejection),
                diagnosticSnapshot: requestDiagnosticSnapshot,
                infoDiagnosticSnapshot: requestInfoDiagnosticSnapshot);
            return RestChatCompletionEndpoint.invalidRequestResponse(
                message: validationRejection.errorDescription
                    ?? String(describing: validationRejection),
                code: "invalid_request");
        } catch {
            OpenAiChatRequestDiagnostics.logRequestRejection(
                reason: String(describing: error),
                diagnosticSnapshot: requestDiagnosticSnapshot,
                infoDiagnosticSnapshot: requestInfoDiagnosticSnapshot);
            return RestChatCompletionEndpoint.invalidRequestResponse(
                message: String(describing: error),
                code: "invalid_request");
        }
        let workerHealthSnapshot: WorkerHealthSnapshot = chatContext.chatExecutor.workerHealthSnapshot();
        if workerHealthSnapshot.status.isReady() == false {
            return RestChatCompletionEndpoint.workerUnavailableResponse();
        }
        let requestedModelId: String = chatCompletionRequest.model();
        guard let resolvedModelId: String = RestChatCompletionEndpoint.resolveAvailableGenerationModelId(
            requestedModelId: requestedModelId,
            workerHealthSnapshot: workerHealthSnapshot,
            resolvedRuntimeConfig: chatContext.resolvedRuntimeConfig) else {
            return RestChatCompletionEndpoint.invalidRequestParameterResponse(
                message: "model is not loaded by the local worker",
                parameter: "model",
                code: "model_not_found");
        }
        if discoveredModelSupportsChat(chatContext, resolvedModelId: resolvedModelId) == false {
            return RestChatCompletionEndpoint.invalidRequestParameterResponse(
                message: "the requested model does not support chat generation",
                parameter: "model",
                code: "model_capability_mismatch");
        }
        let requestParts: OpenAiChatCompletionRequestParts;
        do {
            requestParts = try chatCompletionRequest.intoParts();
        } catch let validationRejection as OpenAiChatCompletionValidationError {
            return RestChatCompletionEndpoint.invalidRequestResponse(
                message: validationRejection.errorDescription
                    ?? String(describing: validationRejection),
                code: "invalid_request");
        } catch {
            return RestChatCompletionEndpoint.invalidRequestResponse(
                message: String(describing: error),
                code: "invalid_request");
        }
        guard let requestIdentifier: UInt64 = chatContext.requestIdAllocator.allocate() else {
            return RestChatCompletionEndpoint.serviceUnavailableResponse(
                message: "the local request identifier space is exhausted",
                code: "request_id_exhausted");
        }
        let chatGenerationCommand: ChatGenerationCommand;
        do {
            chatGenerationCommand = try OpenAiChatTranslation.translateRequestParts(
                requestParts,
                requestId: RequestId(rawRequestId: requestIdentifier));
        } catch let translationRejection as OpenAiChatTranslationError {
            return RestChatCompletionEndpoint.invalidRequestResponse(
                message: String(describing: translationRejection),
                code: "invalid_request");
        } catch {
            return RestChatCompletionEndpoint.invalidRequestResponse(
                message: String(describing: error),
                code: "invalid_request");
        }
        // Policy defaults fill omissions only, so presence comes from the
        // public parts before translation normalizes the budget fields.
        let generationSettings: ChatGenerationSettings = RequestGenerationDefaults.apply(
            resolvedRuntimeConfig: chatContext.resolvedRuntimeConfig,
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
                resolvedRuntimeConfig: chatContext.resolvedRuntimeConfig,
                instancePaths: chatContext.instancePaths,
                modelId: resolvedModelId),
            structuredGeneration: chatGenerationCommand.structuredGeneration);
        let streamEvents: Array<ChatGenerationStreamEvent>;
        do {
            streamEvents = try chatContext.chatExecutor.startChatGeneration(admittedCommand);
        } catch let startError as GenerationStartError {
            return RestChatCompletionEndpoint.generationStartFailureResponse(startError);
        } catch {
            return RestChatCompletionEndpoint.workerUnavailableResponse();
        }
        guard let createdAtUnixSeconds: UInt64 = RestChatCompletionEndpoint.currentUnixSeconds() else {
            return RestChatCompletionEndpoint.serviceUnavailableResponse(
                message: "the local server could not timestamp the chat stream",
                code: "chat_stream_timestamp_failed");
        }
        let completionId: String =
            "chatcmpl-\(CompletionIdNamespace.shared.rawValue)-\(requestIdentifier)";
        let chatResponse: RestHttpResponse;
        if requestParts.stream {
            chatResponse = RestChatCompletionEndpoint.streamingResponse(
                streamEvents,
                streamEncoder: OpenAiChatStreamEncoder(
                    requestId: requestIdentifier,
                    completionId: completionId,
                    createdUnixSeconds: createdAtUnixSeconds,
                    modelId: resolvedModelId,
                    includesUsage: requestParts.includesUsageInStream,
                    reasoningExcluded: requestParts.reasoningExcluded));
        } else {
            chatResponse = RestChatCompletionEndpoint.nonStreamingResponse(
                streamEvents,
                completionId: completionId,
                createdUnixSeconds: createdAtUnixSeconds,
                modelId: resolvedModelId,
                reasoningExcluded: requestParts.reasoningExcluded,
                structuredOutput: requestParts.structuredOutput);
        }
        return RestChatCompletionEndpoint.attachingUnenforcedWarning(
            chatResponse,
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

    private static func discoveredModelSupportsChat(
        _ chatContext: RestChatRouteContext,
        resolvedModelId: String
    ) -> Bool {
        guard let discoveredModel: DiscoveryDiscoveredModel = chatContext.resolvedRuntimeConfig.discoveredModels
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
        completionId: String,
        createdUnixSeconds: UInt64,
        modelId: String,
        reasoningExcluded: Bool,
        structuredOutput: OpenAiStructuredOutput?
    ) -> RestHttpResponse {
        var chatCompletionCollector: OpenAiChatCompletionCollector = OpenAiChatCompletionCollector(
            completionId: completionId,
            createdUnixSeconds: createdUnixSeconds,
            modelId: modelId,
            reasoningExcluded: reasoningExcluded);
        for streamEvent: ChatGenerationStreamEvent in streamEvents {
            if case let .completed(
                promptTokenCount, generatedTokenCount, _, cachedTokenCount, completionReason) = streamEvent {
                if structuredOutput != nil {
                    chatCompletionCollector.replaceVisibleTextWithExtractedJson();
                }
                do {
                    let chatCompletionResponse: OpenAiChatCompletionResponse =
                        try chatCompletionCollector.intoResponse(
                            promptTokenCount: promptTokenCount,
                            generatedTokenCount: generatedTokenCount,
                            cachedTokenCount: cachedTokenCount,
                            completionReason: completionReason);
                    return RestChatCompletionEndpoint.jsonResponse(
                        statusCode: 200,
                        errorResponse: nil,
                        chatCompletionResponse: chatCompletionResponse);
                } catch {
                    return RestChatCompletionEndpoint.serviceUnavailableResponse(
                        message: "the local server could not assemble the chat completion",
                        code: "chat_completion_assembly_failed");
                }
            }
            if case let .failed(failureReason) = streamEvent {
                // A context overrun is the caller's prompt shape, so it is a
                // 400; every other worker failure is the local server's fault.
                let failureStatusCode: Int;
                if case .contextLengthExceeded = failureReason {
                    failureStatusCode = 400;
                } else {
                    failureStatusCode = 503;
                }
                return RestChatCompletionEndpoint.jsonResponse(
                    statusCode: failureStatusCode,
                    errorResponse: OpenAiChatCompletionCollector.failureEnvelope(failureReason),
                    chatCompletionResponse: nil);
            }
            if case .streamError = streamEvent {
                return RestChatCompletionEndpoint.serviceUnavailableResponse(
                    message: "the local worker became unavailable while processing the chat request",
                    code: "chat_worker_unavailable");
            }
            if chatCompletionCollector.ingestEvent(streamEvent) != nil {
                return RestChatCompletionEndpoint.serviceUnavailableResponse(
                    message: "the local worker became unavailable while processing the chat request",
                    code: "chat_worker_unavailable");
            }
        }
        // The stream ended before the terminal frame: the worker is gone.
        return RestChatCompletionEndpoint.serviceUnavailableResponse(
            message: "the local worker became unavailable while processing the chat request",
            code: "chat_worker_unavailable");
    }

    private static func streamingResponse(
        _ streamEvents: Array<ChatGenerationStreamEvent>,
        streamEncoder: OpenAiChatStreamEncoder
    ) -> RestHttpResponse {
        let streamBodyText: String;
        do {
            streamBodyText = try RestChatCompletionEndpoint.encodedStreamBody(
                streamEvents, streamEncoder: streamEncoder);
        } catch {
            // Our own wire types are the only thing being serialized; a
            // failure here is the server's fault, mirroring the Rust
            // initial-event encoding failure path.
            return RestChatCompletionEndpoint.serviceUnavailableResponse(
                message: "the local server could not start the chat stream",
                code: "chat_stream_encoding_failed");
        }
        return RestHttpResponse(
            statusCode: 200,
            contentType: RestChatCompletionEndpoint.eventStreamContentType,
            bodyBytes: Data(streamBodyText.utf8));
    }

    private static func encodedStreamBody(
        _ streamEvents: Array<ChatGenerationStreamEvent>,
        streamEncoder: OpenAiChatStreamEncoder
    ) throws -> String {
        var streamBodyText: String = try streamEncoder.initialFrame();
        for streamEvent: ChatGenerationStreamEvent in streamEvents {
            for encodedFrame: String in try streamEncoder.encode(streamEvent) {
                streamBodyText += encodedFrame;
            }
        }
        return streamBodyText;
    }

    /// Discloses prompt-injected JSON on every structured-output answer:
    /// success there is prompt cooperation, not grammar enforcement.
    private static func attachingUnenforcedWarning(
        _ chatResponse: RestHttpResponse,
        structuredOutput: OpenAiStructuredOutput?
    ) -> RestHttpResponse {
        guard let structuredOutput = structuredOutput else {
            return chatResponse;
        }
        return RestHttpResponse(
            statusCode: chatResponse.statusCode,
            contentType: chatResponse.contentType,
            bodyBytes: chatResponse.bodyBytes,
            additionalHeaderLines: chatResponse.additionalHeaderLines
                + ["Warning: \(structuredOutput.unenforcedWarningHeader())"]);
    }

    private static func generationStartFailureResponse(
        _ startError: GenerationStartError
    ) -> RestHttpResponse {
        switch (startError) {
        case .capacityUnavailable:
            return RestChatCompletionEndpoint.jsonResponse(
                statusCode: 429,
                errorResponse: OpenAiErrorResponse.capacityUnavailable(
                    message: "the generation queue is full"),
                chatCompletionResponse: nil);
        case let .modelLoadFailed(modelLoadFailureReason):
            return RestChatCompletionEndpoint.jsonResponse(
                statusCode: 503,
                errorResponse: OpenAiErrorResponse.modelLoadFailed(
                    modelLoadFailureReason: modelLoadFailureReason),
                chatCompletionResponse: nil);
        case let .requestTooLarge(actualIpcMessageBytes, maximumIpcMessageBytes):
            return RestChatCompletionEndpoint.invalidRequestResponse(
                message: "the request expands to \(actualIpcMessageBytes) bytes for local processing, "
                    + "exceeding the \(maximumIpcMessageBytes)-byte limit; reduce image sizes or "
                    + "conversation history",
                code: "request_too_large",
                statusCodeOverride: 413);
        case .workerUnavailable:
            return RestChatCompletionEndpoint.workerUnavailableResponse();
        }
    }

    private static func workerUnavailableResponse() -> RestHttpResponse {
        return RestChatCompletionEndpoint.serviceUnavailableResponse(
            message: "the local worker is unavailable",
            code: "worker_unavailable");
    }

    private static func serviceUnavailableResponse(
        message: String,
        code: String
    ) -> RestHttpResponse {
        return RestChatCompletionEndpoint.jsonResponse(
            statusCode: 503,
            errorResponse: OpenAiErrorResponse.serviceUnavailable(message: message, code: code),
            chatCompletionResponse: nil);
    }

    private static func invalidRequestResponse(
        message: String,
        code: String,
        statusCodeOverride: Int? = nil
    ) -> RestHttpResponse {
        return RestChatCompletionEndpoint.jsonResponse(
            statusCode: statusCodeOverride ?? 400,
            errorResponse: OpenAiErrorResponse.invalidRequest(
                message: message, parameter: nil, code: code),
            chatCompletionResponse: nil);
    }

    private static func invalidRequestParameterResponse(
        message: String,
        parameter: String,
        code: String
    ) -> RestHttpResponse {
        return RestChatCompletionEndpoint.jsonResponse(
            statusCode: 400,
            errorResponse: OpenAiErrorResponse.invalidRequest(
                message: message, parameter: parameter, code: code),
            chatCompletionResponse: nil);
    }

    private static func jsonResponse(
        statusCode: Int,
        errorResponse: OpenAiErrorResponse?,
        chatCompletionResponse: OpenAiChatCompletionResponse?
    ) -> RestHttpResponse {
        let wireValue: JsonWireValue = errorResponse?.wireValue()
            ?? chatCompletionResponse?.wireValue()
            ?? .null;
        return (try? RestHttpResponse.json(statusCode: statusCode, wireValue: wireValue))
            ?? RestHttpResponse.text(statusCode: 500, body: "the response could not be serialized");
    }

    private static func currentUnixSeconds() -> UInt64? {
        let secondsSinceEpoch: TimeInterval = Date().timeIntervalSince1970;
        guard secondsSinceEpoch >= 0 else {
            return nil;
        }
        return UInt64(secondsSinceEpoch);
    }
}

/// Per-process completion-id namespace: startup nanoseconds, process id, and
/// an in-process counter distinguish every daemon lifetime, mirroring the
/// Rust completion_id_namespace().
public final class CompletionIdNamespace: @unchecked Sendable {

    static let shared: CompletionIdNamespace = CompletionIdNamespace();

    let rawValue: String;

    private let counterLock: NSLock;
    private var nextInstanceCounter: UInt64;

    private init() {
        let startedAtUnixNanoseconds: UInt64 = UInt64(
            Date().timeIntervalSince1970 * 1_000_000_000);
        let processIdentifier: UInt64 = UInt64(ProcessInfo.processInfo.processIdentifier);
        self.counterLock = NSLock();
        self.nextInstanceCounter = 0;
        self.counterLock.lock();
        let applicationInstanceId: UInt64 = self.nextInstanceCounter;
        self.nextInstanceCounter += 1;
        self.counterLock.unlock();
        self.rawValue = String(format: "%llx-%llx-%llx",
            startedAtUnixNanoseconds, processIdentifier, applicationInstanceId);
    }
}
