import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;

@testable import Supervisor;

/// A chat executor scripted per journey: it records every admitted command
/// and answers with either a start failure or the scripted ordered events.
final class ScriptedChatExecutor: ChatGenerationExecuting, @unchecked Sendable {

    let scriptedHealthSnapshot: WorkerHealthSnapshot;
    let scriptedStreamEvents: Array<ChatGenerationStreamEvent>;
    let scriptedStartError: GenerationStartError?;
    private let recordsLock: NSLock;
    private var recordedCommands: Array<ChatGenerationCommand>;

    init(
        healthSnapshot: WorkerHealthSnapshot,
        streamEvents: Array<ChatGenerationStreamEvent> = Array(),
        startError: GenerationStartError? = nil
    ) {
        self.scriptedHealthSnapshot = healthSnapshot;
        self.scriptedStreamEvents = streamEvents;
        self.scriptedStartError = startError;
        self.recordsLock = NSLock();
        self.recordedCommands = Array();
    }

    var receivedCommands: Array<ChatGenerationCommand> {
        self.recordsLock.lock();
        defer { self.recordsLock.unlock(); }
        return self.recordedCommands;
    }

    func startChatGeneration(
        _ generationCommand: ChatGenerationCommand
    ) throws -> Array<ChatGenerationStreamEvent> {
        self.recordsLock.lock();
        self.recordedCommands.append(generationCommand);
        self.recordsLock.unlock();
        if let scriptedStartError = self.scriptedStartError {
            throw scriptedStartError;
        }
        return self.scriptedStreamEvents;
    }

    func workerHealthSnapshot() -> WorkerHealthSnapshot {
        return self.scriptedHealthSnapshot;
    }
}

/// Shared support for the REST chat completion journeys: resolved
/// configuration, a chat-enabled serving route table, direct handler
/// invocation, and JSON/SSE response decoding.
enum RestChatJourneySupport {

    static let nonStreamingModelId: String = "astronomical/non-streaming-test-model";
    static let streamingModelId: String = "astronomical/streaming-test-model";
    static let negativeModelId: String = "astronomical/negative-chat-test-worker";

    static func readyChatCapabilities() -> WorkerModelCapabilities {
        return WorkerModelCapabilities.from(chatCapabilities: ChatModelCapabilities(
            supportsReasoning: true,
            supportsToolCalls: true,
            hasVision: true,
            maxInputTokens: 241_664,
            maxOutputTokens: 20_480,
            contextWindow: 262_144));
    }

    static func makeResolvedConfig(
        discoveredModels: Array<DiscoveryDiscoveredModel> = Array()
    ) throws -> ResolvedRuntimeConfig {
        var modelPolicyCatalog: Dictionary<String, RuntimeModelPolicy> = Dictionary();
        for discoveredModel: DiscoveryDiscoveredModel in discoveredModels {
            modelPolicyCatalog[discoveredModel.modelId] = try ResolvedModelPolicyCatalog.resolve(
                userConfig: try emptyConfig(),
                discoveredModels: discoveredModels,
                artifactContextWindows: Dictionary<String, UInt32>())[discoveredModel.modelId];
        }
        return ResolvedRuntimeConfig(
            configurationGeneration: "0123456789abcdef0123456789abcdef",
            workerExecutablePath: FilePath(string: "/opt/astronomical/bin/astronomical-inference-worker"),
            discoveredModels: discoveredModels,
            modelDiscoveryDiagnostics: Array<DiscoveryModelDiscoveryDiagnostic>(),
            configuredModelDirectories: Array<FilePath>(),
            modelPolicyCatalog: modelPolicyCatalog,
            unmatchedModelConfigIds: Array<String>(),
            maximumMlxMemoryBytes: nil,
            performanceAttributionEnabled: false,
            completionAttributionEnabled: false,
            experimentalQwenThinkingChannelSeedEnabled: false,
            persistentPromptCacheEnabled: true,
            configuredPersistentPromptCacheEnabled: nil,
            configuredPromptCacheMaximumSizeBytes: 50_000_000_000,
            promptCacheConfig: PromptCacheConfig(
                rootDirectory: FilePath(string: "/state/prompt-cache"),
                maximumSizeBytes: 50_000_000_000),
            bindAddress: "127.0.0.1:0",
            bindEndpoint: SocketEndpoint.loopback(port: 0),
            loggingConfig: LoggingConfig(
                directory: FilePath(string: "/state/logs"),
                level: LogLevel.warn,
                retainedFiles: 7));
    }

    private static func emptyConfig() throws -> AstronomicalConfig {
        let temporaryStateDirectory: String = NSTemporaryDirectory()
            + "restchat-\(UUID().uuidString.prefix(8))";
        try FileManager.default.createDirectory(
            atPath: temporaryStateDirectory, withIntermediateDirectories: true);
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            FilePath(string: temporaryStateDirectory),
            runtimeInstance: AstronomicalRuntimeInstance.development);
        try FileManager.default.createDirectory(
            atPath: instancePaths.stateDirectory.string,
            withIntermediateDirectories: true);
        defer { try? FileManager.default.removeItem(atPath: temporaryStateDirectory); }
        return try AstronomicalConfig.loadFromInstancePaths(instancePaths);
    }

    /// Builds the serving route table with the chat route attached, exactly
    /// as the daemon wires it, over a scripted executor.
    static func chatRouteTable(
        resolvedRuntimeConfig: ResolvedRuntimeConfig,
        chatExecutor: ChatGenerationExecuting
    ) throws -> RestRouteTable {
        return RestEndpointRoutes.servingRouteTable(
            resolvedRuntimeConfig: resolvedRuntimeConfig,
            workerHealthState: WorkerHealthState(),
            instancePaths: AstronomicalInstancePaths.forExplicitStateDirectory(
                FilePath(string: "/rest-chat-journey-state"),
                defaultBindAddress: SocketEndpoint.loopback(port: 0)),
            buildIdentity: ApplicationBuildIdentity(
                version: "0.0.0-test", buildNumber: 0, commit: "journey", isDirty: false),
            chatContext: RestChatRouteContext(
                chatExecutor: chatExecutor,
                requestIdAllocator: ChatRequestIdAllocator(),
                resolvedRuntimeConfig: resolvedRuntimeConfig,
                instancePaths: AstronomicalInstancePaths.forExplicitStateDirectory(
                    FilePath(string: "/rest-chat-journey-state"),
                    defaultBindAddress: SocketEndpoint.loopback(port: 0))));
    }

    /// Posts one request body straight to the routed handler at `routePath`,
    /// the in-process equivalent of the Rust oneshot journeys.
    static func postChat(
        routeTable: RestRouteTable,
        routePath: String = RestChatCompletionEndpoint.routePath,
        requestBody: String
    ) throws -> RestHttpResponse {
        let routeOutcome: RestRouteOutcome = routeTable.outcome(
            method: "POST",
            path: routePath);
        guard case let .handler(routeHandler) = routeOutcome else {
            throw RestChatJourneyFailure.chatRouteMissing(routeOutcome);
        }
        let chatRequest: RestHttpRequest = RestHttpRequest(
            method: "POST",
            path: routePath,
            requestTarget: routePath,
            headersByLowercasedName: ["content-type": "application/json"],
            bodyBytes: Data(requestBody.utf8));
        return try routeHandler(chatRequest);
    }

    static func responseStatusCode(_ chatResponse: RestHttpResponse) -> Int {
        return chatResponse.statusCode;
    }

    static func responseText(_ chatResponse: RestHttpResponse) -> String {
        return String(decoding: chatResponse.bodyBytes, as: UTF8.self);
    }

    static func responseHeaderValue(
        _ chatResponse: RestHttpResponse,
        headerName: String
    ) -> String? {
        for headerLine: String in chatResponse.additionalHeaderLines {
            let headerParts: Array<Substring> = headerLine.split(separator: ":", maxSplits: 1);
            if headerParts.count == 2,
               headerParts[0].trimmingCharacters(in: .whitespaces).caseInsensitiveCompare(headerName) == .orderedSame {
                return headerParts[1].trimmingCharacters(in: .whitespaces);
            }
        }
        return nil;
    }

    static func decodeObjectEnvelope(_ chatResponse: RestHttpResponse) throws -> [String: Any] {
        guard let envelopeObject: [String: Any] = try JSONSerialization.jsonObject(
            with: chatResponse.bodyBytes) as? [String: Any] else {
            throw RestChatJourneyFailure.nonJsonBody;
        }
        return envelopeObject;
    }

    static func requireErrorObject(_ envelope: [String: Any]) throws -> [String: Any] {
        guard let errorObject: [String: Any] = envelope["error"] as? [String: Any] else {
            throw RestChatJourneyFailure.missingErrorObject;
        }
        return errorObject;
    }

    static func requireChoice(_ envelope: [String: Any]) throws -> [String: Any] {
        guard let choices: Array<Any> = envelope["choices"] as? Array<Any>,
              choices.count > 0,
              let firstChoice: [String: Any] = choices[0] as? [String: Any] else {
            throw RestChatJourneyFailure.missingChoice;
        }
        return firstChoice;
    }

    static func requireMessage(_ choice: [String: Any]) throws -> [String: Any] {
        guard let assistantMessage: [String: Any] = choice["message"] as? [String: Any] else {
            throw RestChatJourneyFailure.missingAssistantMessage;
        }
        return assistantMessage;
    }

    static func requireUsage(_ envelope: [String: Any]) throws -> [String: Any] {
        guard let usageObject: [String: Any] = envelope["usage"] as? [String: Any] else {
            throw RestChatJourneyFailure.missingUsage;
        }
        return usageObject;
    }

    /// The parsed data-payload stream of one SSE chat response.
    struct ParsedChatSseStream {

        let payloads: Array<[String: Any]>;
        let sawDone: Bool;

        static func parse(_ responseText: String) -> ParsedChatSseStream {
            var parsedPayloads: Array<[String: Any]> = Array();
            var parsedSawDone: Bool = false;
            for responseLine: Substring in responseText.split(separator: "\n", omittingEmptySubsequences: false) {
                guard responseLine.hasPrefix("data: ") else {
                    continue;
                }
                let dataPayload: Substring = responseLine.dropFirst("data: ".count);
                if dataPayload == "[DONE]" {
                    parsedSawDone = true;
                    continue;
                }
                guard let payloadObject: [String: Any] = try? JSONSerialization.jsonObject(
                    with: Data(dataPayload.utf8)) as? [String: Any] else {
                    continue;
                }
                parsedPayloads.append(payloadObject);
            }
            return ParsedChatSseStream(payloads: parsedPayloads, sawDone: parsedSawDone);
        }

        func deltaText(forKey fieldName: String) -> String {
            var collectedText: String = String();
            for payload: [String: Any] in self.payloads {
                if let deltaText: String = Self.deltaObject(payload)?[fieldName] as? String {
                    collectedText += deltaText;
                }
            }
            return collectedText;
        }

        func finishReason() -> String? {
            for payload: [String: Any] in self.payloads.reversed() {
                if let finishReason: String = Self.choiceObject(payload)?["finish_reason"] as? String {
                    return finishReason;
                }
            }
            return nil;
        }

        func toolCallDeltas() -> Array<[String: Any]> {
            var collectedToolCalls: Array<[String: Any]> = Array();
            for payload: [String: Any] in self.payloads {
                guard let toolCalls: Array<Any> = Self.deltaObject(payload)?["tool_calls"] as? Array<Any> else {
                    continue;
                }
                for toolCall: Any in toolCalls {
                    if let toolCallObject: [String: Any] = toolCall as? [String: Any] {
                        collectedToolCalls.append(toolCallObject);
                    }
                }
            }
            return collectedToolCalls;
        }

        func terminalPayload(finishReason finishReasonName: String) -> [String: Any]? {
            return self.payloads.first { (payload: [String: Any]) -> Bool in
                return Self.choiceObject(payload)?["finish_reason"] as? String == finishReasonName;
            };
        }

        private static func choiceObject(_ payload: [String: Any]) -> [String: Any]? {
            guard let choices: Array<Any> = payload["choices"] as? Array<Any>,
                  choices.count > 0 else {
                return nil;
            }
            return choices[0] as? [String: Any];
        }

        private static func deltaObject(_ payload: [String: Any]) -> [String: Any]? {
            return Self.choiceObject(payload)?["delta"] as? [String: Any];
        }
    }
}

/// Typed journey failures so assertions point at the broken expectation.
enum RestChatJourneyFailure: Error {
    case chatRouteMissing(RestRouteOutcome);
    case nonJsonBody;
    case missingErrorObject;
    case missingChoice;
    case missingAssistantMessage;
    case missingUsage;
    case expectedObject(String);
    case expectedArray(String);
    case expectedString(String);
}
