import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;

@testable import Supervisor;

/// Shared support for the REST Responses journeys: a chat-enabled serving
/// route table wired with the Responses context, direct handler invocation,
/// and semantic SSE stream decoding.
enum RestResponsesJourneySupport {

    static let responsesModelId: String = "astronomical/responses-endpoint-test-model";

    static func readyResponsesCapabilities() -> WorkerModelCapabilities {
        return RestChatJourneySupport.readyChatCapabilities();
    }

    static func makeResolvedConfig(
        discoveredModels: Array<DiscoveryDiscoveredModel> = Array()
    ) throws -> ResolvedRuntimeConfig {
        return try RestChatJourneySupport.makeResolvedConfig(discoveredModels: discoveredModels);
    }

    /// Builds the serving route table with both generation routes attached,
    /// exactly as the daemon wires them, over one scripted executor.
    static func responsesRouteTable(
        resolvedRuntimeConfig: ResolvedRuntimeConfig,
        responsesExecutor: ChatGenerationExecuting
    ) throws -> RestRouteTable {
        let requestIdAllocator: ChatRequestIdAllocator = ChatRequestIdAllocator();
        let journeyInstancePaths: AstronomicalInstancePaths =
            AstronomicalInstancePaths.forExplicitStateDirectory(
                FilePath(string: "/rest-responses-journey-state"),
                defaultBindAddress: SocketEndpoint.loopback(port: 0));
        return RestEndpointRoutes.servingRouteTable(
            resolvedRuntimeConfig: resolvedRuntimeConfig,
            workerHealthState: WorkerHealthState(),
            instancePaths: journeyInstancePaths,
            buildIdentity: ApplicationBuildIdentity(
                version: "0.0.0-test", buildNumber: 0, commit: "journey", isDirty: false),
            chatContext: RestChatRouteContext(
                chatExecutor: responsesExecutor,
                requestIdAllocator: requestIdAllocator,
                resolvedRuntimeConfig: resolvedRuntimeConfig,
                instancePaths: journeyInstancePaths),
            responsesContext: RestResponsesRouteContext(
                responsesExecutor: responsesExecutor,
                requestIdAllocator: requestIdAllocator,
                resolvedRuntimeConfig: resolvedRuntimeConfig,
                instancePaths: journeyInstancePaths));
    }

    /// Posts one request body straight to the routed handler at
    /// /v1/responses, the in-process equivalent of the Rust oneshot journeys.
    static func postResponses(
        routeTable: RestRouteTable,
        requestBody: String
    ) throws -> RestHttpResponse {
        let routeOutcome: RestRouteOutcome = routeTable.outcome(
            method: "POST",
            path: RestResponsesEndpoint.routePath);
        guard case let .handler(routeHandler) = routeOutcome else {
            throw RestChatJourneyFailure.chatRouteMissing(routeOutcome);
        }
        let responsesRequest: RestHttpRequest = RestHttpRequest(
            method: "POST",
            path: RestResponsesEndpoint.routePath,
            requestTarget: RestResponsesEndpoint.routePath,
            headersByLowercasedName: ["content-type": "application/json"],
            bodyBytes: Data(requestBody.utf8));
        return try routeHandler(responsesRequest);
    }

    static func responseText(_ responsesResponse: RestHttpResponse) -> String {
        return String(decoding: responsesResponse.bodyBytes, as: UTF8.self);
    }

    static func responseHeaderValue(
        _ responsesResponse: RestHttpResponse,
        headerName: String
    ) -> String? {
        for headerLine: String in responsesResponse.additionalHeaderLines {
            let headerParts: Array<Substring> = headerLine.split(separator: ":", maxSplits: 1);
            if headerParts.count == 2,
               headerParts[0].trimmingCharacters(in: .whitespaces).caseInsensitiveCompare(headerName) == .orderedSame {
                return headerParts[1].trimmingCharacters(in: .whitespaces);
            }
        }
        return nil;
    }

    /// The parsed named-frame stream of one Responses SSE response.
    struct ParsedResponsesSseStream {

        struct ParsedResponsesSseEvent {

            let eventType: String;
            let jsonPayload: [String: Any];
        }

        let events: Array<ParsedResponsesSseEvent>;

        static func parse(_ responseText: String) -> ParsedResponsesSseStream {
            var parsedEvents: Array<ParsedResponsesSseEvent> = Array();
            for sseFrame: Substring in responseText.split(separator: "\n\n", omittingEmptySubsequences: false) {
                var parsedEventType: String? = nil;
                var sseDataLines: Array<String> = Array();
                for sseFrameLine: Substring in sseFrame.split(separator: "\n", omittingEmptySubsequences: false) {
                    if sseFrameLine.hasPrefix("event: ") {
                        parsedEventType = String(sseFrameLine.dropFirst("event: ".count));
                    }
                    if sseFrameLine.hasPrefix("data: ") {
                        sseDataLines.append(String(sseFrameLine.dropFirst("data: ".count)));
                    }
                }
                guard let resolvedEventType: String = parsedEventType,
                      sseDataLines.isEmpty == false,
                      let payloadData: Any = try? JSONSerialization.jsonObject(
                        with: Data(sseDataLines.joined(separator: "\n").utf8)),
                      let payloadObject: [String: Any] = payloadData as? [String: Any] else {
                    continue;
                }
                parsedEvents.append(ParsedResponsesSseEvent(
                    eventType: resolvedEventType, jsonPayload: payloadObject));
            }
            return ParsedResponsesSseStream(events: parsedEvents);
        }

        func eventTypes() -> Array<String> {
            return self.events.map({ (sseEvent: ParsedResponsesSseEvent) -> String in
                return sseEvent.eventType;
            });
        }

        func visibleTextForOpencode() -> String {
            var collectedText: String = String();
            for sseEvent: ParsedResponsesSseEvent in self.events {
                if sseEvent.eventType == "response.output_text.delta",
                   let deltaText: String = sseEvent.jsonPayload["delta"] as? String {
                    collectedText += deltaText;
                }
            }
            return collectedText;
        }

        func reasoningSummaryText() -> String {
            var collectedText: String = String();
            for sseEvent: ParsedResponsesSseEvent in self.events {
                if sseEvent.eventType == "response.reasoning_summary_text.delta",
                   let deltaText: String = sseEvent.jsonPayload["delta"] as? String {
                    collectedText += deltaText;
                }
            }
            return collectedText;
        }

        func firstPayloadForEventType(_ expectedEventType: String) -> [String: Any]? {
            for sseEvent: ParsedResponsesSseEvent in self.events {
                if sseEvent.eventType == expectedEventType {
                    return sseEvent.jsonPayload;
                }
            }
            return nil;
        }

        func completedResponse() -> [String: Any]? {
            return self.firstPayloadForEventType("response.completed")?["response"] as? [String: Any];
        }
    }

    /// One discovered chat model for canonicalization journeys.
    static func discoveredChatModel(modelId: String) -> DiscoveryDiscoveredModel {
        return DiscoveryDiscoveredModel(
            modelId: modelId,
            providerModelId: nil,
            modelFamily: .qwen35,
            revision: "test-revision",
            modelDirectory: FilePath(string: "/models/\(modelId)"),
            capabilities: .chat(DiscoveryChatModelCapabilities(
                contextWindowTokens: 2_048,
                maximumInputTokens: 1_024,
                maximumOutputTokens: 128,
                supportsVision: false,
                supportsReasoning: true,
                supportsToolCalls: true)),
            license: nil,
            modelSizeBytes: 0);
    }
}
