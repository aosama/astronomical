import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import JourneyCategories;
import RestContract;

@testable import Supervisor;

/// The two-variable open condition for the real chat journey, kept outside
/// the suite type so the `@Suite` attribute does not reference it circularly.
enum RestChatRealModelJourneyGate {

    static func isOpen() -> Bool {
        return RealModelJourneyGate.qwen35ArtifactDirectory() != nil
            && RestChatRealModelJourneyGate.workerExecutablePath() != nil;
    }

    /// The built inference-worker executable this machine runs the journey
    /// against; the repository never commits a developer path.
    static func workerExecutablePath() -> String? {
        return RealModelJourneyGate.installedArtifactDirectory(
            environmentVariableName: "ASTRONOMICAL_INFERENCE_WORKER_EXECUTABLE");
    }
}

/**
 * The gated real-model chat journey: when this machine names an installed
 * Qwen3.5 artifact directory and the built inference-worker executable
 * through environment variables, the full serving stack — supervisor, spawned
 * worker process, real weights — answers the Romeo and Juliet fixture over
 * the same POST /v1/chat/completions route the daemon serves.
 *
 * Assertions are structural and config-derived (non-empty visible answer,
 * valid finish reason, positive token usage, well-formed SSE frames); no
 * golden-master constants couple the journey to one packaging variant.
 * The suite is serialized so real-model journeys never multiply wired GPU
 * memory, and the supervisor's own model-load bound stays inside the 120
 * second test ceiling.
 */
@Suite(.serialized, .tags(.realModelJourney), .enabled(if: RestChatRealModelJourneyGate.isOpen()))
final class RestChatRealModelJourneyTests {

    private static let modelLoadTimeoutSeconds: TimeInterval = 110;
    private static let romeoAndJulietPrompt: String =
        "You are a concise literature assistant. In one sentence, name the play "
        + "these lines come from: \"O Romeo, Romeo, wherefore art thou Romeo?\"";

    // MARK: Journeys

    @Test
    func should_answer_a_non_streaming_romeo_and_juliet_chat_over_the_real_stack() throws {
        let journey: RestChatRealModelJourney = try RestChatRealModelJourneyTests.launch();

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: journey.routeTable,
            requestBody: "{\"model\":\"\(journey.modelId)\","
                + "\"messages\":[{\"role\":\"user\",\"content\":"
                + "\"\(RestChatRealModelJourneyTests.romeoAndJulietPrompt)\"}],\"stream\":false}");
        _ = try? journey.supervisor.shutdown();

        #expect(chatResponse.statusCode == 200);
        let envelope: [String: Any] = try RestChatJourneySupport.decodeObjectEnvelope(chatResponse);
        #expect(envelope["object"] as? String == "chat.completion");
        #expect(envelope["model"] as? String == journey.modelId);
        let choice: [String: Any] = try RestChatJourneySupport.requireChoice(envelope);
        let assistantMessage: [String: Any] = try RestChatJourneySupport.requireMessage(choice);
        let visibleContent: String = try #require(assistantMessage["content"] as? String);
        #expect(visibleContent.isEmpty == false,
            "the real model must produce a visible answer for the fixture prompt");
        let finishReason: String = try #require(choice["finish_reason"] as? String);
        #expect(finishReason == "stop" || finishReason == "length",
            "the finish reason must be a real completion boundary, got \(finishReason)");
        let usage: [String: Any] = try RestChatJourneySupport.requireUsage(envelope);
        #expect((usage["prompt_tokens"] as? UInt64 ?? 0) > 0);
        #expect((usage["completion_tokens"] as? UInt64 ?? 0) > 0);
    }

    @Test
    func should_stream_a_romeo_and_juliet_chat_with_a_terminated_frame_sequence() throws {
        let journey: RestChatRealModelJourney = try RestChatRealModelJourneyTests.launch();

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: journey.routeTable,
            requestBody: "{\"model\":\"\(journey.modelId)\","
                + "\"messages\":[{\"role\":\"user\",\"content\":"
                + "\"\(RestChatRealModelJourneyTests.romeoAndJulietPrompt)\"}],\"stream\":true,"
                + "\"stream_options\":{\"include_usage\":true}}");
        _ = try? journey.supervisor.shutdown();

        #expect(chatResponse.statusCode == 200);
        #expect(chatResponse.contentType.hasPrefix("text/event-stream"));
        let parsedStream: RestChatJourneySupport.ParsedChatSseStream =
            RestChatJourneySupport.ParsedChatSseStream.parse(
                RestChatJourneySupport.responseText(chatResponse));
        #expect(parsedStream.sawDone, "the real stream must terminate with data: [DONE]");
        #expect(parsedStream.deltaText(forKey: "content").isEmpty == false,
            "the real stream must carry visible text deltas");
        let finishReason: String = try #require(parsedStream.finishReason());
        #expect(finishReason == "stop" || finishReason == "length",
            "the finish reason must be a real completion boundary, got \(finishReason)");
    }

    // MARK: Gate and stack assembly

    /// One launched real serving stack bound to one journey.
    struct RestChatRealModelJourney {

        let supervisor: WorkerSupervisor;
        let routeTable: RestRouteTable;
        let modelId: String;
    }

    static func launch() throws -> RestChatRealModelJourney {
        let artifactDirectory: String = try #require(
            RealModelJourneyGate.qwen35ArtifactDirectory());
        let workerExecutablePath: String = try #require(RestChatRealModelJourneyGate.workerExecutablePath());
        let discoveredModels: Array<DiscoveryDiscoveredModel> = try DiscoveryModels
            .discoverModels(modelDirectories: [FilePath(string: artifactDirectory)])
            .flatMap { (directoryScan: DiscoveryModelDiscoveryDirectoryScan) -> Array<DiscoveryDiscoveredModel> in
                return directoryScan.discoveredModels;
            };
        let chatDiscoveredModels: Array<DiscoveryDiscoveredModel> = discoveredModels.filter(
            { (discoveredModel: DiscoveryDiscoveredModel) -> Bool in
                if case .chat = discoveredModel.capabilities {
                    return true;
                }
                return false;
            });
        let journeyModel: DiscoveryDiscoveredModel = try #require(
            chatDiscoveredModels.first,
            "the gated artifact directory must discover at least one chat model");
        let temporaryStateDirectory: String = NSTemporaryDirectory()
            + "restchat-real-\(UUID().uuidString.prefix(8))";
        try FileManager.default.createDirectory(
            atPath: temporaryStateDirectory, withIntermediateDirectories: true);
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            FilePath(string: temporaryStateDirectory),
            runtimeInstance: AstronomicalRuntimeInstance.development);
        try FileManager.default.createDirectory(
            atPath: instancePaths.stateDirectory.string,
            withIntermediateDirectories: true);
        let userConfig: AstronomicalConfig = try AstronomicalConfig.loadFromInstancePaths(
            instancePaths);
        var modelPolicyCatalog: Dictionary<String, RuntimeModelPolicy> = Dictionary();
        for discoveredModel: DiscoveryDiscoveredModel in chatDiscoveredModels {
            modelPolicyCatalog[discoveredModel.modelId] = try ResolvedModelPolicyCatalog.resolve(
                userConfig: userConfig,
                discoveredModels: chatDiscoveredModels,
                artifactContextWindows: Dictionary<String, UInt32>())[discoveredModel.modelId];
        }
        let resolvedRuntimeConfig: ResolvedRuntimeConfig = ResolvedRuntimeConfig(
            configurationGeneration: "realmodeljourney0000000000000000000000",
            workerExecutablePath: FilePath(string: workerExecutablePath),
            discoveredModels: chatDiscoveredModels,
            modelDiscoveryDiagnostics: Array<DiscoveryModelDiscoveryDiagnostic>(),
            configuredModelDirectories: [FilePath(string: artifactDirectory)],
            modelPolicyCatalog: modelPolicyCatalog,
            unmatchedModelConfigIds: Array<String>(),
            maximumMlxMemoryBytes: nil,
            performanceAttributionEnabled: false,
            completionAttributionEnabled: false,
            persistentPromptCacheEnabled: false,
            configuredPersistentPromptCacheEnabled: nil,
            configuredPromptCacheMaximumSizeBytes: 0,
            promptCacheConfig: PromptCacheConfig(
                rootDirectory: FilePath(string: instancePaths.stateDirectory.string),
                maximumSizeBytes: 0),
            bindAddress: "127.0.0.1:0",
            bindEndpoint: SocketEndpoint.loopback(port: 0),
            loggingConfig: LoggingConfig(
                directory: FilePath(string: instancePaths.stateDirectory.string),
                level: LogLevel.warn,
                retainedFiles: 1));
        let supervisor: WorkerSupervisor = try WorkerSupervisor.launch(
            workerExecutablePath: workerExecutablePath,
            workerArguments: [],
            workerStartupConfiguration: resolvedRuntimeConfig.workerStartupConfiguration(),
            modelPolicyCatalog: modelPolicyCatalog,
            modelLoadTimeout: RestChatRealModelJourneyTests.modelLoadTimeoutSeconds);
        let routeTable: RestRouteTable = RestEndpointRoutes.servingRouteTable(
            resolvedRuntimeConfig: resolvedRuntimeConfig,
            workerHealthState: supervisor.ownedHealthState(),
            instancePaths: instancePaths,
            buildIdentity: ApplicationBuildIdentity(
                version: "0.0.0-real-model-journey",
                buildNumber: 0,
                commit: "journey",
                isDirty: false),
            chatContext: RestChatRouteContext(
                chatExecutor: supervisor,
                requestIdAllocator: ChatRequestIdAllocator(),
                resolvedRuntimeConfig: resolvedRuntimeConfig,
                instancePaths: instancePaths));
        return RestChatRealModelJourney(
            supervisor: supervisor,
            routeTable: routeTable,
            modelId: journeyModel.modelId);
    }
}
