import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import JourneyCategories;

@testable import Supervisor;

/**
 * Hermetic journeys for the daemon IPC chat verb: the trust-boundary gates,
 * default filling, request-id allocation, frame relay, and terminal-frame
 * stream closure. The executor is a stub, so no worker process runs.
 */
@Suite(.tags(.hermeticJourney)) final class DaemonIpcChatTests {

    @Test
    func should_relay_streamed_frames_and_close_at_the_terminal_frame() throws {
        let recordingExecutor: RecordingChatExecutor = RecordingChatExecutor(
            healthSnapshot: DaemonIpcChatTests.readySnapshot(modelId: "m1"),
            streamEvents: [
                .textFragment("Hel"),
                .reasoningFragment("pondering"),
                .textFragment("lo"),
                .completed(
                    promptTokenCount: 12,
                    generatedTokenCount: 3,
                    reasoningTokenCount: 1,
                    cachedTokenCount: 0,
                    reason: .endOfSequence),
            ]);
        let service: DaemonIpcService = try DaemonIpcChatTests.startService(chatExecutor: recordingExecutor);
        defer { service.shutdown() }

        let chatClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: service.socketPath);
        try chatClient.sendRequest(DaemonRequest.chatGenerate(
            model: "m1",
            messages: [.user(content: "say hi", images: [])],
            settings: DaemonIpcChatTests.explicitSettings(),
            schemaJson: nil));

        #expect(try chatClient.nextResponse() == .chatGenerationText(text: "Hel"));
        #expect(try chatClient.nextResponse() == .chatGenerationReasoning(text: "pondering"));
        #expect(try chatClient.nextResponse() == .chatGenerationText(text: "lo"));
        #expect(try chatClient.nextResponse() == .chatGenerationCompleted(
            promptTokenCount: 12,
            generatedTokenCount: 3,
            reasoningTokenCount: 1,
            cachedTokenCount: 0,
            reason: .endOfSequence));
        // The terminal frame closes the stream.
        #expect(try chatClient.nextResponse() == nil);

        let receivedCommand: ChatGenerationCommand = try #require(recordingExecutor.receivedCommands.first);
        #expect(receivedCommand.requestId == RequestId(rawRequestId: 1));
        #expect(receivedCommand.model == "m1");
        #expect(receivedCommand.tools == []);
        #expect(receivedCommand.toolChoice == .auto);
        #expect(receivedCommand.qwenThinkingChannelSeed == nil);
        #expect(receivedCommand.structuredGeneration == nil);

        // Request identifiers stay monotonic across separate connections; the
        // listener serves exactly one request per connection.
        let secondClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: service.socketPath);
        try secondClient.sendRequest(DaemonRequest.chatGenerate(
            model: "m1",
            messages: [],
            settings: DaemonIpcChatTests.explicitSettings(),
            schemaJson: nil));
        _ = try secondClient.nextResponse();
        #expect(recordingExecutor.receivedCommands.last?.requestId == RequestId(rawRequestId: 2));
    }

    @Test
    func should_reject_chat_generation_when_the_worker_is_unavailable() throws {
        let recordingExecutor: RecordingChatExecutor = RecordingChatExecutor(
            healthSnapshot: WorkerHealthSnapshot.unavailable(.unavailable),
            streamEvents: []);
        let service: DaemonIpcService = try DaemonIpcChatTests.startService(chatExecutor: recordingExecutor);
        defer { service.shutdown() }

        let chatClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: service.socketPath);
        try chatClient.sendRequest(DaemonRequest.chatGenerate(
            model: "m1",
            messages: [.user(content: "say hi", images: [])],
            settings: DaemonIpcChatTests.explicitSettings(),
            schemaJson: nil));

        #expect(try chatClient.nextResponse() == .generationRejected(
            reason: "the daemon worker is not ready to serve chat generation"));
        #expect(try chatClient.nextResponse() == nil);
        #expect(recordingExecutor.receivedCommands.isEmpty);
    }

    @Test
    func should_reject_an_unknown_model_with_near_match_suggestions() throws {
        let recordingExecutor: RecordingChatExecutor = RecordingChatExecutor(
            healthSnapshot: DaemonIpcChatTests.readySnapshot(modelId: nil),
            streamEvents: []);
        let service: DaemonIpcService = try DaemonIpcChatTests.startService(
            chatExecutor: recordingExecutor,
            catalogModelIds: ["m3c7", "m9"]);
        defer { service.shutdown() }

        let chatClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: service.socketPath);
        try chatClient.sendRequest(DaemonRequest.chatGenerate(
            model: "m3c",
            messages: [.user(content: "say hi", images: [])],
            settings: DaemonIpcChatTests.explicitSettings(),
            schemaJson: nil));

        let rejectionFrame: DaemonResponse? = try chatClient.nextResponse();
        let rejectionResponse: DaemonResponse = try #require(rejectionFrame);
        guard case let .generationRejected(rejectionReason) = rejectionResponse else {
            Issue.record("expected a generation rejection, got \(rejectionResponse)");
            return;
        }
        #expect(rejectionReason.contains("the model m3c is unknown"));
        #expect(rejectionReason.contains("did you mean"));
        #expect(recordingExecutor.receivedCommands.isEmpty);
    }

    @Test
    func should_map_a_model_load_start_failure_to_a_rejection() throws {
        let failingExecutor: FailingChatExecutor = FailingChatExecutor(
            healthSnapshot: DaemonIpcChatTests.readySnapshot(modelId: nil),
            startFailure: .modelLoadFailed(modelLoadFailureReason: "artifact unsupported"));
        let service: DaemonIpcService = try DaemonIpcChatTests.startService(
            chatExecutor: failingExecutor,
            catalogModelIds: ["m1"]);
        defer { service.shutdown() }

        let chatClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: service.socketPath);
        try chatClient.sendRequest(DaemonRequest.chatGenerate(
            model: "m1",
            messages: [.user(content: "say hi", images: [])],
            settings: DaemonIpcChatTests.explicitSettings(),
            schemaJson: nil));

        #expect(try chatClient.nextResponse() == .generationRejected(
            reason: "the model could not be loaded: artifact unsupported"));
        #expect(try chatClient.nextResponse() == nil);
    }

    @Test
    func should_fill_omitted_settings_from_policy_then_capabilities() throws {
        // The catalog policy carries 512 output tokens and a 700 temperature;
        // the ready worker advertises 4096 output tokens. The first request
        // omits everything, so the policy fills; the second requests a policy
        // without defaults, so the ready-model capabilities fill the gap.
        let recordingExecutor: RecordingChatExecutor = RecordingChatExecutor(
            healthSnapshot: DaemonIpcChatTests.readySnapshot(modelId: "capability-only"),
            streamEvents: [.completed(
                promptTokenCount: 1,
                generatedTokenCount: 0,
                reasoningTokenCount: 0,
                cachedTokenCount: 0,
                reason: .endOfSequence)]);
        let service: DaemonIpcService = try DaemonIpcChatTests.startService(
            chatExecutor: recordingExecutor,
            catalogModelIds: ["m1", "capability-only"]);
        defer { service.shutdown() }

        let policyClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: service.socketPath);
        try policyClient.sendRequest(DaemonRequest.chatGenerate(
            model: "m1",
            messages: [],
            settings: DaemonIpcChatTests.emptySettings(),
            schemaJson: nil));
        _ = try policyClient.nextResponse();
        let policyFilledCommand: ChatGenerationCommand = try #require(recordingExecutor.receivedCommands.first);
        #expect(policyFilledCommand.settings.maxOutputTokens == 512);
        #expect(policyFilledCommand.settings.temperatureThousandths == 700);

        let capabilityClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: service.socketPath);
        try capabilityClient.sendRequest(DaemonRequest.chatGenerate(
            model: "capability-only",
            messages: [],
            settings: DaemonIpcChatTests.emptySettings(),
            schemaJson: nil));
        _ = try capabilityClient.nextResponse();
        let capabilityFilledCommand: ChatGenerationCommand = try #require(recordingExecutor.receivedCommands.last);
        #expect(capabilityFilledCommand.settings.maxOutputTokens == 4096);
    }

    @Test
    func should_validate_the_schema_and_enforce_an_output_instruction() throws {
        let recordingExecutor: RecordingChatExecutor = RecordingChatExecutor(
            healthSnapshot: DaemonIpcChatTests.readySnapshot(modelId: "m1"),
            streamEvents: [.completed(
                promptTokenCount: 1,
                generatedTokenCount: 0,
                reasoningTokenCount: 0,
                cachedTokenCount: 0,
                reason: .endOfSequence)]);
        let service: DaemonIpcService = try DaemonIpcChatTests.startService(chatExecutor: recordingExecutor);
        defer { service.shutdown() }

        let schemaClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: service.socketPath);
        try schemaClient.sendRequest(DaemonRequest.chatGenerate(
            model: "m1",
            messages: [.user(content: "structured please", images: [])],
            settings: DaemonIpcChatTests.explicitSettings(),
            schemaJson: "{\"type\":\"object\"}"));
        _ = try schemaClient.nextResponse();

        let schemaCommand: ChatGenerationCommand = try #require(recordingExecutor.receivedCommands.first);
        #expect(schemaCommand.structuredGeneration == .jsonSchema(schemaJson: "{\"type\":\"object\"}"));
        let instructionMessage: ChatMessage? = schemaCommand.messages.first;
        guard case let .system(instructionText) = instructionMessage else {
            Issue.record("expected a leading schema instruction message, got \(String(describing: instructionMessage))");
            return;
        }
        #expect(instructionText.contains("matching this schema"));

        let malformedClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: service.socketPath);
        try malformedClient.sendRequest(DaemonRequest.chatGenerate(
            model: "m1",
            messages: [],
            settings: DaemonIpcChatTests.explicitSettings(),
            schemaJson: "{not json"));
        let rejectionFrame: DaemonResponse? = try malformedClient.nextResponse();
        let rejectionResponse: DaemonResponse = try #require(rejectionFrame);
        guard case .generationRejected = rejectionResponse else {
            Issue.record("expected a schema rejection, got \(rejectionResponse)");
            return;
        }
        #expect(recordingExecutor.receivedCommands.count == 1);
    }

    @Test
    func should_keep_handshake_and_status_single_frame_through_the_streaming_dispatch() throws {
        let recordingExecutor: RecordingChatExecutor = RecordingChatExecutor(
            healthSnapshot: DaemonIpcChatTests.readySnapshot(modelId: "m1"),
            streamEvents: []);
        let service: DaemonIpcService = try DaemonIpcChatTests.startService(chatExecutor: recordingExecutor);
        defer { service.shutdown() }

        let handshakeClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: service.socketPath);
        try handshakeClient.sendRequest(DaemonRequest.handshake);
        #expect(try handshakeClient.nextResponse() == .handshakeAccepted(
            protocolVersion: DaemonProtocol.protocolVersion,
            applicationName: DaemonProtocol.applicationName));
        #expect(try handshakeClient.nextResponse() == nil);

        let statusClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: service.socketPath);
        try statusClient.sendRequest(DaemonRequest.status);
        #expect(try statusClient.nextResponse() == .status(
            workerStatus: .ready,
            readyModelId: "m1",
            defaultModelId: DefaultModel.builtinDefaultModelId));
        #expect(try statusClient.nextResponse() == nil);
    }

    // MARK: - Fixtures

    private static func startService(
        chatExecutor: ChatGenerationExecuting,
        catalogModelIds: Array<String> = ["m1"]
    ) throws -> DaemonIpcService {
        var resolvedConfig: ResolvedRuntimeConfig = try DaemonIpcChatTests.resolvedConfig();
        for catalogModelId: String in catalogModelIds {
            resolvedConfig.modelPolicyCatalog[catalogModelId] = catalogModelId == "capability-only"
                ? DaemonIpcChatTests.capabilityOnlyPolicy(modelId: catalogModelId)
                : DaemonIpcChatTests.defaultsPolicy(modelId: catalogModelId);
        }
        let temporaryStateDirectory: String = NSTemporaryDirectory() + "asup-chat-\(UUID().uuidString.prefix(8))";
        try FileManager.default.createDirectory(atPath: temporaryStateDirectory, withIntermediateDirectories: true);
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forStateDirectory(
            FilePath(string: temporaryStateDirectory),
            runtimeInstance: AstronomicalRuntimeInstance.development);
        return try DaemonIpcService.start(
            instancePaths: instancePaths,
            healthProvider: {
                let healthSnapshot: WorkerHealthSnapshot = chatExecutor.workerHealthSnapshot();
                return DaemonStatusReport(
                    workerStatus: healthSnapshot.status.daemonWorkerStatus(),
                    readyModelId: healthSnapshot.readyModelId);
            },
            chatContext: DaemonIpcChatContext(
                chatExecutor: chatExecutor,
                resolvedRuntimeConfig: resolvedConfig,
                instancePaths: instancePaths));
    }

    private static func resolvedConfig() throws -> ResolvedRuntimeConfig {
        return ResolvedRuntimeConfig(
            configurationGeneration: "gen-1",
            workerExecutablePath: FilePath(string: "/opt/astronomical/bin/astronomical-inference-worker"),
            discoveredModels: [],
            modelDiscoveryDiagnostics: [],
            configuredModelDirectories: [],
            modelPolicyCatalog: [:],
            unmatchedModelConfigIds: [],
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

    private static func defaultsPolicy(modelId: String) -> RuntimeModelPolicy {
        return RuntimeModelPolicy(
            modelDirectory: FilePath(string: "/fictional/models/\(modelId)"),
            generationDefaults: RuntimeModelGenerationDefaults(
                maximumOutputTokens: 512,
                configuredMaximumOutputTokens: 512,
                temperatureThousandths: 700,
                topPThousandths: nil),
            configuredMaximumContextTokens: 4096,
            defaultMaximumContextTokens: 8192,
            configuredChunkingFields: ConfiguredChunkingFields.inactive(),
            workerModelConfiguration: DaemonIpcChatTests.autoregressiveConfiguration(modelId: modelId));
    }

    private static func capabilityOnlyPolicy(modelId: String) -> RuntimeModelPolicy {
        return RuntimeModelPolicy(
            modelDirectory: FilePath(string: "/fictional/models/\(modelId)"),
            generationDefaults: RuntimeModelGenerationDefaults(
                maximumOutputTokens: 0,
                configuredMaximumOutputTokens: 0,
                temperatureThousandths: nil,
                topPThousandths: nil),
            configuredMaximumContextTokens: 4096,
            defaultMaximumContextTokens: 8192,
            configuredChunkingFields: ConfiguredChunkingFields.inactive(),
            workerModelConfiguration: DaemonIpcChatTests.autoregressiveConfiguration(modelId: modelId));
    }

    private static func autoregressiveConfiguration(modelId: String) -> WorkerModelConfiguration {
        return WorkerModelConfiguration.autoregressive(WorkerAutoregressiveModelConfiguration(
            modelId: modelId,
            maximumContextTokens: 4096,
            maximumOutputTokens: 4096,
            chunking: WorkerChunkingConfiguration(
                fixedPromptProcessingChunkSizeTokens: 512,
                fixedSsdStreamingPromptProcessingChunkSizeTokens: 512,
                fullAttentionKeyValueGrowthTokens: 512,
                prefillGraphSubmissionLayerInterval: 1,
                experimentalSsdPagingPrefillGraphSubmissionLayerInterval: 1,
                experimentalSsdPagingGenerationGraphSubmissionLayerInterval: 1,
                promptCacheBlockTokens: nil,
                promptCacheCommonPrefixStrideBlocks: 1,
                experimentalDecodeStageAttributionEnabled: false,
                experimentalQuantizedKvCacheEnabled: false,
                experimentalFusedMoeDecodeEnabled: false)));
    }

    private static func readySnapshot(modelId: String?) -> WorkerHealthSnapshot {
        if let modelId = modelId {
            return WorkerHealthSnapshot.readyWithModel(
                modelId: modelId,
                capabilities: WorkerModelCapabilities.from(chatCapabilities: ChatModelCapabilities(
                    supportsReasoning: true,
                    supportsToolCalls: true,
                    hasVision: false,
                    maxInputTokens: 4095,
                    maxOutputTokens: 4096,
                    contextWindow: 4096)));
        }
        return WorkerHealthSnapshot.readyWithoutModel(
            machineMlxMemoryCeilingBytes: 17179869184,
            effectiveMlxMemoryCeilingBytes: 8589934592,
            minimumMlxMemoryCeilingBytes: 1);
    }

    private static func explicitSettings() -> ChatGenerationSettings {
        return ChatGenerationSettings(
            maxOutputTokens: 16,
            temperatureThousandths: 500,
            topPThousandths: nil,
            seed: nil,
            thinkingBudget: nil);
    }

    private static func emptySettings() -> ChatGenerationSettings {
        return ChatGenerationSettings(
            maxOutputTokens: 0,
            temperatureThousandths: nil,
            topPThousandths: nil,
            seed: nil,
            thinkingBudget: nil);
    }
}

/**
 * Records every command it receives and answers with the scripted stream.
 */
final class RecordingChatExecutor: ChatGenerationExecuting, @unchecked Sendable {
    private let recordsLock: NSLock;
    private var recordedCommands: Array<ChatGenerationCommand>;
    private let scriptedHealthSnapshot: WorkerHealthSnapshot;
    private let scriptedStreamEvents: Array<ChatGenerationStreamEvent>;

    init(healthSnapshot: WorkerHealthSnapshot, streamEvents: Array<ChatGenerationStreamEvent>) {
        self.recordsLock = NSLock();
        self.recordedCommands = [];
        self.scriptedHealthSnapshot = healthSnapshot;
        self.scriptedStreamEvents = streamEvents;
    }

    var receivedCommands: Array<ChatGenerationCommand> {
        self.recordsLock.lock();
        defer { self.recordsLock.unlock(); }
        return self.recordedCommands;
    }

    func startChatGeneration(_ generationCommand: ChatGenerationCommand) throws -> Array<ChatGenerationStreamEvent> {
        self.recordsLock.lock();
        self.recordedCommands.append(generationCommand);
        self.recordsLock.unlock();
        return self.scriptedStreamEvents;
    }

    func workerHealthSnapshot() -> WorkerHealthSnapshot {
        return self.scriptedHealthSnapshot;
    }
}

/**
 * Answers health queries but fails every start with the scripted error.
 */
final class FailingChatExecutor: ChatGenerationExecuting, @unchecked Sendable {
    private let scriptedHealthSnapshot: WorkerHealthSnapshot;
    private let scriptedStartFailure: GenerationStartError;

    init(healthSnapshot: WorkerHealthSnapshot, startFailure: GenerationStartError) {
        self.scriptedHealthSnapshot = healthSnapshot;
        self.scriptedStartFailure = startFailure;
    }

    func startChatGeneration(_ generationCommand: ChatGenerationCommand) throws -> Array<ChatGenerationStreamEvent> {
        throw self.scriptedStartFailure;
    }

    func workerHealthSnapshot() -> WorkerHealthSnapshot {
        return self.scriptedHealthSnapshot;
    }
}
