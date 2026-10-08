import Foundation

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import Supervisor;
import JourneyCategories;

@testable import Supervisor;

/// Hermetic daemon IPC journeys for the model-lifecycle verbs and the
/// embeddings verb, porting apps/supervisor/tests/hermetic/
/// daemon_ipc_embeddings.rs, daemon_ipc_models.rs, and daemon_ipc_lifecycle.rs.
@Suite(.serialized, .tags(.hermeticJourney))
final class DaemonIpcModelsTests {

    // MARK: - Embeddings (daemon_ipc_embeddings.rs)

    @Test
    func should_route_embeddings_requests_to_the_executor_and_return_the_vectors() throws {
        let stateDirectory: String = DaemonIpcModelsTests.freshInstanceStateDirectory("embed-routed");
        defer { try? FileManager.default.removeItem(atPath: stateDirectory) }
        let recordingExecutor: RecordingEmbeddingsExecutor = RecordingEmbeddingsExecutor(
            healthSnapshot: DaemonIpcModelsTests.readySnapshot(modelId: "test/local-embedder"),
            embeddingsOutcome: .success(EmbeddingsOutput(
                embeddings: [[0.25, -0.5]],
                inputTokenCounts: [3]
            ))
        );
        let service: DaemonIpcService = try DaemonIpcModelsTests.startService(
            stateDirectory: stateDirectory,
            embeddingsExecutor: recordingExecutor
        );
        defer { service.shutdown() }

        let daemonClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: service.socketPath);
        try daemonClient.sendRequest(DaemonRequest.embedGenerate(
            model: nil,
            inputs: ["hello"],
            dimensions: nil));

        #expect(try daemonClient.nextResponse() == .embeddingsCompleted(
            model: "test/local-embedder",
            vectors: [[0.25, -0.5]],
            inputTokenCounts: [3]));

        let receivedEmbeddingsCommands: Array<EmbeddingsCommand> = recordingExecutor.receivedEmbeddingsCommands();
        #expect(receivedEmbeddingsCommands.count == 1);
        #expect(receivedEmbeddingsCommands[0].model == "test/local-embedder");
        #expect(receivedEmbeddingsCommands[0].inputs == ["hello"]);
        #expect(receivedEmbeddingsCommands[0].requestId == RequestId(rawRequestId: 1));
    }

    @Test
    func should_reject_embeddings_when_no_model_is_resident_and_none_was_requested() throws {
        let stateDirectory: String = DaemonIpcModelsTests.freshInstanceStateDirectory("embed-no-model");
        defer { try? FileManager.default.removeItem(atPath: stateDirectory) }
        let recordingExecutor: RecordingEmbeddingsExecutor = RecordingEmbeddingsExecutor(
            healthSnapshot: WorkerHealthSnapshot.readyWithoutModel(
                machineMlxMemoryCeilingBytes: 17179869184,
                effectiveMlxMemoryCeilingBytes: 8589934592,
                minimumMlxMemoryCeilingBytes: 1),
            embeddingsOutcome: .success(EmbeddingsOutput(embeddings: [[]], inputTokenCounts: []))
        );
        let service: DaemonIpcService = try DaemonIpcModelsTests.startService(
            stateDirectory: stateDirectory,
            embeddingsExecutor: recordingExecutor
        );
        defer { service.shutdown() }

        let daemonClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: service.socketPath);
        try daemonClient.sendRequest(DaemonRequest.embedGenerate(
            model: nil,
            inputs: ["hello"],
            dimensions: nil));

        let responseFrame: DaemonResponse? = try daemonClient.nextResponse();
        guard case let .generationRejected(reason) = responseFrame else {
            Issue.record("expected a generation rejection, got \(String(describing: responseFrame))");
            return;
        }
        #expect(reason.contains("no model"), "the rejection should tell the caller no model is loaded: \(reason)");
    }

    // MARK: - Models verbs (daemon_ipc_models.rs)

    @Test
    func should_list_discovered_models_over_daemon_ipc() throws {
        let stateDirectory: String = DaemonIpcModelsTests.freshInstanceStateDirectory("models-list");
        defer { try? FileManager.default.removeItem(atPath: stateDirectory) }
        let discoveredModel: DiscoveryDiscoveredModel = DaemonIpcModelsTests.chatDiscoveredModel(
            modelId: "test/local-chatter"
        );
        let recordingExecutor: RecordingEmbeddingsExecutor = RecordingEmbeddingsExecutor(
            healthSnapshot: DaemonIpcModelsTests.readySnapshot(modelId: "test/local-chatter"),
            embeddingsOutcome: .success(EmbeddingsOutput(embeddings: [[]], inputTokenCounts: []))
        );
        let service: DaemonIpcService = try DaemonIpcModelsTests.startService(
            stateDirectory: stateDirectory,
            embeddingsExecutor: recordingExecutor,
            discoveredModels: [discoveredModel]
        );
        defer { service.shutdown() }

        let daemonClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: service.socketPath);
        try daemonClient.sendRequest(DaemonRequest.modelsList);
        #expect(try daemonClient.nextResponse() == .modelsList(models: [
            DaemonListedModel(
                modelId: "test/local-chatter",
                family: "qwen3_5",
                contextWindow: 2_048,
                supportsEmbeddings: false,
                isResident: true,
                sizeBytes: 1_000_000_000),
        ]));
    }

    @Test
    func should_project_the_release_catalog_over_daemon_ipc() throws {
        let stateDirectory: String = DaemonIpcModelsTests.freshInstanceStateDirectory("catalog");
        defer { try? FileManager.default.removeItem(atPath: stateDirectory) }
        let recordingExecutor: RecordingEmbeddingsExecutor = RecordingEmbeddingsExecutor(
            healthSnapshot: DaemonIpcModelsTests.readySnapshot(modelId: "test/local-chatter"),
            embeddingsOutcome: .success(EmbeddingsOutput(embeddings: [[]], inputTokenCounts: []))
        );
        let service: DaemonIpcService = try DaemonIpcModelsTests.startService(
            stateDirectory: stateDirectory,
            embeddingsExecutor: recordingExecutor
        );
        defer { service.shutdown() }

        let daemonClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: service.socketPath);
        try daemonClient.sendRequest(DaemonRequest.catalog);
        let responseFrame: DaemonResponse? = try daemonClient.nextResponse();
        guard case let .catalog(entries) = responseFrame else {
            Issue.record("the daemon should answer with a catalog response");
            return;
        }
        // Structural assertions only: the bundled catalog ships with the
        // release, so couple to its invariants, not to specific entries.
        #expect(!entries.isEmpty, "the bundled catalog should project its entries");
        for catalogEntry: DaemonCatalogEntry in entries {
            #expect(!catalogEntry.huggingfaceId.isEmpty);
            #expect(!catalogEntry.readyOnThisMac, "no catalog entry can be ready without a coordinator or discovery");
            #expect(catalogEntry.requestableModelId == nil, "an entry that is not ready cannot expose a requestable id");
            #expect(catalogEntry.downloadState == nil, "an entry with no active download cannot expose a download state");
        }
    }

    @Test
    func should_report_no_active_download_without_a_coordinator() throws {
        let stateDirectory: String = DaemonIpcModelsTests.freshInstanceStateDirectory("download-status");
        defer { try? FileManager.default.removeItem(atPath: stateDirectory) }
        let recordingExecutor: RecordingEmbeddingsExecutor = RecordingEmbeddingsExecutor(
            healthSnapshot: DaemonIpcModelsTests.readySnapshot(modelId: "test/local-chatter"),
            embeddingsOutcome: .success(EmbeddingsOutput(embeddings: [[]], inputTokenCounts: []))
        );
        let service: DaemonIpcService = try DaemonIpcModelsTests.startService(
            stateDirectory: stateDirectory,
            embeddingsExecutor: recordingExecutor
        );
        defer { service.shutdown() }

        let daemonClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: service.socketPath);
        try daemonClient.sendRequest(DaemonRequest.downloadStatus);
        #expect(try daemonClient.nextResponse() == .downloadStatus(job: nil));
    }

    @Test
    func should_start_a_download_rejection_without_a_coordinator() throws {
        let stateDirectory: String = DaemonIpcModelsTests.freshInstanceStateDirectory("download-start");
        defer { try? FileManager.default.removeItem(atPath: stateDirectory) }
        let recordingExecutor: RecordingEmbeddingsExecutor = RecordingEmbeddingsExecutor(
            healthSnapshot: DaemonIpcModelsTests.readySnapshot(modelId: "test/local-chatter"),
            embeddingsOutcome: .success(EmbeddingsOutput(embeddings: [[]], inputTokenCounts: []))
        );
        let service: DaemonIpcService = try DaemonIpcModelsTests.startService(
            stateDirectory: stateDirectory,
            embeddingsExecutor: recordingExecutor
        );
        defer { service.shutdown() }

        let daemonClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: service.socketPath);
        try daemonClient.sendRequest(DaemonRequest.downloadStart(modelId: "test/any-model"));
        let responseFrame: DaemonResponse? = try daemonClient.nextResponse();
        guard case let .requestRejected(reason) = responseFrame else {
            Issue.record("a download start without a coordinator must be rejected");
            return;
        }
        #expect(reason.contains("no Library download coordinator"));
    }

    @Test
    func should_set_and_report_the_default_model_over_daemon_ipc() throws {
        let stateDirectory: String = DaemonIpcModelsTests.freshInstanceStateDirectory("default-model");
        defer { try? FileManager.default.removeItem(atPath: stateDirectory) }
        let recordingExecutor: RecordingEmbeddingsExecutor = RecordingEmbeddingsExecutor(
            healthSnapshot: DaemonIpcModelsTests.readySnapshot(modelId: "test/local-chatter"),
            embeddingsOutcome: .success(EmbeddingsOutput(embeddings: [[]], inputTokenCounts: []))
        );
        let service: DaemonIpcService = try DaemonIpcModelsTests.startService(
            stateDirectory: stateDirectory,
            embeddingsExecutor: recordingExecutor
        );
        defer { service.shutdown() }

        // Derive the expected normalized id from the bundled catalog itself so
        // the test survives catalog repackaging.
        let bundledCatalog: DownloadCatalog = try #require(try? DownloadCatalog.loadBundled());
        let firstCatalogEntry: DownloadCatalogEntry = try #require(bundledCatalog.entries.first);
        let requestableModelId: String = ModelIdentity.leafModelId(
            modelId: firstCatalogEntry.huggingfaceId
        );

        let setClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: service.socketPath);
        try setClient.sendRequest(DaemonRequest.defaultModelSet(modelId: requestableModelId));
        #expect(try setClient.nextResponse() == .defaultModelSet(defaultModelId: requestableModelId));

        // The persisted default is visible to a fresh status probe, proving
        // the write round-trips through the instance config file.
        let statusClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: service.socketPath);
        try statusClient.sendRequest(DaemonRequest.status);
        #expect(try statusClient.nextResponse() == .status(
            workerStatus: .ready,
            readyModelId: "test/local-chatter",
            defaultModelId: requestableModelId));
    }

    @Test
    func should_reject_an_unknown_default_model_over_daemon_ipc() throws {
        let stateDirectory: String = DaemonIpcModelsTests.freshInstanceStateDirectory("default-model-reject");
        defer { try? FileManager.default.removeItem(atPath: stateDirectory) }
        let recordingExecutor: RecordingEmbeddingsExecutor = RecordingEmbeddingsExecutor(
            healthSnapshot: DaemonIpcModelsTests.readySnapshot(modelId: "test/local-chatter"),
            embeddingsOutcome: .success(EmbeddingsOutput(embeddings: [[]], inputTokenCounts: []))
        );
        let service: DaemonIpcService = try DaemonIpcModelsTests.startService(
            stateDirectory: stateDirectory,
            embeddingsExecutor: recordingExecutor
        );
        defer { service.shutdown() }

        let daemonClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: service.socketPath);
        try daemonClient.sendRequest(DaemonRequest.defaultModelSet(modelId: "nope/unknown-model"));
        let responseFrame: DaemonResponse? = try daemonClient.nextResponse();
        guard case let .requestRejected(reason) = responseFrame else {
            Issue.record("an unknown default model must be rejected");
            return;
        }
        #expect(reason.contains("nope/unknown-model is unknown"));
    }

    // MARK: - Service lifecycle (daemon_ipc_lifecycle.rs)

    @Test
    func should_answer_handshake_with_application_identity_on_the_instance_socket() throws {
        let stateDirectory: String = DaemonIpcModelsTests.freshInstanceStateDirectory("handshake-identity");
        defer { try? FileManager.default.removeItem(atPath: stateDirectory) }
        let recordingExecutor: RecordingEmbeddingsExecutor = RecordingEmbeddingsExecutor(
            healthSnapshot: WorkerHealthSnapshot.readyWithoutModel(
                machineMlxMemoryCeilingBytes: 17179869184,
                effectiveMlxMemoryCeilingBytes: 8589934592,
                minimumMlxMemoryCeilingBytes: 1),
            embeddingsOutcome: .success(EmbeddingsOutput(embeddings: [[]], inputTokenCounts: []))
        );
        let service: DaemonIpcService = try DaemonIpcModelsTests.startService(
            stateDirectory: stateDirectory,
            embeddingsExecutor: recordingExecutor
        );
        defer { service.shutdown() }

        let daemonClient: DaemonIpcClient = try DaemonIpcClient.connect(socketPath: service.socketPath);
        try daemonClient.sendRequest(DaemonRequest.handshake);
        #expect(try daemonClient.nextResponse() == .handshakeAccepted(
            protocolVersion: DaemonProtocol.protocolVersion,
            applicationName: DaemonProtocol.applicationName));
    }

    @Test
    func should_remove_the_daemon_socket_file_on_shutdown() throws {
        let stateDirectory: String = DaemonIpcModelsTests.freshInstanceStateDirectory("shutdown");
        defer { try? FileManager.default.removeItem(atPath: stateDirectory) }
        let recordingExecutor: RecordingEmbeddingsExecutor = RecordingEmbeddingsExecutor(
            healthSnapshot: WorkerHealthSnapshot.readyWithoutModel(
                machineMlxMemoryCeilingBytes: 17179869184,
                effectiveMlxMemoryCeilingBytes: 8589934592,
                minimumMlxMemoryCeilingBytes: 1),
            embeddingsOutcome: .success(EmbeddingsOutput(embeddings: [[]], inputTokenCounts: []))
        );
        let service: DaemonIpcService = try DaemonIpcModelsTests.startService(
            stateDirectory: stateDirectory,
            embeddingsExecutor: recordingExecutor
        );
        let socketPath: String = service.socketPath;
        #expect(FileManager.default.fileExists(atPath: socketPath), "the socket file should exist while serving");

        service.shutdown();
        #expect(!FileManager.default.fileExists(atPath: socketPath), "the socket file should be removed when the service stops");
    }

    @Test
    func should_refuse_a_second_daemon_listener_on_the_same_socket() throws {
        let stateDirectory: String = DaemonIpcModelsTests.freshInstanceStateDirectory("second-listener");
        defer { try? FileManager.default.removeItem(atPath: stateDirectory) }
        let recordingExecutor: RecordingEmbeddingsExecutor = RecordingEmbeddingsExecutor(
            healthSnapshot: WorkerHealthSnapshot.readyWithoutModel(
                machineMlxMemoryCeilingBytes: 17179869184,
                effectiveMlxMemoryCeilingBytes: 8589934592,
                minimumMlxMemoryCeilingBytes: 1),
            embeddingsOutcome: .success(EmbeddingsOutput(embeddings: [[]], inputTokenCounts: []))
        );
        let instancePaths: AstronomicalInstancePaths = DaemonIpcModelsTests.instancePaths(
            stateDirectory: stateDirectory
        );
        let runningService: DaemonIpcService = try DaemonIpcModelsTests.startService(
            instancePaths: instancePaths,
            embeddingsExecutor: recordingExecutor
        );
        defer { runningService.shutdown() }

        #expect(throws: (any Error).self) {
            _ = try DaemonIpcModelsTests.startService(
                instancePaths: instancePaths,
                embeddingsExecutor: recordingExecutor
            );
        };
    }

    // MARK: - Fixtures

    private static func freshInstanceStateDirectory(_ journeyName: String) -> String {
        // Short names: a unix socket path must fit in sockaddr_un.
        let stateDirectory: String = NSTemporaryDirectory() + "asup-dipc-\(journeyName)-\(UUID().uuidString.prefix(6))";
        try? FileManager.default.createDirectory(atPath: stateDirectory, withIntermediateDirectories: true);
        return stateDirectory;
    }

    private static func instancePaths(stateDirectory: String) -> AstronomicalInstancePaths {
        return AstronomicalInstancePaths.forStateDirectory(
            FilePath(string: stateDirectory),
            runtimeInstance: AstronomicalRuntimeInstance.development
        );
    }

    private static func startService(
        stateDirectory: String,
        embeddingsExecutor: RecordingEmbeddingsExecutor
    ) throws -> DaemonIpcService {
        return try DaemonIpcModelsTests.startService(
            instancePaths: DaemonIpcModelsTests.instancePaths(stateDirectory: stateDirectory),
            embeddingsExecutor: embeddingsExecutor
        );
    }

    private static func startService(
        instancePaths: AstronomicalInstancePaths,
        embeddingsExecutor: RecordingEmbeddingsExecutor
    ) throws -> DaemonIpcService {
        return try DaemonIpcModelsTests.startService(
            instancePaths: instancePaths,
            embeddingsExecutor: embeddingsExecutor,
            discoveredModels: []
        );
    }

    private static func startService(
        stateDirectory: String,
        embeddingsExecutor: RecordingEmbeddingsExecutor,
        discoveredModels: Array<DiscoveryDiscoveredModel>
    ) throws -> DaemonIpcService {
        return try DaemonIpcModelsTests.startService(
            instancePaths: DaemonIpcModelsTests.instancePaths(stateDirectory: stateDirectory),
            embeddingsExecutor: embeddingsExecutor,
            discoveredModels: discoveredModels
        );
    }

    private static func startService(
        instancePaths: AstronomicalInstancePaths,
        embeddingsExecutor: RecordingEmbeddingsExecutor,
        discoveredModels: Array<DiscoveryDiscoveredModel>
    ) throws -> DaemonIpcService {
        let downloadCatalog: DownloadCatalog = try DownloadCatalog.loadBundled();
        let capturedResolvedConfig: ResolvedRuntimeConfig = {
            var resolvedConfig: ResolvedRuntimeConfig = DaemonIpcModelsTests.resolvedConfig();
            resolvedConfig.discoveredModels = discoveredModels;
            return resolvedConfig;
        }();
        return try DaemonIpcService.start(
            instancePaths: instancePaths,
            healthProvider: {
                let healthSnapshot: WorkerHealthSnapshot = embeddingsExecutor.workerHealthSnapshot();
                return DaemonStatusReport(
                    workerStatus: healthSnapshot.status.daemonWorkerStatus(),
                    readyModelId: healthSnapshot.readyModelId);
            },
            modelsContext: DaemonIpcModelsContext(
                embeddingsExecutor: embeddingsExecutor,
                liveResolvedRuntimeConfigProvider: { return capturedResolvedConfig; },
                downloadCatalog: downloadCatalog,
                libraryDownloadCoordinator: nil,
                instancePaths: instancePaths
            )
        );
    }

    private static func readySnapshot(modelId: String) -> WorkerHealthSnapshot {
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

    private static func chatDiscoveredModel(modelId: String) -> DiscoveryDiscoveredModel {
        return DiscoveryDiscoveredModel(
            modelId: modelId,
            providerModelId: "test-org/\(modelId)",
            modelFamily: .qwen35,
            revision: "revision-1",
            modelDirectory: FilePath(string: "/fictional/models/\(modelId)"),
            capabilities: .chat(DiscoveryChatModelCapabilities(
                contextWindowTokens: 2_048,
                maximumInputTokens: 2_048,
                maximumOutputTokens: 1_024,
                supportsVision: false,
                supportsReasoning: false,
                supportsToolCalls: false)),
            license: nil,
            modelSizeBytes: 1_000_000_000);
    }

    private static func resolvedConfig() -> ResolvedRuntimeConfig {
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
}

/// Captures the embeddings commands the daemon dispatched, like the Rust
/// StubGenerationExecutor's embeddings log.
final class RecordingEmbeddingsExecutor: EmbeddingsExecuting, @unchecked Sendable {

    private let stateLock: NSLock = NSLock();
    private let recordedHealthSnapshot: WorkerHealthSnapshot;
    private let recordedEmbeddingsOutcome: Result<EmbeddingsOutput, EmbeddingsExecutionError>;
    private var recordedEmbeddingsCommands: Array<EmbeddingsCommand> = [];

    init(
        healthSnapshot: WorkerHealthSnapshot,
        embeddingsOutcome: Result<EmbeddingsOutput, EmbeddingsExecutionError>
    ) {
        self.recordedHealthSnapshot = healthSnapshot;
        self.recordedEmbeddingsOutcome = embeddingsOutcome;
    }

    func startEmbeddingsGeneration(
        _ embeddingsCommand: EmbeddingsCommand
    ) throws -> EmbeddingsOutput {
        self.stateLock.lock();
        self.recordedEmbeddingsCommands.append(embeddingsCommand);
        self.stateLock.unlock();
        switch (self.recordedEmbeddingsOutcome) {
        case let .success(embeddingsOutput):
            return embeddingsOutput;
        case let .failure(embeddingsError):
            throw embeddingsError;
        }
    }

    func workerHealthSnapshot() -> WorkerHealthSnapshot {
        return self.recordedHealthSnapshot;
    }

    func receivedEmbeddingsCommands() -> Array<EmbeddingsCommand> {
        self.stateLock.lock();
        let currentCommands: Array<EmbeddingsCommand> = self.recordedEmbeddingsCommands;
        self.stateLock.unlock();
        return currentCommands;
    }
}
