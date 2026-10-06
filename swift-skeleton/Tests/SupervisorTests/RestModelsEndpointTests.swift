import Foundation;
import XCTest;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;

@testable import Supervisor;

/**
 * Acceptance journey for the models advertisement and cache statistics REST
 * endpoints: an HTTP client on the loopback receives every discoverable
 * model in the OpenAI list shape with the Astronomical owner and generation
 * endpoint paths, can retrieve one model by identifier including a
 * provider-prefixed identifier, receives the shared failure envelope for an
 * unknown model, and sees the persistent prompt cache statistics with a
 * zeroed summary while no worker has reported cache activity yet. When
 * discovery is empty but a worker is ready, the ready model is advertised
 * from its acknowledged worker capabilities.
 */
final class RestModelsEndpointTests: XCTestCase {

    func testModelsJourneyAdvertisesDiscoveredChatModel() throws {
        let resolvedConfig: ResolvedRuntimeConfig = try self.makeResolvedConfig();
        let server: RestHttpServer = try self.startServingServer(resolvedConfig: resolvedConfig);

        let responseText: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");
        let (statusCode, envelope): (Int, [String: Any]) = try self.decodeObjectEnvelope(responseText);
        XCTAssertEqual(statusCode, 200);
        XCTAssertEqual(envelope["object"] as? String, "list");
        let models: Array<[String: Any]> = try self.requireModelsArray(envelope);
        XCTAssertEqual(models.count, 1);
        let advertisedModel: [String: Any] = models[0];
        XCTAssertEqual(advertisedModel["id"] as? String, "synthetic-chat-model");
        XCTAssertEqual(advertisedModel["owned_by"] as? String, "astronomical");
        XCTAssertGreaterThan(advertisedModel["created"] as? UInt64 ?? 0, 0);
        XCTAssertEqual(advertisedModel["context_window"] as? UInt32, 4_096);
        XCTAssertEqual(advertisedModel["max_output_tokens"] as? UInt32, 1_024);
        let supportedEndpoints: Array<String> = (advertisedModel["supported_endpoints"] as? [String]) ?? [];
        XCTAssertTrue(supportedEndpoints.contains("/v1/chat/completions"));
        XCTAssertTrue(supportedEndpoints.contains("/v1/responses"));
        server.stop();
    }

    func testModelRetrievalResolvesPlainProviderPrefixedAndUnknownIdentifiers() throws {
        let resolvedConfig: ResolvedRuntimeConfig = try self.makeResolvedConfig();
        let server: RestHttpServer = try self.startServingServer(resolvedConfig: resolvedConfig);

        let plainResponse: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /v1/models/synthetic-chat-model HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");
        let (plainStatus, plainEnvelope): (Int, [String: Any]) = try self.decodeObjectEnvelope(plainResponse);
        XCTAssertEqual(plainStatus, 200);
        XCTAssertEqual(plainEnvelope["id"] as? String, "synthetic-chat-model");

        let prefixedResponse: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /v1/models/vendor/synthetic-chat-model HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");
        let (prefixedStatus, prefixedEnvelope): (Int, [String: Any]) = try self.decodeObjectEnvelope(prefixedResponse);
        XCTAssertEqual(prefixedStatus, 200, "a provider-prefixed identifier must resolve");
        XCTAssertEqual(prefixedEnvelope["id"] as? String, "synthetic-chat-model");

        let unknownResponse: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /v1/models/unknown-model HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");
        let (unknownStatus, unknownEnvelope): (Int, [String: Any]) = try self.decodeObjectEnvelope(unknownResponse);
        XCTAssertEqual(unknownStatus, 404);
        let errorObject: [String: Any] = try self.requireErrorObject(unknownEnvelope);
        XCTAssertEqual(errorObject["type"] as? String, "invalid_request_error");
        server.stop();
    }

    func testCacheStatsJourneyAnswersTheZeroedPersistentCacheSummary() throws {
        let resolvedConfig: ResolvedRuntimeConfig = try self.makeResolvedConfig();
        let server: RestHttpServer = try self.startServingServer(resolvedConfig: resolvedConfig);

        let responseText: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /v1/cache/stats HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");
        let (statusCode, envelope): (Int, [String: Any]) = try self.decodeObjectEnvelope(responseText);
        XCTAssertEqual(statusCode, 200);
        XCTAssertEqual(envelope["persistent_prompt_cache_hits"] as? UInt64, 0);
        XCTAssertEqual(envelope["persistent_prompt_cache_misses"] as? UInt64, 0);
        XCTAssertEqual(envelope["persistent_prompt_cache_maximum_size_bytes"] as? UInt64, 50_000_000_000);
        server.stop();
    }

    func testEmptyDiscoveryWithAReadyWorkerAdvertisesTheReadyModel() throws {
        let workerCapabilities: WorkerModelCapabilities = WorkerModelCapabilities.from(
            chatCapabilities: ChatModelCapabilities(
                supportsReasoning: false,
                supportsToolCalls: true,
                hasVision: false,
                maxInputTokens: 2_047,
                maxOutputTokens: 2_047,
                contextWindow: 2_048));
        let workerHealthState: WorkerHealthState = WorkerHealthState();
        workerHealthState.publish(WorkerHealthSnapshot.readyWithModel(
            modelId: "ready-chat-model",
            capabilities: workerCapabilities));
        let resolvedConfig: ResolvedRuntimeConfig = try self.makeResolvedConfig(discoveredModels: []);
        let server: RestHttpServer = try self.startServingServer(
            resolvedConfig: resolvedConfig,
            workerHealthState: workerHealthState);

        let responseText: String? = RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n");
        let (statusCode, envelope): (Int, [String: Any]) = try self.decodeObjectEnvelope(responseText);
        XCTAssertEqual(statusCode, 200);
        let models: Array<[String: Any]> = try self.requireModelsArray(envelope);
        XCTAssertEqual(models.count, 1);
        XCTAssertEqual(models[0]["id"] as? String, "ready-chat-model");
        XCTAssertEqual(models[0]["context_window"] as? UInt32, 2_048);
        server.stop();
    }

    // MARK: - Journey helpers

    private func startServingServer(
        resolvedConfig: ResolvedRuntimeConfig,
        workerHealthState: WorkerHealthState = WorkerHealthState()
    ) throws -> RestHttpServer {
        let routeTable: RestRouteTable = RestEndpointRoutes.servingRouteTable(
            resolvedRuntimeConfig: resolvedConfig,
            workerHealthState: workerHealthState);
        return try RestHttpServer.start(
            bindEndpoint: SocketEndpoint.loopback(port: 0),
            routeTable: routeTable);
    }

    private func makeResolvedConfig(
        discoveredModels: Array<DiscoveryDiscoveredModel>? = nil
    ) throws -> ResolvedRuntimeConfig {
        let discoveredModels: Array<DiscoveryDiscoveredModel> = discoveredModels ?? [
            DiscoveryDiscoveredModel(
                modelId: "synthetic-chat-model",
                providerModelId: nil,
                modelFamily: .modernbert,
                revision: "0000000000000000000000000000000000000000",
                modelDirectory: FilePath(string: "/models/synthetic-chat-model"),
                capabilities: .chat(DiscoveryChatModelCapabilities(
                    contextWindowTokens: 4_096,
                    maximumInputTokens: 4_095,
                    maximumOutputTokens: 1_024,
                    supportsVision: false,
                    supportsReasoning: false,
                    supportsToolCalls: false)),
                license: nil,
                modelSizeBytes: 400_000_000)
        ];
        var modelPolicyCatalog: Dictionary<String, RuntimeModelPolicy> = Dictionary();
        for discoveredModel: DiscoveryDiscoveredModel in discoveredModels {
            modelPolicyCatalog[discoveredModel.modelId] = try ResolvedModelPolicyCatalog.resolve(
                userConfig: try self.emptyConfig(),
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

    private func emptyConfig() throws -> AstronomicalConfig {
        let temporaryStateDirectory: String = NSTemporaryDirectory() + "arestm-\(UUID().uuidString.prefix(8))";
        try FileManager.default.createDirectory(atPath: temporaryStateDirectory, withIntermediateDirectories: true);
        self.temporaryRootPath = temporaryStateDirectory;
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            FilePath(string: temporaryStateDirectory),
            runtimeInstance: AstronomicalRuntimeInstance.development);
        try FileManager.default.createDirectory(
            atPath: instancePaths.stateDirectory.string,
            withIntermediateDirectories: true);
        return try AstronomicalConfig.loadFromInstancePaths(instancePaths);
    }

    private func decodeObjectEnvelope(_ responseText: String?) throws -> (Int, [String: Any]) {
        let bodyText: String = try self.requireResponseText(responseText);
        guard let statusToken: Substring = bodyText.split(separator: " ", maxSplits: 2).dropFirst().first else {
            throw RestModelsTestFailure.malformedStatusLine;
        }
        guard let statusCode: Int = Int(statusToken) else {
            throw RestModelsTestFailure.malformedStatusLine;
        }
        guard let bodyStart: String.Index = bodyText.range(of: "\r\n\r\n")?.upperBound else {
            throw RestModelsTestFailure.missingResponseBody;
        }
        let bodyJsonText: String = String(bodyText[bodyStart...]);
        if bodyJsonText.isEmpty {
            throw RestModelsTestFailure.missingResponseBody;
        }
        guard let envelopeObject: [String: Any] = try JSONSerialization.jsonObject(with: Data(bodyJsonText.utf8)) as? [String: Any] else {
            throw RestModelsTestFailure.nonJsonEnvelope;
        }
        return (statusCode, envelopeObject);
    }

    private func requireModelsArray(_ envelope: [String: Any]) throws -> Array<[String: Any]> {
        guard let modelsArray: Array<Any> = envelope["data"] as? Array<Any> else {
            throw RestModelsTestFailure.missingModelsArray;
        }
        var modelObjects: Array<[String: Any]> = Array();
        for modelEntry: Any in modelsArray {
            guard let modelObject: [String: Any] = modelEntry as? [String: Any] else {
                throw RestModelsTestFailure.missingModelsArray;
            }
            modelObjects.append(modelObject);
        }
        return modelObjects;
    }

    private func requireErrorObject(_ envelope: [String: Any]) throws -> [String: Any] {
        guard let errorObject: [String: Any] = envelope["error"] as? [String: Any] else {
            throw RestModelsTestFailure.missingErrorObject;
        }
        return errorObject;
    }

    private func requireResponseText(_ responseText: String?) throws -> String {
        guard let unwrappedResponseText: String = responseText else {
            throw RestModelsTestFailure.missingResponse;
        }
        return unwrappedResponseText;
    }

    private var temporaryRootPath: String?;

    deinit {
        if let temporaryRootPath: String = self.temporaryRootPath {
            try? FileManager.default.removeItem(atPath: temporaryRootPath);
        }
    }
}

private enum RestModelsTestFailure: Error {
    case missingResponse;
    case malformedStatusLine;
    case missingResponseBody;
    case nonJsonEnvelope;
    case missingModelsArray;
    case missingErrorObject;
}
