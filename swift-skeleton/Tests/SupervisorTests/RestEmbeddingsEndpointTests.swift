import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;

@testable import Supervisor;

/// A chat-independent embeddings executor scripted per journey: it records
/// every admitted command and answers with a start failure, an execution
/// failure, or the scripted completed output.
final class ScriptedEmbeddingsExecutor: EmbeddingsExecuting, @unchecked Sendable {

    let scriptedHealthSnapshot: WorkerHealthSnapshot;
    let scriptedOutput: EmbeddingsOutput?;
    let scriptedExecutionError: EmbeddingsExecutionError?;
    let scriptedStartError: GenerationStartError?;
    private let recordsLock: NSLock;
    private var recordedCommands: Array<EmbeddingsCommand>;

    init(
        healthSnapshot: WorkerHealthSnapshot,
        output: EmbeddingsOutput? = nil,
        executionError: EmbeddingsExecutionError? = nil,
        startError: GenerationStartError? = nil
    ) {
        self.scriptedHealthSnapshot = healthSnapshot;
        self.scriptedOutput = output;
        self.scriptedExecutionError = executionError;
        self.scriptedStartError = startError;
        self.recordsLock = NSLock();
        self.recordedCommands = Array();
    }

    var receivedCommands: Array<EmbeddingsCommand> {
        self.recordsLock.lock();
        defer { self.recordsLock.unlock(); }
        return self.recordedCommands;
    }

    func startEmbeddingsGeneration(
        _ embeddingsCommand: EmbeddingsCommand
    ) throws -> EmbeddingsOutput {
        self.recordsLock.lock();
        self.recordedCommands.append(embeddingsCommand);
        self.recordsLock.unlock();
        if let scriptedStartError = self.scriptedStartError {
            throw scriptedStartError;
        }
        if let scriptedExecutionError = self.scriptedExecutionError {
            throw scriptedExecutionError;
        }
        return try #require(self.scriptedOutput);
    }

    func workerHealthSnapshot() -> WorkerHealthSnapshot {
        return self.scriptedHealthSnapshot;
    }
}

/**
 * Acceptance journeys for the POST /v1/embeddings surface: the Romeo and
 * Juliet fixture embeds through the scripted executor in both encoding
 * formats, chat-only models and empty inputs reject before dispatch, start
 * and execution failures map to their stable envelopes without leaking
 * worker reasons or local paths. The executor is scripted, so no worker
 * process runs.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class RestEmbeddingsEndpointTests {

    private static let embeddingModelId: String = "nomicai-modernbert-embed-base-8bit";
    private static let romeoLine: String = "O Romeo, Romeo, wherefore art thou Romeo?";

    @Test
    func should_embed_romeo_and_juliet_through_the_public_http_journey() throws {
        let scriptedExecutor: ScriptedEmbeddingsExecutor =
            RestEmbeddingsEndpointTests.embeddingsExecutor();
        let routeTable: RestRouteTable = try RestEmbeddingsEndpointTests.embeddingsRouteTable(
            discoveredModels: [RestEmbeddingsEndpointTests.embeddingModel()],
            embeddingsExecutor: scriptedExecutor);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            routePath: RestEmbeddingsEndpoint.routePath, requestBody: "{\"model\":\"\(RestEmbeddingsEndpointTests.embeddingModelId)\","
                + "\"input\":\"\(RestEmbeddingsEndpointTests.romeoLine)\"}")
            ;

        #expect(chatResponse.statusCode == 200);
        let envelope: [String: Any] = try RestChatJourneySupport.decodeObjectEnvelope(chatResponse);
        #expect(envelope["object"] as? String == "list");
        #expect(envelope["model"] as? String == RestEmbeddingsEndpointTests.embeddingModelId);
        let dataRows: Array<Any> = try #require(envelope["data"] as? Array<Any>);
        let firstRow: [String: Any] = try #require(dataRows[0] as? [String: Any]);
        #expect(firstRow["object"] as? String == "embedding");
        let vectorComponents: Array<Any> = try #require(firstRow["embedding"] as? Array<Any>);
        #expect(vectorComponents.count == 768);
        let usage: [String: Any] = try RestChatJourneySupport.requireUsage(envelope);
        #expect(usage["prompt_tokens"] as? UInt64 == 8);
        #expect(usage["completion_tokens"] as? UInt64 == 0);

        let commands: Array<EmbeddingsCommand> = scriptedExecutor.receivedCommands;
        #expect(commands.count == 1);
        #expect(commands[0].model == RestEmbeddingsEndpointTests.embeddingModelId);
        #expect(commands[0].inputs == [RestEmbeddingsEndpointTests.romeoLine]);
    }

    @Test
    func should_render_base64_encoded_vectors_for_base64_format() throws {
        let scriptedExecutor: ScriptedEmbeddingsExecutor = ScriptedEmbeddingsExecutor(
            healthSnapshot: RestEmbeddingsEndpointTests.readySnapshot(),
            output: EmbeddingsOutput(
                embeddings: [[0.25, -0.5]],
                inputTokenCounts: [2]));
        let routeTable: RestRouteTable = try RestEmbeddingsEndpointTests.embeddingsRouteTable(
            discoveredModels: [RestEmbeddingsEndpointTests.embeddingModel()],
            embeddingsExecutor: scriptedExecutor);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            routePath: RestEmbeddingsEndpoint.routePath, requestBody: "{\"model\":\"\(RestEmbeddingsEndpointTests.embeddingModelId)\","
                + "\"input\":\"hello\",\"encoding_format\":\"base64\"}")
            ;

        #expect(chatResponse.statusCode == 200);
        let envelope: [String: Any] = try RestChatJourneySupport.decodeObjectEnvelope(chatResponse);
        let dataRows: Array<Any> = try #require(envelope["data"] as? Array<Any>);
        let firstRow: [String: Any] = try #require(dataRows[0] as? [String: Any]);
        let encodedVectorText: String = try #require(firstRow["embedding"] as? String);
        let decodedVectorBytes: Data = try #require(Data(base64Encoded: encodedVectorText));
        #expect(decodedVectorBytes.count == 8);
        let firstComponent: Float = decodedVectorBytes.withUnsafeBytes { (vectorBytes) -> Float in
            return Float(bitPattern: vectorBytes.loadUnaligned(fromByteOffset: 0, as: UInt32.self).littleEndian);
        };
        #expect(firstComponent == 0.25);
    }

    @Test
    func should_reject_a_chat_model_before_embeddings_dispatch() throws {
        let scriptedExecutor: ScriptedEmbeddingsExecutor =
            RestEmbeddingsEndpointTests.embeddingsExecutor();
        let routeTable: RestRouteTable = try RestEmbeddingsEndpointTests.embeddingsRouteTable(
            discoveredModels: [RestEmbeddingsEndpointTests.chatOnlyModel()],
            embeddingsExecutor: scriptedExecutor);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            routePath: RestEmbeddingsEndpoint.routePath, requestBody: "{\"model\":\"chat-only-model\","
                + "\"input\":\"\(RestEmbeddingsEndpointTests.romeoLine)\"}")
            ;

        #expect(chatResponse.statusCode == 400);
        let errorObject: [String: Any] = try RestChatJourneySupport.requireErrorObject(
            try RestChatJourneySupport.decodeObjectEnvelope(chatResponse));
        #expect(errorObject["code"] as? String == "model_capability_mismatch");
        #expect(errorObject["param"] as? String == "model");
        #expect(scriptedExecutor.receivedCommands.isEmpty);
    }

    @Test
    func should_reject_empty_input_before_embeddings_dispatch() throws {
        let scriptedExecutor: ScriptedEmbeddingsExecutor =
            RestEmbeddingsEndpointTests.embeddingsExecutor();
        let routeTable: RestRouteTable = try RestEmbeddingsEndpointTests.embeddingsRouteTable(
            discoveredModels: [RestEmbeddingsEndpointTests.embeddingModel()],
            embeddingsExecutor: scriptedExecutor);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            routePath: RestEmbeddingsEndpoint.routePath, requestBody: "{\"model\":\"\(RestEmbeddingsEndpointTests.embeddingModelId)\",\"input\":[]}")
            ;

        #expect(chatResponse.statusCode == 400);
        let errorObject: [String: Any] = try RestChatJourneySupport.requireErrorObject(
            try RestChatJourneySupport.decodeObjectEnvelope(chatResponse));
        #expect(errorObject["param"] as? String == "input");
        #expect(scriptedExecutor.receivedCommands.isEmpty);
    }

    @Test
    func should_reject_malformed_json_with_the_invalid_json_code() throws {
        let routeTable: RestRouteTable = try RestEmbeddingsEndpointTests.embeddingsRouteTable(
            discoveredModels: [RestEmbeddingsEndpointTests.embeddingModel()],
            embeddingsExecutor: RestEmbeddingsEndpointTests.embeddingsExecutor());

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            routePath: RestEmbeddingsEndpoint.routePath, requestBody: "{\"model\":")
            ;

        #expect(chatResponse.statusCode == 400);
        let errorObject: [String: Any] = try RestChatJourneySupport.requireErrorObject(
            try RestChatJourneySupport.decodeObjectEnvelope(chatResponse));
        #expect(errorObject["code"] as? String == "invalid_json");
    }

    @Test
    func should_not_expose_worker_embeddings_failure_reasons_or_local_paths() throws {
        let fictionalPrivatePath: String = "/Users/fictional-person/private-model/encoder.safetensors";
        let scriptedExecutor: ScriptedEmbeddingsExecutor = ScriptedEmbeddingsExecutor(
            healthSnapshot: RestEmbeddingsEndpointTests.readySnapshot(),
            executionError: .workerFailure(.fatalExecution(
                reason: "native execution failed while mapping \(fictionalPrivatePath)")));
        let routeTable: RestRouteTable = try RestEmbeddingsEndpointTests.embeddingsRouteTable(
            discoveredModels: [RestEmbeddingsEndpointTests.embeddingModel()],
            embeddingsExecutor: scriptedExecutor);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            routePath: RestEmbeddingsEndpoint.routePath, requestBody: "{\"model\":\"\(RestEmbeddingsEndpointTests.embeddingModelId)\","
                + "\"input\":\"\(RestEmbeddingsEndpointTests.romeoLine)\"}")
            ;

        #expect(chatResponse.statusCode == 500);
        let envelope: [String: Any] = try RestChatJourneySupport.decodeObjectEnvelope(chatResponse);
        let errorObject: [String: Any] = try RestChatJourneySupport.requireErrorObject(envelope);
        #expect(errorObject["code"] as? String == "embeddings_failed");
        #expect(errorObject["message"] as? String
            == "embedding generation failed in the local worker");
        let responseText: String = RestChatJourneySupport.responseText(chatResponse);
        #expect(responseText.contains(fictionalPrivatePath) == false,
            "worker failure must not leak a local path");
    }

    @Test
    func should_map_worker_context_length_rejection_to_a_bad_request() throws {
        let scriptedExecutor: ScriptedEmbeddingsExecutor = ScriptedEmbeddingsExecutor(
            healthSnapshot: RestEmbeddingsEndpointTests.readySnapshot(),
            executionError: .workerFailure(.contextLengthExceeded(
                actualTotalContextTokens: 9_000,
                maximumContextTokens: 8_192)));
        let routeTable: RestRouteTable = try RestEmbeddingsEndpointTests.embeddingsRouteTable(
            discoveredModels: [RestEmbeddingsEndpointTests.embeddingModel()],
            embeddingsExecutor: scriptedExecutor);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            routePath: RestEmbeddingsEndpoint.routePath, requestBody: "{\"model\":\"\(RestEmbeddingsEndpointTests.embeddingModelId)\","
                + "\"input\":\"\(RestEmbeddingsEndpointTests.romeoLine)\"}")
            ;

        #expect(chatResponse.statusCode == 400);
        let errorObject: [String: Any] = try RestChatJourneySupport.requireErrorObject(
            try RestChatJourneySupport.decodeObjectEnvelope(chatResponse));
        #expect(errorObject["code"] as? String == "context_length_exceeded");
        #expect(errorObject["param"] as? String == "input");
    }

    @Test
    func should_map_admission_failures_to_their_stable_envelopes() throws {
        let journeyExpectations: Array<(GenerationStartError, Int, String)> = [
            (.capacityUnavailable, 429, "server_capacity"),
            (.workerUnavailable, 503, "worker_unavailable"),
            (.modelLoadFailed(modelLoadFailureReason: "quantization rejected"), 503, "model_load_failed"),
            (.requestTooLarge(actualIpcMessageBytes: 4096, maximumIpcMessageBytes: 1024), 413, "request_too_large"),
        ];
        for (startError, expectedStatus, expectedCode) in journeyExpectations {
            let scriptedExecutor: ScriptedEmbeddingsExecutor = ScriptedEmbeddingsExecutor(
                healthSnapshot: RestEmbeddingsEndpointTests.readySnapshot(),
                startError: startError);
            let routeTable: RestRouteTable = try RestEmbeddingsEndpointTests.embeddingsRouteTable(
                discoveredModels: [RestEmbeddingsEndpointTests.embeddingModel()],
                embeddingsExecutor: scriptedExecutor);

            let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
                routeTable: routeTable,
                routePath: RestEmbeddingsEndpoint.routePath, requestBody: "{\"model\":\"\(RestEmbeddingsEndpointTests.embeddingModelId)\","
                    + "\"input\":\"\(RestEmbeddingsEndpointTests.romeoLine)\"}")
                ;

            #expect(chatResponse.statusCode == expectedStatus);
            let errorObject: [String: Any] = try RestChatJourneySupport.requireErrorObject(
                try RestChatJourneySupport.decodeObjectEnvelope(chatResponse));
            #expect(errorObject["code"] as? String == expectedCode);
        }
    }

    @Test
    func should_reject_a_busy_embedding_engine_with_too_many_requests() throws {
        let scriptedExecutor: ScriptedEmbeddingsExecutor = ScriptedEmbeddingsExecutor(
            healthSnapshot: RestEmbeddingsEndpointTests.readySnapshot(),
            executionError: .workerFailure(.engineBusy));
        let routeTable: RestRouteTable = try RestEmbeddingsEndpointTests.embeddingsRouteTable(
            discoveredModels: [RestEmbeddingsEndpointTests.embeddingModel()],
            embeddingsExecutor: scriptedExecutor);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            routePath: RestEmbeddingsEndpoint.routePath, requestBody: "{\"model\":\"\(RestEmbeddingsEndpointTests.embeddingModelId)\","
                + "\"input\":\"\(RestEmbeddingsEndpointTests.romeoLine)\"}")
            ;

        #expect(chatResponse.statusCode == 429);
        let errorObject: [String: Any] = try RestChatJourneySupport.requireErrorObject(
            try RestChatJourneySupport.decodeObjectEnvelope(chatResponse));
        #expect(errorObject["code"] as? String == "server_capacity");
        #expect(errorObject["message"] as? String == "the embedding engine is busy");
    }

    // MARK: Shared fixture helpers

    private static func readySnapshot() -> WorkerHealthSnapshot {
        return WorkerHealthSnapshot.readyWithoutModel(
            machineMlxMemoryCeilingBytes: 1,
            effectiveMlxMemoryCeilingBytes: 1,
            minimumMlxMemoryCeilingBytes: 1);
    }

    private static func embeddingsExecutor() -> ScriptedEmbeddingsExecutor {
        return ScriptedEmbeddingsExecutor(
            healthSnapshot: readySnapshot(),
            output: EmbeddingsOutput(
                embeddings: [Array(repeating: 0.0, count: 768)],
                inputTokenCounts: [8]));
    }

    private static func embeddingModel() -> DiscoveryDiscoveredModel {
        return DiscoveryDiscoveredModel(
            modelId: embeddingModelId,
            providerModelId: "mlx-community/\(embeddingModelId)",
            modelFamily: .modernbert,
            revision: "fixture-revision",
            modelDirectory: FilePath(string: "/fixtures/models/modernbert"),
            capabilities: .embeddings(DiscoveryEmbeddingModelCapabilities(
                vectorWidth: 768,
                maximumInputTokens: 8_192)),
            license: .apache20,
            modelSizeBytes: 1);
    }

    private static func chatOnlyModel() -> DiscoveryDiscoveredModel {
        return DiscoveryDiscoveredModel(
            modelId: "chat-only-model",
            providerModelId: nil,
            modelFamily: .qwen35,
            revision: "fixture-revision",
            modelDirectory: FilePath(string: "/fixtures/models/chat-only"),
            capabilities: .chat(DiscoveryChatModelCapabilities(
                contextWindowTokens: 2_048,
                maximumInputTokens: 1_024,
                maximumOutputTokens: 128,
                supportsVision: false,
                supportsReasoning: true,
                supportsToolCalls: true)),
            license: nil,
            modelSizeBytes: 1);
    }

    private static func embeddingsRouteTable(
        discoveredModels: Array<DiscoveryDiscoveredModel>,
        embeddingsExecutor: EmbeddingsExecuting
    ) throws -> RestRouteTable {
        return RestEndpointRoutes.servingRouteTable(
            resolvedRuntimeConfig: try RestChatJourneySupport.makeResolvedConfig(
                discoveredModels: discoveredModels),
            workerHealthState: WorkerHealthState(),
            instancePaths: AstronomicalInstancePaths.forExplicitStateDirectory(
                FilePath(string: "/rest-embeddings-journey-state"),
                defaultBindAddress: SocketEndpoint.loopback(port: 0)),
            buildIdentity: ApplicationBuildIdentity(
                version: "0.0.0-test", buildNumber: 0, commit: "journey", isDirty: false),
            embeddingsContext: RestEmbeddingsRouteContext(
                embeddingsExecutor: embeddingsExecutor,
                requestIdAllocator: ChatRequestIdAllocator(),
                resolvedRuntimeConfig: try RestChatJourneySupport.makeResolvedConfig(
                    discoveredModels: discoveredModels)));
    }
}
