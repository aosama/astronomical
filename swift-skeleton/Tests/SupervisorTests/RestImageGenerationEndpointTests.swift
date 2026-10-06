import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;

@testable import Supervisor;

/// A chat-independent image executor scripted per journey: it records every
/// admitted command and answers with a start failure, an execution failure,
/// or the scripted completed output.
final class ScriptedImageGenerationExecutor: ImageGenerationExecuting, @unchecked Sendable {

    let scriptedHealthSnapshot: WorkerHealthSnapshot;
    let scriptedOutput: ImageGenerationOutput?;
    let scriptedExecutionError: ImageGenerationExecutionError?;
    let scriptedStartError: GenerationStartError?;
    private let recordsLock: NSLock;
    private var recordedCommands: Array<ImageGenerationCommand>;

    init(
        healthSnapshot: WorkerHealthSnapshot,
        output: ImageGenerationOutput? = nil,
        executionError: ImageGenerationExecutionError? = nil,
        startError: GenerationStartError? = nil
    ) {
        self.scriptedHealthSnapshot = healthSnapshot;
        self.scriptedOutput = output;
        self.scriptedExecutionError = executionError;
        self.scriptedStartError = startError;
        self.recordsLock = NSLock();
        self.recordedCommands = Array();
    }

    var receivedCommands: Array<ImageGenerationCommand> {
        self.recordsLock.lock();
        defer { self.recordsLock.unlock(); }
        return self.recordedCommands;
    }

    func startImageGeneration(
        _ imageGenerationCommand: ImageGenerationCommand
    ) throws -> ImageGenerationOutput {
        self.recordsLock.lock();
        self.recordedCommands.append(imageGenerationCommand);
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
 * Acceptance journeys for the POST /v1/images/generations surface: the
 * fixture image renders as base64 with the worker-reported metadata,
 * unsupported fields and out-of-envelope dimensions reject before any queue
 * admission, wrong-modality models reject, and worker failures answer with
 * sanitized envelopes. The executor is scripted, so no worker process runs.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class RestImageGenerationEndpointTests {

    private static let imageModelId: String = "black-forest-labs/FLUX.2-klein-4B";
    private static let qwenImageModelId: String = "mlx-community/Qwen-Image-2.1";

    @Test
    func should_generate_one_base64_png_through_the_public_http_journey() throws {
        let scriptedExecutor: ScriptedImageGenerationExecutor =
            RestImageGenerationEndpointTests.imageExecutor();
        let routeTable: RestRouteTable = try RestImageGenerationEndpointTests.imageRouteTable(
            discoveredModels: [RestImageGenerationEndpointTests.fluxImageModel()],
            imageExecutor: scriptedExecutor);

        let imageResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            routePath: RestImageGenerationEndpoint.routePath,
            requestBody: "{\"model\":\"\(RestImageGenerationEndpointTests.imageModelId)\","
                + "\"prompt\":\"Romeo\",\"width\":1024,\"height\":1024,\"seed\":7,"
                + "\"response_format\":\"b64_json\"}");

        #expect(imageResponse.statusCode == 200);
        let envelope: [String: Any] = try RestChatJourneySupport.decodeObjectEnvelope(imageResponse);
        let dataRows: Array<Any> = try #require(envelope["data"] as? Array<Any>);
        let firstRow: [String: Any] = try #require(dataRows[0] as? [String: Any]);
        let base64Payload: String = try #require(firstRow["b64_json"] as? String);
        #expect(Data(base64Encoded: base64Payload) != nil,
            "the image payload should be valid base64");
        #expect(firstRow["mime_type"] as? String == "image/png");
        #expect(firstRow["model_revision"] as? String == "fixture-revision");
        #expect(firstRow["seed"] as? UInt64 == 7);
        #expect(firstRow["width"] as? UInt64 == 1_024);
        #expect(firstRow["height"] as? UInt64 == 1_024);

        let commands: Array<ImageGenerationCommand> = scriptedExecutor.receivedCommands;
        #expect(commands.count == 1);
        #expect(commands[0].model == RestImageGenerationEndpointTests.imageModelId);
        #expect(commands[0].settings.seed == 7);
        // The diffusion schedule is server-owned: the executed command carries
        // the discovery default, never a caller-supplied step count.
        #expect(commands[0].settings.steps == 4);
        #expect(commands[0].settings.guidanceThousandths == 1_000);
    }

    @Test
    func should_reject_every_unsupported_image_field_before_dispatch() throws {
        let invalidRequests: Array<(String, String)> = [
            ("{\"model\":\"\(RestImageGenerationEndpointTests.imageModelId)\",\"prompt\":\" \",\"width\":1024,\"height\":1024,\"response_format\":\"b64_json\"}", "prompt"),
            ("{\"model\":\"\(RestImageGenerationEndpointTests.imageModelId)\",\"prompt\":\"Romeo\",\"width\":63,\"height\":1024,\"response_format\":\"b64_json\"}", "width"),
            ("{\"model\":\"\(RestImageGenerationEndpointTests.imageModelId)\",\"prompt\":\"Romeo\",\"width\":80,\"height\":1023,\"response_format\":\"b64_json\"}", "height"),
            ("{\"model\":\"\(RestImageGenerationEndpointTests.imageModelId)\",\"prompt\":\"Romeo\",\"width\":1024,\"height\":1024,\"steps\":4,\"response_format\":\"b64_json\"}", "request"),
            ("{\"model\":\"\(RestImageGenerationEndpointTests.imageModelId)\",\"prompt\":\"Romeo\",\"width\":1024,\"height\":1024,\"guidance\":1.0,\"response_format\":\"b64_json\"}", "request"),
            ("{\"model\":\"\(RestImageGenerationEndpointTests.imageModelId)\",\"prompt\":\"Romeo\",\"width\":1024,\"height\":1024,\"response_format\":\"url\"}", "response_format"),
            ("{\"model\":\"\(RestImageGenerationEndpointTests.imageModelId)\",\"prompt\":\"Romeo\",\"width\":1024,\"height\":1024,\"response_format\":\"b64_json\",\"n\":2}", "n"),
            ("{\"model\":\"\(RestImageGenerationEndpointTests.imageModelId)\",\"prompt\":\"Romeo\",\"width\":1024,\"height\":1024,\"response_format\":\"b64_json\",\"quality\":\"hd\"}", "request"),
        ];
        for (requestBody, expectedParameter) in invalidRequests {
            let scriptedExecutor: ScriptedImageGenerationExecutor =
                RestImageGenerationEndpointTests.imageExecutor();
            let routeTable: RestRouteTable = try RestImageGenerationEndpointTests.imageRouteTable(
                discoveredModels: [RestImageGenerationEndpointTests.fluxImageModel()],
                imageExecutor: scriptedExecutor);

            let imageResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
                routeTable: routeTable,
                routePath: RestImageGenerationEndpoint.routePath,
                requestBody: requestBody);

            #expect(imageResponse.statusCode == 400);
            let errorObject: [String: Any] = try RestChatJourneySupport.requireErrorObject(
                try RestChatJourneySupport.decodeObjectEnvelope(imageResponse));
            #expect(errorObject["param"] as? String == expectedParameter);
            #expect(scriptedExecutor.receivedCommands.isEmpty);
        }
    }

    @Test
    func should_reject_out_of_envelope_dimensions_before_queue_admission() throws {
        let outOfEnvelopeRequests: Array<(String, String, String)> = [
            ("64", "width", "at least 256 pixels"),
            ("336", "height", "multiple of 32 pixels"),
        ];
        for (violatedPixels, expectedParameter, expectedConstraint) in outOfEnvelopeRequests {
            let widthText: String = expectedParameter == "width" ? violatedPixels : "1024";
            let heightText: String = expectedParameter == "height" ? violatedPixels : "1024";
            let scriptedExecutor: ScriptedImageGenerationExecutor =
                RestImageGenerationEndpointTests.imageExecutor();
            let routeTable: RestRouteTable = try RestImageGenerationEndpointTests.imageRouteTable(
                discoveredModels: [RestImageGenerationEndpointTests.qwenImageModel()],
                imageExecutor: scriptedExecutor);

            let imageResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
                routeTable: routeTable,
                routePath: RestImageGenerationEndpoint.routePath,
                requestBody: "{\"model\":\"\(RestImageGenerationEndpointTests.qwenImageModelId)\","
                    + "\"prompt\":\"Romeo\",\"width\":\(widthText),\"height\":\(heightText),"
                    + "\"response_format\":\"b64_json\"}");

            #expect(imageResponse.statusCode == 400);
            let envelope: [String: Any] =
                try RestChatJourneySupport.decodeObjectEnvelope(imageResponse);
            let errorObject: [String: Any] =
                try RestChatJourneySupport.requireErrorObject(envelope);
            #expect(errorObject["param"] as? String == expectedParameter);
            #expect(errorObject["code"] as? String == "invalid_request");
            let rejectionMessage: String = try #require(errorObject["message"] as? String);
            #expect(rejectionMessage.contains(expectedConstraint),
                "the message should name the violated constraint: \(rejectionMessage)");
            #expect(rejectionMessage.contains(RestImageGenerationEndpointTests.qwenImageModelId),
                "the message should name the target model: \(rejectionMessage)");
            #expect(scriptedExecutor.receivedCommands.isEmpty,
                "an out-of-envelope request must not be admitted to the queue");
        }
    }

    @Test
    func should_admit_an_in_envelope_request_at_the_envelope_boundaries() throws {
        let scriptedExecutor: ScriptedImageGenerationExecutor =
            RestImageGenerationEndpointTests.imageExecutor();
        let routeTable: RestRouteTable = try RestImageGenerationEndpointTests.imageRouteTable(
            discoveredModels: [RestImageGenerationEndpointTests.qwenImageModel()],
            imageExecutor: scriptedExecutor);

        let imageResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            routePath: RestImageGenerationEndpoint.routePath,
            requestBody: "{\"model\":\"\(RestImageGenerationEndpointTests.qwenImageModelId)\","
                + "\"prompt\":\"A moonlit balcony scene from Romeo and Juliet\","
                + "\"width\":1024,\"height\":256,\"response_format\":\"b64_json\"}");

        #expect(imageResponse.statusCode == 200);
        #expect(scriptedExecutor.receivedCommands.count == 1);
    }

    @Test
    func should_reject_a_chat_model_before_image_queue_admission() throws {
        let scriptedExecutor: ScriptedImageGenerationExecutor =
            RestImageGenerationEndpointTests.imageExecutor();
        let routeTable: RestRouteTable = try RestImageGenerationEndpointTests.imageRouteTable(
            discoveredModels: [RestImageGenerationEndpointTests.chatOnlyModel()],
            imageExecutor: scriptedExecutor);

        let imageResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            routePath: RestImageGenerationEndpoint.routePath,
            requestBody: "{\"model\":\"chat-only-model\",\"prompt\":\"Romeo\","
                + "\"width\":1024,\"height\":1024,\"response_format\":\"b64_json\"}");

        #expect(imageResponse.statusCode == 400);
        let errorObject: [String: Any] = try RestChatJourneySupport.requireErrorObject(
            try RestChatJourneySupport.decodeObjectEnvelope(imageResponse));
        #expect(errorObject["code"] as? String == "model_capability_mismatch");
        #expect(errorObject["param"] as? String == "model");
        #expect(scriptedExecutor.receivedCommands.isEmpty);
    }

    @Test
    func should_not_expose_worker_image_failure_reasons_or_local_paths() throws {
        let fictionalPrivatePath: String = "/Users/fictional-person/private-model/diffusion.safetensors";
        let scriptedExecutor: ScriptedImageGenerationExecutor = ScriptedImageGenerationExecutor(
            healthSnapshot: RestImageGenerationEndpointTests.readySnapshot(),
            executionError: .workerFailure(.fatalExecution(
                reason: "native execution failed while mapping \(fictionalPrivatePath)")));
        let routeTable: RestRouteTable = try RestImageGenerationEndpointTests.imageRouteTable(
            discoveredModels: [RestImageGenerationEndpointTests.fluxImageModel()],
            imageExecutor: scriptedExecutor);

        let imageResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            routePath: RestImageGenerationEndpoint.routePath,
            requestBody: "{\"model\":\"\(RestImageGenerationEndpointTests.imageModelId)\","
                + "\"prompt\":\"Romeo\",\"width\":1024,\"height\":1024,"
                + "\"response_format\":\"b64_json\"}");

        #expect(imageResponse.statusCode == 500);
        let envelope: [String: Any] = try RestChatJourneySupport.decodeObjectEnvelope(imageResponse);
        let errorObject: [String: Any] = try RestChatJourneySupport.requireErrorObject(envelope);
        #expect(errorObject["code"] as? String == "image_generation_failed");
        #expect(errorObject["message"] as? String
            == "image generation failed in the local worker");
        let responseText: String = RestChatJourneySupport.responseText(imageResponse);
        #expect(responseText.contains(fictionalPrivatePath) == false,
            "worker failure must not leak a local path");
    }

    @Test
    func should_map_worker_and_deadline_failures_to_their_envelopes() throws {
        let journeyExpectations: Array<(ImageGenerationExecutionError, Int, String, String)> = [
            (.workerFailure(.engineBusy), 429, "server_capacity", "the image engine is busy"),
            (.workerFailure(.modelDoesNotSupportImageGeneration), 400, "model_capability_mismatch",
                "the requested model does not support image generation"),
            (.workerUnavailable, 503, "worker_unavailable", "the local worker is unavailable"),
            (.deadlineExceeded, 504, "image_generation_timeout",
                "image generation exceeded its bounded execution deadline"),
        ];
        for (executionError, expectedStatus, expectedCode, expectedMessage) in journeyExpectations {
            let scriptedExecutor: ScriptedImageGenerationExecutor = ScriptedImageGenerationExecutor(
                healthSnapshot: RestImageGenerationEndpointTests.readySnapshot(),
                executionError: executionError);
            let routeTable: RestRouteTable = try RestImageGenerationEndpointTests.imageRouteTable(
                discoveredModels: [RestImageGenerationEndpointTests.fluxImageModel()],
                imageExecutor: scriptedExecutor);

            let imageResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
                routeTable: routeTable,
                routePath: RestImageGenerationEndpoint.routePath,
                requestBody: "{\"model\":\"\(RestImageGenerationEndpointTests.imageModelId)\","
                    + "\"prompt\":\"Romeo\",\"width\":1024,\"height\":1024,"
                    + "\"response_format\":\"b64_json\"}");

            #expect(imageResponse.statusCode == expectedStatus);
            let errorObject: [String: Any] = try RestChatJourneySupport.requireErrorObject(
                try RestChatJourneySupport.decodeObjectEnvelope(imageResponse));
            #expect(errorObject["code"] as? String == expectedCode);
            #expect(errorObject["message"] as? String == expectedMessage);
        }
    }

    @Test
    func should_reject_malformed_json_with_the_invalid_json_code() throws {
        let routeTable: RestRouteTable = try RestImageGenerationEndpointTests.imageRouteTable(
            discoveredModels: [RestImageGenerationEndpointTests.fluxImageModel()],
            imageExecutor: RestImageGenerationEndpointTests.imageExecutor());

        let imageResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            routePath: RestImageGenerationEndpoint.routePath,
            requestBody: "{\"model\":");

        #expect(imageResponse.statusCode == 400);
        let errorObject: [String: Any] = try RestChatJourneySupport.requireErrorObject(
            try RestChatJourneySupport.decodeObjectEnvelope(imageResponse));
        #expect(errorObject["code"] as? String == "invalid_json");
    }

    // MARK: Shared fixture helpers

    private static func readySnapshot() -> WorkerHealthSnapshot {
        return WorkerHealthSnapshot.readyWithoutModel(
            machineMlxMemoryCeilingBytes: 1,
            effectiveMlxMemoryCeilingBytes: 1,
            minimumMlxMemoryCeilingBytes: 1);
    }

    private static func imageExecutor() -> ScriptedImageGenerationExecutor {
        return ScriptedImageGenerationExecutor(
            healthSnapshot: readySnapshot(),
            output: ImageGenerationOutput(
                generatedImage: GeneratedImage(
                    mimeType: "image/png",
                    encodedBytes: Array("fixture-png".utf8)),
                resultMetadata: ImageGenerationResultMetadata(
                    widthPixels: 1_024,
                    heightPixels: 1_024,
                    steps: 4,
                    guidanceThousandths: 1_000,
                    seed: 7,
                    elapsedMillis: 42)));
    }

    private static func fluxImageModel() -> DiscoveryDiscoveredModel {
        return DiscoveryDiscoveredModel(
            modelId: imageModelId,
            providerModelId: "black-forest-labs/FLUX.2-klein-4B",
            modelFamily: .flux2Klein,
            revision: "fixture-revision",
            modelDirectory: FilePath(string: "/fixtures/models/flux2-klein"),
            capabilities: .imageGeneration(DiscoveryImageGenerationCapabilities(
                supportsTextToImage: true,
                supportsImageEditing: false,
                supportsMultipleReferenceImages: false,
                defaultSteps: 4,
                minimumDimensionPixels: 64,
                maximumDimensionPixels: 1_024,
                dimensionMultiplePixels: 16)),
            license: .apache20,
            modelSizeBytes: 1);
    }

    private static func qwenImageModel() -> DiscoveryDiscoveredModel {
        return DiscoveryDiscoveredModel(
            modelId: qwenImageModelId,
            providerModelId: "mlx-community/Qwen-Image-2.1",
            modelFamily: .qwenImage21,
            revision: "fixture-revision",
            modelDirectory: FilePath(string: "/fixtures/models/qwen-image-21"),
            capabilities: .imageGeneration(DiscoveryImageGenerationCapabilities(
                supportsTextToImage: true,
                supportsImageEditing: false,
                supportsMultipleReferenceImages: false,
                defaultSteps: 40,
                minimumDimensionPixels: 256,
                maximumDimensionPixels: 1_024,
                dimensionMultiplePixels: 32)),
            license: .qwenResearch,
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

    private static func imageRouteTable(
        discoveredModels: Array<DiscoveryDiscoveredModel>,
        imageExecutor: ImageGenerationExecuting
    ) throws -> RestRouteTable {
        return RestEndpointRoutes.servingRouteTable(
            resolvedRuntimeConfig: try RestChatJourneySupport.makeResolvedConfig(
                discoveredModels: discoveredModels),
            workerHealthState: WorkerHealthState(),
            instancePaths: AstronomicalInstancePaths.forExplicitStateDirectory(
                FilePath(string: "/rest-image-journey-state"),
                defaultBindAddress: SocketEndpoint.loopback(port: 0)),
            buildIdentity: ApplicationBuildIdentity(
                version: "0.0.0-test", buildNumber: 0, commit: "journey", isDirty: false),
            imageContext: RestImageGenerationRouteContext(
                imageExecutor: imageExecutor,
                requestIdAllocator: ChatRequestIdAllocator(),
                resolvedRuntimeConfig: try RestChatJourneySupport.makeResolvedConfig(
                    discoveredModels: discoveredModels)));
    }
}
