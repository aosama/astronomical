import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import JourneyCategories;

@testable import Supervisor;

/**
 * Acceptance journeys for cross-modality model rejection, migrating the
 * image-only rows of apps/supervisor/tests/rest_api/openai_image_generation.rs:
 * an image-only model must refuse chat completions and Responses requests
 * before queue admission, with no generation command ever dispatched.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class RestModelModalityRejectionTests {

    @Test
    func should_reject_an_image_only_model_from_chat_before_queue_admission() throws {
        let imageModelId: String = "black-forest-labs/FLUX.2-klein-4B";
        let scriptedExecutor: ScriptedChatExecutor = ScriptedChatExecutor(
            healthSnapshot: WorkerHealthSnapshot.readyWithModel(
                modelId: imageModelId,
                capabilities: RestChatJourneySupport.readyChatCapabilities()));
        let routeTable: RestRouteTable = try RestChatJourneySupport.chatRouteTable(
            resolvedRuntimeConfig: try RestChatJourneySupport.makeResolvedConfig(
                discoveredModels: [
                    RestModelModalityRejectionTests.imageOnlyDiscoveredModel(modelId: imageModelId),
                ]),
            chatExecutor: scriptedExecutor);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(imageModelId)\",\"messages\":[{\"role\":\"user\",\"content\":\"Wherefore art thou Romeo?\"}],\"stream\":false}");

        #expect(chatResponse.statusCode == 400);
        let errorObject: [String: Any] = try RestChatJourneySupport.requireErrorObject(
            try RestChatJourneySupport.decodeObjectEnvelope(chatResponse));
        #expect(errorObject["param"] as? String == "model");
        #expect(errorObject["code"] as? String == "model_capability_mismatch");
        #expect(scriptedExecutor.receivedCommands.isEmpty,
            "the wrong-modality request must never reach the generation executor");
    }

    @Test
    func should_reject_an_image_only_model_from_responses_before_queue_admission() throws {
        let imageModelId: String = "black-forest-labs/FLUX.2-klein-4B";
        let scriptedExecutor: ScriptedChatExecutor = ScriptedChatExecutor(
            healthSnapshot: WorkerHealthSnapshot.readyWithModel(
                modelId: imageModelId,
                capabilities: RestResponsesJourneySupport.readyResponsesCapabilities()));
        let routeTable: RestRouteTable = try RestResponsesJourneySupport.responsesRouteTable(
            resolvedRuntimeConfig: try RestResponsesJourneySupport.makeResolvedConfig(
                discoveredModels: [
                    RestModelModalityRejectionTests.imageOnlyDiscoveredModel(modelId: imageModelId),
                ]),
            responsesExecutor: scriptedExecutor);

        let responsesResponse: RestHttpResponse = try RestResponsesJourneySupport.postResponses(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(imageModelId)\",\"input\":\"Wherefore art thou Romeo?\"}");

        #expect(responsesResponse.statusCode == 400);
        guard let envelopeObject: [String: Any] = try? JSONSerialization.jsonObject(
            with: responsesResponse.bodyBytes) as? [String: Any],
            let errorObject: [String: Any] = envelopeObject["error"] as? [String: Any] else {
            Issue.record(Comment(stringLiteral: "expected an error envelope, got \(RestResponsesJourneySupport.responseText(responsesResponse))"));
            return;
        }
        #expect(errorObject["param"] as? String == "model");
        #expect(errorObject["code"] as? String == "model_capability_mismatch");
        #expect(scriptedExecutor.receivedCommands.isEmpty,
            "the wrong-modality request must never reach the generation executor");
    }

    // MARK: Journey fixtures

    /// One discovered image-only model: resolving it yields a Flux worker
    /// policy, so every text surface must refuse it before admission.
    private static func imageOnlyDiscoveredModel(modelId: String) -> DiscoveryDiscoveredModel {
        return DiscoveryDiscoveredModel(
            modelId: modelId,
            providerModelId: modelId,
            modelFamily: .flux2Klein,
            revision: "fixture-revision",
            modelDirectory: FilePath(string: "/models/flux-fixture"),
            capabilities: .imageGeneration(DiscoveryImageGenerationCapabilities(
                supportsTextToImage: true,
                supportsImageEditing: false,
                supportsMultipleReferenceImages: false,
                defaultSteps: 4,
                minimumDimensionPixels: 64,
                maximumDimensionPixels: 1_024,
                dimensionMultiplePixels: 16)),
            license: nil,
            modelSizeBytes: 4_000);
    }
}
