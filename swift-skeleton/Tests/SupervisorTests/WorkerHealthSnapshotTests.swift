import Foundation

import Testing

import AstronomicalConfig
import IpcProtocol
import JourneyCategories

@testable import Supervisor

/**
 * Hermetic coverage for the worker health snapshot, migrating the journeys
 * of apps/supervisor/tests/hermetic/worker_health_snapshot.rs: the ready
 * identity publishes from the worker's readiness event, a pending
 * prompt-cache clear survives resident-model replacement, per-measurement
 * throughput averages count only requests that reported them, and the
 * serving session carries across a replacement alongside the MLX ceilings.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class WorkerHealthSnapshotTests {

    @Test
    func should_publish_the_ready_model_identity_from_the_worker_readiness_event() throws {
        let scriptedWorker: String = FakeWorkerEventEmitter.frameEmitterFunction()
            + FakeWorkerEventEmitter.emitLine(payload: WorkerHealthSnapshotTests.chatAndImageReadyEventPayload())
            + "read _\n"
            + "exit 0\n";
        let supervisor: WorkerSupervisor = try WorkerSupervisor.launch(
            workerExecutablePath: "/bin/bash",
            workerArguments: ["-c", scriptedWorker],
            workerStartupConfiguration: nil,
            modelPolicyCatalog: [:],
            modelLoadTimeout: 10);
        defer { _ = try? supervisor.shutdown() }

        let healthDeadline: Date = Date().addingTimeInterval(2);
        var workerHealthSnapshot: WorkerHealthSnapshot = supervisor.workerHealthSnapshot();
        while workerHealthSnapshot.status != .ready {
            workerHealthSnapshot = supervisor.workerHealthSnapshot();
            if Date() >= healthDeadline {
                Issue.record("worker health did not become ready; last status \(workerHealthSnapshot.status)");
                return;
            }
            Thread.sleep(forTimeInterval: 0.025);
        }

        #expect(workerHealthSnapshot.readyModelId == "astronomical/test-worker");
        let readyCapabilities: WorkerModelCapabilities = workerHealthSnapshot.readyModelCapabilities!;
        let chatCapabilities: ChatModelCapabilities = readyCapabilities.chat!;
        #expect(chatCapabilities.supportsReasoning);
        #expect(chatCapabilities.supportsToolCalls);
        #expect(chatCapabilities.hasVision);
        #expect(chatCapabilities.maxInputTokens == 241_664);
        #expect(chatCapabilities.maxOutputTokens == 20_480);
        #expect(chatCapabilities.contextWindow == 262_144);
        let imageCapabilities: ImageGenerationCapabilities = readyCapabilities.imageGeneration!;
        #expect(imageCapabilities.minimumWidthPixels == 64);
        #expect(imageCapabilities.maximumWidthPixels == 1_024);
        #expect(imageCapabilities.minimumHeightPixels == 64);
        #expect(imageCapabilities.maximumHeightPixels == 1_024);
        #expect(imageCapabilities.dimensionMultiplePixels == 16);
        #expect(imageCapabilities.maximumSteps == 4);
        #expect(imageCapabilities.maximumGuidanceThousandths == 1_000);
        #expect(imageCapabilities.outputMimeTypes == ["image/png"]);
    }

    @Test
    func should_preserve_a_pending_cache_clear_across_model_replacement() throws {
        let modelCapabilities: ChatModelCapabilities = ChatModelCapabilities(
            supportsReasoning: true,
            supportsToolCalls: true,
            hasVision: false,
            maxInputTokens: 1_024,
            maxOutputTokens: 128,
            contextWindow: 2_048);
        var previousHealthSnapshot: WorkerHealthSnapshot = WorkerHealthSnapshot.readyWithModel(
            modelId: "example/old-model",
            capabilities: WorkerModelCapabilities.from(chatCapabilities: modelCapabilities));
        previousHealthSnapshot.pendingPromptCacheClear = PendingPromptCacheClear(
            modelId: "example/cached-model");

        let replacementHealthSnapshot: WorkerHealthSnapshot = WorkerHealthSnapshot.readyWithReplacementModel(
            modelId: "example/new-model",
            capabilities: WorkerModelCapabilities.from(chatCapabilities: modelCapabilities),
            minimumMlxMemoryCeilingBytes: 1,
            previousHealthSnapshot: previousHealthSnapshot);

        #expect(
            replacementHealthSnapshot.pendingPromptCacheClear == previousHealthSnapshot.pendingPromptCacheClear,
            "a replacement must not lose the queued prompt-cache clear");
    }

    @Test
    func should_average_only_requests_that_report_each_throughput_measurement() throws {
        var servingSessionSnapshot: ServingSessionSnapshot = ServingSessionSnapshot.empty();

        servingSessionSnapshot.recordCompletedRequest(
            promptTokenCount: 1_000,
            cachedTokenCount: 100,
            prefillTokPerSecond: 10.0,
            generationTokPerSecond: nil);
        servingSessionSnapshot.recordCompletedRequest(
            promptTokenCount: 2_000,
            cachedTokenCount: 200,
            prefillTokPerSecond: nil,
            generationTokPerSecond: 20.0);
        servingSessionSnapshot.recordPromptWorkReuse(WorkerPromptWorkReuse(
            targetEligibleTokenCount: 10_000,
            targetRestoredTokenCount: 8_000));
        servingSessionSnapshot.recordCompletedRequest(
            promptTokenCount: 3_000,
            cachedTokenCount: 300,
            prefillTokPerSecond: 30.0,
            generationTokPerSecond: 40.0);

        #expect(servingSessionSnapshot.completedRequestCount == 3);
        #expect(servingSessionSnapshot.totalPromptTokenCount == 6_000);
        #expect(servingSessionSnapshot.totalReusedPromptTokenCount == 600);
        #expect(servingSessionSnapshot.averagePrefillTokPerSecond == 20.0);
        #expect(servingSessionSnapshot.averageGenerationTokPerSecond == 30.0);
        #expect(servingSessionSnapshot.targetPromptWorkTokenCount == 10_000);
        #expect(servingSessionSnapshot.targetReusedPromptWorkTokenCount == 8_000);
    }

    @Test
    func should_preserve_the_serving_session_when_the_resident_model_is_replaced() throws {
        let modelCapabilities: ChatModelCapabilities = ChatModelCapabilities(
            supportsReasoning: true,
            supportsToolCalls: true,
            hasVision: false,
            maxInputTokens: 100,
            maxOutputTokens: 20,
            contextWindow: 120);
        var previousHealthSnapshot: WorkerHealthSnapshot = WorkerHealthSnapshot.readyWithModel(
            modelId: "first-model",
            capabilities: WorkerModelCapabilities.from(chatCapabilities: modelCapabilities));
        previousHealthSnapshot.machineMlxMemoryCeilingBytes = 40_000;
        previousHealthSnapshot.servingSession.recordCompletedRequest(
            promptTokenCount: 1_000,
            cachedTokenCount: 750,
            prefillTokPerSecond: 10.0,
            generationTokPerSecond: 20.0);

        let replacementHealthSnapshot: WorkerHealthSnapshot = WorkerHealthSnapshot.readyWithReplacementModel(
            modelId: "second-model",
            capabilities: WorkerModelCapabilities.from(chatCapabilities: modelCapabilities),
            minimumMlxMemoryCeilingBytes: 3_000,
            previousHealthSnapshot: previousHealthSnapshot);

        #expect(replacementHealthSnapshot.machineMlxMemoryCeilingBytes == 40_000);
        #expect(replacementHealthSnapshot.minimumMlxMemoryCeilingBytes == 3_000);
        #expect(
            replacementHealthSnapshot.servingSession == previousHealthSnapshot.servingSession,
            "daemon-session totals must survive a resident-model replacement");
    }

    // MARK: Journey fixtures

    private static func chatAndImageReadyEventPayload() -> String {
        return "{\"kind\":\"ready\",\"model_id\":\"astronomical/test-worker\","
            + "\"capabilities\":{\"chat\":{\"supports_reasoning\":true,\"supports_tool_calls\":true,"
            + "\"has_vision\":true,\"max_input_tokens\":241664,\"max_output_tokens\":20480,"
            + "\"context_window\":262144},"
            + "\"image_generation\":{\"minimum_width_pixels\":64,\"maximum_width_pixels\":1024,"
            + "\"minimum_height_pixels\":64,\"maximum_height_pixels\":1024,"
            + "\"dimension_multiple_pixels\":16,\"maximum_steps\":4,"
            + "\"maximum_guidance_thousandths\":1000,\"output_mime_types\":[\"image/png\"]},"
            + "\"embeddings\":null}}";
    }
}
