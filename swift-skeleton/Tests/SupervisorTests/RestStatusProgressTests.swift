import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;

@testable import Supervisor;

/**
 * Serving activity and per-request progress journeys, migrating the
 * progress slice of apps/supervisor/tests/rest_api/application/
 * image_status.rs and the status-document slice of its contracts.rs: the
 * status document reports the ready model with a zeroed serving session
 * and prompt-cache summary, the active request's phase through activity
 * and progress sections, live elapsed time keeps advancing between worker
 * frames, and a completed request restores idle without progress.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class RestStatusProgressTests {

    /// Image activity reports its phase with completed and total steps.
    @Test
    func should_expose_image_generation_progress_with_completed_and_total_steps_when_active() throws {
        let statusJourney: StatusProgressJourney = try StatusProgressJourney.launch();
        defer { statusJourney.dispose() }
        var healthSnapshot: WorkerHealthSnapshot = WorkerHealthSnapshot.readyWithModel(
            modelId: StatusProgressJourney.modelId,
            capabilities: RestChatJourneySupport.readyChatCapabilities());
        healthSnapshot.activity = .imageGeneration;
        healthSnapshot.activeRequestProgress = .imageGeneration(
            phase: .denoising,
            completedSteps: 2,
            totalSteps: 4,
            elapsedMillis: 1_000);
        statusJourney.workerHealthState.publish(healthSnapshot);

        let statusDocument: [String: Any] = try statusJourney.getStatusDocument();

        #expect(statusDocument["activity"] as? String == "image_generation");
        let progressDocument: [String: Any] = try StatusProgressJourney.requireObject(
            statusDocument, field: "progress");
        #expect(progressDocument["phase"] as? String == "denoising");
        #expect(progressDocument["completed_steps"] as? Int == 2);
        #expect(progressDocument["total_steps"] as? Int == 4);
        #expect(progressDocument["elapsed_ms"] as? Int == 1_000);
    }

    /// Prefill progress reports the phase, token counts, live elapsed time,
    /// and the completed chunk observation.
    @Test
    func should_expose_prefill_progress_with_live_elapsed_time() throws {
        let statusJourney: StatusProgressJourney = try StatusProgressJourney.launch();
        defer { statusJourney.dispose() }
        var healthSnapshot: WorkerHealthSnapshot = WorkerHealthSnapshot.readyWithModel(
            modelId: StatusProgressJourney.modelId,
            capabilities: RestChatJourneySupport.readyChatCapabilities());
        healthSnapshot.activity = .promptProcessing;
        healthSnapshot.activeRequestProgress = .prefill(
            promptProcessingPhase: .target,
            processedTokens: 5,
            totalTokens: 10,
            requestStartedAt: Date().addingTimeInterval(-2),
            elapsedMillis: 500,
            completedPrefillChunkTokens: 64);
        statusJourney.workerHealthState.publish(healthSnapshot);

        let statusDocument: [String: Any] = try statusJourney.getStatusDocument();

        #expect(statusDocument["activity"] as? String == "prompt_processing");
        let progressDocument: [String: Any] = try StatusProgressJourney.requireObject(
            statusDocument, field: "progress");
        #expect(progressDocument["phase"] as? String == "target");
        #expect(progressDocument["processed_tokens"] as? Int == 5);
        #expect(progressDocument["total_tokens"] as? Int == 10);
        #expect((progressDocument["elapsed_ms"] as? Int ?? 0) >= 1_000,
            "the elapsed time must keep advancing after the last worker frame");
        #expect(progressDocument["completed_prefill_chunk_tokens"] as? Int == 64);
    }

    /// A ready idle worker answers the full status contract: the ready model
    /// identity and size, no MLX snapshot yet, a zeroed serving session, and
    /// a zeroed persistent prompt-cache summary.
    @Test
    func should_report_ready_status_idle_activity_and_model_id_for_a_ready_worker() throws {
        let statusJourney: StatusProgressJourney = try StatusProgressJourney.launch(discoveredModels: [
            StatusProgressJourney.discoveredReadyModel(modelSizeBytes: 18_420_000_000),
        ]);
        defer { statusJourney.dispose() }
        var healthSnapshot: WorkerHealthSnapshot = WorkerHealthSnapshot.readyWithModel(
            modelId: StatusProgressJourney.modelId,
            capabilities: RestChatJourneySupport.readyChatCapabilities());
        healthSnapshot.effectiveMlxMemoryCeilingBytes = 40_000_000_000;
        statusJourney.workerHealthState.publish(healthSnapshot);

        let statusDocument: [String: Any] = try statusJourney.getStatusDocument();

        #expect(statusDocument["status"] as? String == "ready");
        #expect(statusDocument["activity"] as? String == "idle");
        #expect(statusDocument["ready_model_id"] as? String == StatusProgressJourney.modelId);
        #expect(statusDocument["ready_model_size_bytes"] as? UInt64 == 18_420_000_000);
        #expect(statusDocument["mlx_memory_snapshot"] is NSNull);
        #expect(
            statusDocument["last_request_mlx_active_memory_bytes"] == nil,
            "the per-request memory observation must not appear before any request");
        #expect(
            statusDocument["expert_storage_format"] == nil,
            "the removed storage-format key must stay absent");
        #expect(statusDocument["mlx_memory_ceiling_bytes"] as? UInt64 == 40_000_000_000);
        let servingSessionDocument: [String: Any] = try StatusProgressJourney.requireObject(
            statusDocument, field: "serving_session");
        #expect(servingSessionDocument["completed_request_count"] as? UInt64 == 0);
        #expect(servingSessionDocument["total_prompt_token_count"] as? UInt64 == 0);
        #expect(servingSessionDocument["total_reused_prompt_token_count"] as? UInt64 == 0);
        #expect(servingSessionDocument["average_prefill_tok_per_second"] as? Double == 0.0);
        #expect(servingSessionDocument["average_generation_tok_per_second"] as? Double == 0.0);
        let persistentPromptCacheDocument: [String: Any] = try StatusProgressJourney.requireObject(
            statusDocument, field: "persistent_prompt_cache");
        #expect(persistentPromptCacheDocument["hits"] as? UInt64 == 0);
        #expect(persistentPromptCacheDocument["misses"] as? UInt64 == 0);
        #expect(persistentPromptCacheDocument["tokens_saved"] as? UInt64 == 0);
        #expect(persistentPromptCacheDocument["hit_rate"] as? Double == 0.0);
    }

    /// An actively generating worker reports the generating activity while
    /// staying ready.
    @Test
    func should_report_generating_activity_for_an_active_worker() throws {
        let statusJourney: StatusProgressJourney = try StatusProgressJourney.launch();
        defer { statusJourney.dispose() }
        var healthSnapshot: WorkerHealthSnapshot = WorkerHealthSnapshot.readyWithModel(
            modelId: StatusProgressJourney.modelId,
            capabilities: RestChatJourneySupport.readyChatCapabilities());
        healthSnapshot.activity = .generating;
        statusJourney.workerHealthState.publish(healthSnapshot);

        let statusDocument: [String: Any] = try statusJourney.getStatusDocument();

        #expect(statusDocument["status"] as? String == "ready");
        #expect(statusDocument["activity"] as? String == "generating");
    }

    /// Initial prefill progress must not claim a chunk size before the
    /// engine measures one.
    @Test
    func should_omit_completed_prefill_chunk_tokens_before_the_first_measurement() throws {
        let statusJourney: StatusProgressJourney = try StatusProgressJourney.launch();
        defer { statusJourney.dispose() }
        var healthSnapshot: WorkerHealthSnapshot = WorkerHealthSnapshot.readyWithModel(
            modelId: StatusProgressJourney.modelId,
            capabilities: RestChatJourneySupport.readyChatCapabilities());
        healthSnapshot.activity = .promptProcessing;
        healthSnapshot.activeRequestProgress = .prefill(
            promptProcessingPhase: .target,
            processedTokens: 0,
            totalTokens: 2_200,
            requestStartedAt: Date(),
            elapsedMillis: 0,
            completedPrefillChunkTokens: nil);
        statusJourney.workerHealthState.publish(healthSnapshot);

        let progressDocument: [String: Any] = try StatusProgressJourney.requireObject(
            try statusJourney.getStatusDocument(), field: "progress");

        #expect(progressDocument["phase"] as? String == "target");
        #expect(
            progressDocument["completed_prefill_chunk_tokens"] == nil,
            "initial progress must not claim a chunk size before the engine measures one");
    }

    /// Generation preparation reports the layer topology with both elapsed
    /// clocks, and generation collapses into the shared token shape.
    @Test
    func should_expose_preparation_and_generation_progress_shapes() throws {
        let statusJourney: StatusProgressJourney = try StatusProgressJourney.launch();
        defer { statusJourney.dispose() }
        var preparationSnapshot: WorkerHealthSnapshot = WorkerHealthSnapshot.readyWithModel(
            modelId: StatusProgressJourney.modelId,
            capabilities: RestChatJourneySupport.readyChatCapabilities());
        preparationSnapshot.activity = .generationPreparation;
        preparationSnapshot.activeRequestProgress = .generationPreparation(
            requestStartedAt: Date().addingTimeInterval(-3),
            preparationStartedAt: Date().addingTimeInterval(-1),
            totalLayerCount: 48,
            residentExpertCount: 12,
            residentExpertPayloadBytes: 1_073_741_824);
        statusJourney.workerHealthState.publish(preparationSnapshot);
        let preparationDocument: [String: Any] = try StatusProgressJourney.requireObject(
            try statusJourney.getStatusDocument(), field: "progress");
        #expect(preparationDocument["phase"] as? String == "generation_preparation");
        #expect(preparationDocument["processed_tokens"] as? Int == 0);
        #expect(preparationDocument["total_tokens"] as? Int == 1);
        #expect((preparationDocument["elapsed_ms"] as? Int ?? 0) >= 500);
        #expect((preparationDocument["request_elapsed_ms"] as? Int ?? 0) >= 1_500);
        #expect(preparationDocument["total_layer_count"] as? Int == 48);
        #expect(preparationDocument["resident_expert_count"] as? Int == 12);
        #expect(preparationDocument["resident_expert_payload_bytes"] as? Int == 1_073_741_824);

        var generationSnapshot: WorkerHealthSnapshot = WorkerHealthSnapshot.readyWithModel(
            modelId: StatusProgressJourney.modelId,
            capabilities: RestChatJourneySupport.readyChatCapabilities());
        generationSnapshot.activity = .generating;
        generationSnapshot.activeRequestProgress = .generation(
            generatedTokenCount: 7,
            maximumOutputTokens: 512,
            elapsedMillis: 250);
        statusJourney.workerHealthState.publish(generationSnapshot);
        let generationDocument: [String: Any] = try StatusProgressJourney.requireObject(
            try statusJourney.getStatusDocument(), field: "progress");
        #expect(generationDocument["phase"] as? String == "generation");
        #expect(generationDocument["processed_tokens"] as? Int == 7);
        #expect(generationDocument["total_tokens"] as? Int == 512);
        #expect(generationDocument["elapsed_ms"] as? Int == 250);
    }

    /// A real fixture worker's prefill frame drives the live phase, and the
    /// completion restores idle without progress.
    @Test
    func should_publish_prefill_activity_from_the_generation_event_flow_and_restore_idle() throws {
        let harness: FakeWorkerJourneyHarness = try FakeWorkerJourneyHarness.launch();
        defer { harness.dispose() }
        let generationOutcome: GenerationJourneyOutcome = harness.startGenerationThread(requestId: 9_003);

        try harness.pokePrefillProgress(requestId: 9_003, processedTokens: 5, totalTokens: 10);
        let activityDeadline: Date = Date().addingTimeInterval(5);
        var observedProgress: ActiveRequestProgress?;
        while true {
            let healthSnapshot: WorkerHealthSnapshot = harness.supervisor.workerHealthSnapshot();
            if case let .prefill(_, processedTokens, _, _, _, _)? = healthSnapshot.activeRequestProgress,
               processedTokens == 5 {
                observedProgress = healthSnapshot.activeRequestProgress;
                break;
            }
            if Date() >= activityDeadline {
                Issue.record("the prefill progress never reached health state");
                break;
            }
            Thread.sleep(forTimeInterval: 0.01);
        }
        if case let .prefill(_, processedTokens, totalTokens, _, elapsedMillis, completedChunk)? = observedProgress {
            #expect(processedTokens == 5);
            #expect(totalTokens == 10);
            #expect(elapsedMillis == 500);
            #expect(completedChunk == 64);
        }
        #expect(harness.supervisor.workerHealthSnapshot().activity == .promptProcessing);

        try harness.pokeCompletion(requestId: 9_003);
        let completionDeadline: Date = Date().addingTimeInterval(5);
        while generationOutcome.isSuccessful == false && Date() < completionDeadline {
            Thread.sleep(forTimeInterval: 0.01);
        }
        #expect(generationOutcome.isSuccessful);
        let settledHealth: WorkerHealthSnapshot = harness.supervisor.workerHealthSnapshot();
        #expect(settledHealth.activity == .idle);
        #expect(settledHealth.activeRequestProgress == nil);
    }
}

/// One status progress journey: a serving route table over a health state
/// the journey publishes directly, the in-process equivalent of the Rust
/// contracts' health-snapshot injection.
final class StatusProgressJourney {

    static let modelId: String = "astronomical/status-progress-model";

    let workerHealthState: WorkerHealthState;
    private let routeTable: RestRouteTable;
    private let homeDirectoryUrl: URL;

    private init(
        workerHealthState: WorkerHealthState,
        routeTable: RestRouteTable,
        homeDirectoryUrl: URL
    ) {
        self.workerHealthState = workerHealthState;
        self.routeTable = routeTable;
        self.homeDirectoryUrl = homeDirectoryUrl;
    }

    static func launch(
        discoveredModels: Array<DiscoveryDiscoveredModel> = Array()
    ) throws -> StatusProgressJourney {
        let homeDirectoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("astronomical-status-progress-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(at: homeDirectoryUrl, withIntermediateDirectories: true);
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            FilePath(string: homeDirectoryUrl.path),
            runtimeInstance: AstronomicalRuntimeInstance.development);
        let resolvedRuntimeConfig: ResolvedRuntimeConfig = try RestChatJourneySupport.makeResolvedConfig(
            discoveredModels: discoveredModels);
        let workerHealthState: WorkerHealthState = WorkerHealthState();
        // Discovered models mirror the Rust build_application_with_discovered_models
        // builder: the status document answers the ready model's size from
        // the accepted discovery snapshot the reload transition carries.
        var configReloadContext: RestConfigReloadRouteContext? = nil;
        if discoveredModels.isEmpty == false {
            configReloadContext = RestConfigReloadRouteContext(
                transitionState: ConfigTransitionState(
                    reloadableConfig: resolvedRuntimeConfig,
                    configuredConfigSnapshot: resolvedRuntimeConfig),
                runtimeConfigResolver: ResolvedRuntimeConfigResolver(
                    instancePaths: instancePaths,
                    fallbackWorkerExecutablePath: resolvedRuntimeConfig.workerExecutablePath),
                workerControl: nil,
                workerHealthState: workerHealthState,
                generationActivityIdleProvider: { return true });
        }
        let routeTable: RestRouteTable = RestEndpointRoutes.servingRouteTable(
            resolvedRuntimeConfig: resolvedRuntimeConfig,
            workerHealthState: workerHealthState,
            instancePaths: instancePaths,
            buildIdentity: RestChatJourneySupport.journeyBuildIdentity(),
            configReloadContext: configReloadContext);
        return StatusProgressJourney(
            workerHealthState: workerHealthState,
            routeTable: routeTable,
            homeDirectoryUrl: homeDirectoryUrl);
    }

    /// The discovered-model fixture whose size the status document echoes for
    /// the ready model.
    static func discoveredReadyModel(modelSizeBytes: UInt64) -> DiscoveryDiscoveredModel {
        return DiscoveryDiscoveredModel(
            modelId: StatusProgressJourney.modelId,
            providerModelId: nil,
            modelFamily: .qwen35,
            revision: "status-progress-revision",
            modelDirectory: FilePath(string: "/fictional/models/status-progress-model"),
            capabilities: .chat(DiscoveryChatModelCapabilities(
                contextWindowTokens: 262_144,
                maximumInputTokens: 241_664,
                maximumOutputTokens: 20_480,
                supportsVision: true,
                supportsReasoning: true,
                supportsToolCalls: true)),
            license: nil,
            modelSizeBytes: modelSizeBytes);
    }

    func dispose() -> Void {
        try? FileManager.default.removeItem(atPath: self.homeDirectoryUrl.path);
    }

    func getStatusDocument() throws -> [String: Any] {
        let routeOutcome: RestRouteOutcome = self.routeTable.outcome(method: "GET", path: "/v1/status");
        guard case .handler(let routeHandler) = routeOutcome else {
            throw StatusProgressJourneyFailure.statusRouteMissing;
        }
        let statusResponse: RestHttpResponse = try routeHandler(WorkerReplacementJourney.emptyRequest(
            method: "GET",
            path: "/v1/status"));
        return try ConfigReloadJourney.decodeObject(statusResponse);
    }

    static func requireObject(
        _ document: [String: Any],
        field: String
    ) throws -> [String: Any] {
        guard let nestedObject: [String: Any] = document[field] as? [String: Any] else {
            throw StatusProgressJourneyFailure.objectMissing(field);
        }
        return nestedObject;
    }
}

/// Typed failures of the status progress journeys.
enum StatusProgressJourneyFailure: Error {

    case statusRouteMissing;
    case objectMissing(String);
}
