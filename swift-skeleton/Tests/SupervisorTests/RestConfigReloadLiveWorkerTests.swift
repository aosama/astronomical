import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;

@testable import Supervisor;

/**
 * Config-reload and memory-ceiling journeys that drive a real fixture
 * worker process, migrating apps/supervisor/tests/rest_api/config_reload/
 * mixed_reload_configuration_generation.rs and the live-worker
 * maximum_mlx_memory.rs journeys: a mixed reload applies only the derived
 * memory configuration generation and the next model swap accepts it,
 * unrelated pending config changes reject a memory update, and a queued
 * reload memory setting rejected after a generation rolls the live state
 * back to the prior configuration.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class RestConfigReloadLiveWorkerTests {

    private static let mixedModelId: String = "astronomical/mixed-memory-reload-model";
    private static let delayedCompletionModelId: String = "astronomical/delayed-completion-model";
    static let romeoAndJulietPrompt: String =
        "You are a concise literature assistant. In one sentence, name the play "
        + "these lines come from: \"O Romeo, Romeo, wherefore art thou Romeo?\"";

    /// A reload that only changes the memory ceiling plus restart-required
    /// fields must apply the derived memory-only generation to the live
    /// worker, and the next model swap must acknowledge it.
    @Test
    func should_load_a_model_after_a_mixed_reload_applies_only_the_memory_configuration_generation() throws {
        let journey: ReloadLiveWorkerJourney = try ReloadLiveWorkerJourney.launch(
            startupConfiguration: RestChatJourneySupport.makeResolvedConfig().workerStartupConfiguration(),
            modelPolicyCatalog: [
                RestConfigReloadLiveWorkerTests.mixedModelId:
                    ReloadLiveWorkerJourney.autoregressiveModelPolicy(
                        modelId: RestConfigReloadLiveWorkerTests.mixedModelId),
            ]);
        defer { journey.dispose() }
        ConfigReloadJourney.writeConfigFile(
            journey.homeDirectoryUrl,
            configuredFieldsJson: "{\"runtime\":{\"model_directories\":[],"
                + "\"maximum_mlx_memory_gb\":32},\"diagnostics\":{\"log_level\":\"info\"}}");
        let memoryEffectiveGeneration: String = ResolvedConfigurationGeneration.deriveMemoryOnlyTransition(
            priorResolvedGeneration: journey.transitionState.currentReloadableConfig()
                .configurationGeneration,
            maximumMlxMemoryBytes: 32_000_000_000);
        journey.armMemoryAcknowledgement(
            effectiveMlxMemoryCeilingBytes: 32_000_000_000,
            configurationGeneration: memoryEffectiveGeneration);

        let reloadResponse: RestHttpResponse = try journey.postConfigReload();

        #expect(reloadResponse.statusCode == 200);
        let reloadDocument: [String: Any] = try ConfigReloadJourney.decodeObject(reloadResponse);
        #expect(reloadDocument["status"] as? String == "restart_required");
        try WorkerReplacementJourney.waitForEffectiveGeneration(
            supervisor: journey.supervisor,
            expectedGeneration: memoryEffectiveGeneration);
        journey.armModelSwap(RestConfigReloadLiveWorkerTests.mixedModelId);
        let generationOutcome: ReloadLiveWorkerJourney.GenerationOutcome =
            journey.startGenerationOnThread(
                requestId: 9_002,
                modelId: RestConfigReloadLiveWorkerTests.mixedModelId);
        try ReloadLiveWorkerJourney.awaitReadyModel(
            supervisor: journey.supervisor,
            expectedModelId: RestConfigReloadLiveWorkerTests.mixedModelId);
        journey.pokeCompletion(requestId: 9_002);
        try generationOutcome.join(within: 5);
        let completionReason: ChatGenerationCompletionReason? = try generationOutcome
            .requireCompletionReason();
        #expect(completionReason == .endOfSequence);
    }

    /// A memory update must be rejected while unrelated configuration
    /// changes are pending, leaving the file and the live config untouched.
    @Test
    func should_require_full_reload_when_other_configuration_changes_are_pending() throws {
        let journey: ReloadLiveWorkerJourney = try ReloadLiveWorkerJourney.launch(
            startupConfiguration: nil,
            modelPolicyCatalog: Dictionary());
        defer { journey.dispose() }
        ConfigReloadJourney.writeConfigFile(
            journey.homeDirectoryUrl,
            configuredFieldsJson: "{\"diagnostics\":{\"log_level\":\"info\"}}");

        let memoryResponse: RestHttpResponse = try journey.putMaximumMlxMemory(
            maximumMlxMemoryGb: 32);

        #expect(memoryResponse.statusCode == 409);
        let configFileText: String = try String(
            contentsOf: journey.configFileUrl,
            encoding: .utf8);
        #expect(configFileText.contains("info"));
        #expect(journey.transitionState.currentReloadableConfig().maximumMlxMemoryBytes == nil);
    }

    /// A reload memory setting queued behind an active generation and then
    /// rejected by the worker must roll the live configuration back to the
    /// prior resolved generation while keeping the persisted candidate.
    @Test
    func should_rollback_live_state_when_a_reloaded_memory_setting_is_rejected_after_queueing() throws {
        let journey: ReloadLiveWorkerJourney = try ReloadLiveWorkerJourney.launch(
            startupConfiguration: nil,
            modelPolicyCatalog: [
                RestConfigReloadLiveWorkerTests.delayedCompletionModelId:
                    ReloadLiveWorkerJourney.autoregressiveModelPolicy(
                        modelId: RestConfigReloadLiveWorkerTests.delayedCompletionModelId),
            ]);
        defer { journey.dispose() }
        let initialGeneration: String = journey.transitionState.currentReloadableConfig()
            .configurationGeneration;
        journey.armModelSwap(RestConfigReloadLiveWorkerTests.delayedCompletionModelId);
        let generationOutcome: ReloadLiveWorkerJourney.GenerationOutcome =
            journey.startGenerationOnThread(
                requestId: 9_001,
                modelId: RestConfigReloadLiveWorkerTests.delayedCompletionModelId);
        try ReloadLiveWorkerJourney.awaitReadyModel(
            supervisor: journey.supervisor,
            expectedModelId: RestConfigReloadLiveWorkerTests.delayedCompletionModelId);
        ConfigReloadJourney.writeConfigFile(
            journey.homeDirectoryUrl,
            configuredFieldsJson: "{\"runtime\":{\"model_directories\":[],"
                + "\"maximum_mlx_memory_gb\":31}}");

        let reloadResponse: RestHttpResponse = try journey.postConfigReload();
        #expect(reloadResponse.statusCode == 200);
        let conflictingMemoryResponse: RestHttpResponse = try journey.putMaximumMlxMemory(
            maximumMlxMemoryGb: 32);
        #expect(conflictingMemoryResponse.statusCode == 409);

        journey.armMemoryRejection(requestedMlxMemoryCeilingBytes: 31_000_000_000);
        journey.pokeCompletion(requestId: 9_001);
        try generationOutcome.join(within: 5);
        #expect(try generationOutcome.requireCompletionReason() == .endOfSequence);
        let rollbackDeadline: Date = Date().addingTimeInterval(5);
        while journey.transitionState.currentReloadableConfig().configurationGeneration
            != initialGeneration {
            if Date() >= rollbackDeadline {
                Issue.record(
                    "the reloadable generation never rolled back to \(initialGeneration)");
                break;
            }
            Thread.sleep(forTimeInterval: 0.025);
        }
        let configFileText: String = try String(
            contentsOf: journey.configFileUrl,
            encoding: .utf8);
        #expect(configFileText.contains("31"));
    }
}

/// One live-worker reload journey: an executable fixture worker with
/// marker-armed memory, swap, and completion acknowledgements, a resolver
/// config home, and the reload plus memory-limit routes wired to the real
/// supervisor.
final class ReloadLiveWorkerJourney {

    let supervisor: WorkerSupervisor;
    let transitionState: ConfigTransitionState;
    let homeDirectoryUrl: URL;
    let configFileUrl: URL;
    private let routeTable: RestRouteTable;
    private let controlDirectoryPath: String;

    private init(
        supervisor: WorkerSupervisor,
        transitionState: ConfigTransitionState,
        homeDirectoryUrl: URL,
        configFileUrl: URL,
        routeTable: RestRouteTable,
        controlDirectoryPath: String
    ) {
        self.supervisor = supervisor;
        self.transitionState = transitionState;
        self.homeDirectoryUrl = homeDirectoryUrl;
        self.configFileUrl = configFileUrl;
        self.routeTable = routeTable;
        self.controlDirectoryPath = controlDirectoryPath;
    }

    /// Launches the fixture supervisor. A `nil` startup configuration mirrors
    /// Rust's WorkerHandle::launch: the fixture reports idle without any
    /// runtime-policy acknowledgement.
    static func launch(
        startupConfiguration: WorkerStartupConfiguration?,
        modelPolicyCatalog: Dictionary<String, RuntimeModelPolicy>
    ) throws -> ReloadLiveWorkerJourney {
        let homeDirectoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("astronomical-reload-live-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(at: homeDirectoryUrl, withIntermediateDirectories: true);
        ConfigReloadJourney.writeConfigFile(homeDirectoryUrl, configuredFieldsJson: "{}");
        let configFileUrl: URL = homeDirectoryUrl
            .appendingPathComponent(".astronomical-dev/config.json");
        let controlDirectoryUrl: URL = homeDirectoryUrl.appendingPathComponent("worker-control", isDirectory: true);
        try FileManager.default.createDirectory(at: controlDirectoryUrl, withIntermediateDirectories: true);
        let fixtureWorkerPath: String = try ReloadLiveWorkerJourney.writeFixtureScript(
            homeDirectoryUrl: homeDirectoryUrl,
            controlDirectoryPath: controlDirectoryUrl.path,
            startupConfiguration: startupConfiguration);
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            FilePath(string: homeDirectoryUrl.path),
            runtimeInstance: AstronomicalRuntimeInstance.development);
        let resolver: ResolvedRuntimeConfigResolver = ResolvedRuntimeConfigResolver(
            instancePaths: instancePaths,
            fallbackWorkerExecutablePath: FilePath(string: fixtureWorkerPath));
        let initialResolvedConfig: ResolvedRuntimeConfig = try resolver.load();
        let transitionState: ConfigTransitionState = ConfigTransitionState(
            reloadableConfig: initialResolvedConfig,
            configuredConfigSnapshot: initialResolvedConfig);
        let supervisor: WorkerSupervisor = try WorkerSupervisor.launch(
            workerExecutablePath: fixtureWorkerPath,
            workerArguments: Array<String>(),
            workerStartupConfiguration: startupConfiguration,
            modelPolicyCatalog: modelPolicyCatalog,
            modelLoadTimeout: 2);
        try ReloadLiveWorkerJourney.awaitWorkerReady(supervisor: supervisor);
        let routeTable: RestRouteTable = RestEndpointRoutes.servingRouteTable(
            resolvedRuntimeConfig: initialResolvedConfig,
            workerHealthState: supervisor.ownedHealthState(),
            instancePaths: instancePaths,
            buildIdentity: RestChatJourneySupport.journeyBuildIdentity(),
            memoryContext: RestMaximumMlxMemoryRouteContext(
                workerControl: supervisor,
                runtimeConfigResolver: resolver,
                transitionState: transitionState),
            configReloadContext: RestConfigReloadRouteContext(
                transitionState: transitionState,
                runtimeConfigResolver: resolver,
                workerControl: supervisor,
                workerHealthState: supervisor.ownedHealthState(),
                generationActivityIdleProvider: { return true }));
        return ReloadLiveWorkerJourney(
            supervisor: supervisor,
            transitionState: transitionState,
            homeDirectoryUrl: homeDirectoryUrl,
            configFileUrl: configFileUrl,
            routeTable: routeTable,
            controlDirectoryPath: controlDirectoryUrl.path);
    }

    func dispose() -> Void {
        _ = try? self.supervisor.shutdown();
        try? FileManager.default.removeItem(atPath: self.homeDirectoryUrl.path);
    }

    // MARK: Route calls

    func postConfigReload() throws -> RestHttpResponse {
        let routeOutcome: RestRouteOutcome = self.routeTable.outcome(
            method: RestConfigReloadEndpoint.routeMethod,
            path: RestConfigReloadEndpoint.routePath);
        guard case .handler(let routeHandler) = routeOutcome else {
            throw ReloadLiveWorkerJourneyFailure.routeMissing("config reload");
        }
        return try routeHandler(WorkerReplacementJourney.emptyRequest(
            method: RestConfigReloadEndpoint.routeMethod,
            path: RestConfigReloadEndpoint.routePath));
    }

    func putMaximumMlxMemory(maximumMlxMemoryGb: UInt64) throws -> RestHttpResponse {
        let memoryRequest: RestHttpRequest = RestHttpRequest(
            method: RestMaximumMlxMemoryEndpoint.routeMethod,
            path: RestMaximumMlxMemoryEndpoint.routePath,
            requestTarget: RestMaximumMlxMemoryEndpoint.routePath,
            headersByLowercasedName: ["content-type": "application/json"],
            bodyBytes: Data(
                "{\"maximum_mlx_memory_gb\":\(maximumMlxMemoryGb)}".utf8));
        let routeOutcome: RestRouteOutcome = self.routeTable.outcome(
            method: RestMaximumMlxMemoryEndpoint.routeMethod,
            path: RestMaximumMlxMemoryEndpoint.routePath);
        guard case .handler(let routeHandler) = routeOutcome else {
            throw ReloadLiveWorkerJourneyFailure.routeMissing("maximum mlx memory");
        }
        return try routeHandler(memoryRequest);
    }

    // MARK: Fixture steering

    /// Arms the memory-ceiling acknowledgement the endpoint's live update
    /// waits on, recording the requested generation for the next swap ack.
    func armMemoryAcknowledgement(
        effectiveMlxMemoryCeilingBytes: UInt64,
        configurationGeneration: String
    ) -> Void {
        let markerContent: String =
            "\(effectiveMlxMemoryCeilingBytes)|\(configurationGeneration)";
        try? markerContent.write(
            toFile: self.controlDirectoryPath + "/memory_ack",
            atomically: true,
            encoding: .utf8);
    }

    /// Arms the next memory-ceiling command to answer with a rejection.
    func armMemoryRejection(requestedMlxMemoryCeilingBytes: UInt64) -> Void {
        try? String(requestedMlxMemoryCeilingBytes).write(
            toFile: self.controlDirectoryPath + "/memory_reject",
            atomically: true,
            encoding: .utf8);
    }

    /// Arms the next model-swap acknowledgement for the given model.
    func armModelSwap(_ modelId: String) -> Void {
        try? modelId.write(
            toFile: self.controlDirectoryPath + "/swap_model",
            atomically: true,
            encoding: .utf8);
    }

    /// Completes the pending generation with the given request id.
    func pokeCompletion(requestId: UInt64) -> Void {
        try? String(requestId).write(
            toFile: self.controlDirectoryPath + "/complete_request",
            atomically: true,
            encoding: .utf8);
    }

    // MARK: Generation journeys

    func startGenerationOnThread(
        requestId: UInt64,
        modelId: String
    ) -> ReloadLiveWorkerJourney.GenerationOutcome {
        let generationOutcome: ReloadLiveWorkerJourney.GenerationOutcome =
            ReloadLiveWorkerJourney.GenerationOutcome();
        let supervisor: WorkerSupervisor = self.supervisor;
        let romeoAndJulietPrompt: String = RestConfigReloadLiveWorkerTests.romeoAndJulietPrompt;
        let generationThread: Thread = Thread {
            do {
                let streamEvents: Array<ChatGenerationStreamEvent> = try supervisor
                    .startChatGeneration(ChatGenerationCommand(
                        requestId: RequestId(rawRequestId: requestId),
                        model: modelId,
                        messages: [.user(content: romeoAndJulietPrompt, images: [])],
                        tools: [],
                        toolChoice: .none,
                        settings: ChatGenerationSettings(
                            maxOutputTokens: 1,
                            temperatureThousandths: nil,
                            topPThousandths: nil,
                            seed: nil,
                            thinkingBudget: nil),
                        qwenThinkingChannelSeed: nil,
                        structuredGeneration: nil));
                generationOutcome.record(streamEvents: streamEvents);
            } catch {
                generationOutcome.record(error: error);
            }
        };
        generationThread.name = "reload-live-generation-\(requestId)";
        generationOutcome.workerThread = generationThread;
        generationThread.start();
        return generationOutcome;
    }

    // MARK: Bounded waits

    static func awaitReadyModel(
        supervisor: WorkerSupervisor,
        expectedModelId: String
    ) throws -> Void {
        let readinessDeadline: Date = Date().addingTimeInterval(2);
        while supervisor.workerHealthSnapshot().readyModelId != expectedModelId {
            if Date() >= readinessDeadline {
                throw ReloadLiveWorkerJourneyFailure.workerAcknowledgementTimedOut;
            }
            Thread.sleep(forTimeInterval: 0.01);
        }
    }

    private static func awaitWorkerReady(supervisor: WorkerSupervisor) throws -> Void {
        let readinessDeadline: Date = Date().addingTimeInterval(2);
        while supervisor.workerHealthSnapshot().status != .ready {
            if Date() >= readinessDeadline {
                throw ReloadLiveWorkerJourneyFailure.workerAcknowledgementTimedOut;
            }
            Thread.sleep(forTimeInterval: 0.01);
        }
    }

    // MARK: Fixture script

    private static func writeFixtureScript(
        homeDirectoryUrl: URL,
        controlDirectoryPath: String,
        startupConfiguration: WorkerStartupConfiguration?
    ) throws -> String {
        let initialGeneration: String = startupConfiguration?.configurationGeneration ?? "";
        let runtimePolicyPayload: String =
            "{\"kind\":\"runtime_feature_configuration_applied\","
            + "\"worker_runtime_feature_configuration\":{\"configuration_generation\":\"__GENERATION__\","
            + "\"persistent_prompt_cache_enabled\":true,\"prompt_cache_maximum_size_bytes\":1073741824,"
            + "\"loaded_model\":{\"kind\":\"autoregressive\",\"configuration\":{\"model_id\":"
            + "\"__MODEL_ID__\",\"maximum_context_tokens\":2048,\"maximum_output_tokens\":128,"
            + "\"chunking\":{\"fixed_prompt_processing_chunk_size_tokens\":256,"
            + "\"fixed_ssd_streaming_prompt_processing_chunk_size_tokens\":2048,"
            + "\"full_attention_key_value_growth_tokens\":256,"
            + "\"prefill_graph_submission_layer_interval\":0,"
            + "\"experimental_ssd_paging_prefill_graph_submission_layer_interval\":1,"
            + "\"experimental_ssd_paging_generation_graph_submission_layer_interval\":3,"
            + "\"prompt_cache_block_tokens\":null,\"prompt_cache_common_prefix_stride_blocks\":4,"
            + "\"experimental_decode_stage_attribution_enabled\":false,"
            + "\"experimental_quantized_kv_cache_enabled\":false,"
            + "\"experimental_fused_moe_decode_enabled\":false}}}}}";
        try runtimePolicyPayload.write(
            toFile: controlDirectoryPath + "/swap_runtime_policy",
            atomically: true,
            encoding: .utf8);
        let idlePayload: String =
            "{\"kind\":\"idle\",\"machine_mlx_memory_ceiling_bytes\":40000000000,"
            + "\"effective_mlx_memory_ceiling_bytes\":40000000000,\"minimum_mlx_memory_ceiling_bytes\":1}";
        let scriptText: String = "#!/bin/bash\n"
            + FakeWorkerEventEmitter.frameEmitterFunction()
            + FakeWorkerEventEmitter.emitLine(payload: idlePayload)
            + (startupConfiguration == nil
                ? ""
                : FakeWorkerEventEmitter.emitLine(
                    payload: "{\"kind\":\"runtime_feature_configuration_applied\","
                        + "\"worker_runtime_feature_configuration\":{\"configuration_generation\":"
                        + "\"\(initialGeneration)\",\"persistent_prompt_cache_enabled\":true,"
                        + "\"prompt_cache_maximum_size_bytes\":1073741824,\"loaded_model\":null}}"))
            + "printf '%s' \"\(initialGeneration)\" > \"\(controlDirectoryPath)/current_generation\"\n"
            + "while true; do\n"
            + "  if [ -f \"\(controlDirectoryPath)/memory_ack\" ]; then\n"
            + "    memory_bytes=$(cut -d'|' -f1 \"\(controlDirectoryPath)/memory_ack\")\n"
            + "    memory_generation=$(cut -d'|' -f2 \"\(controlDirectoryPath)/memory_ack\")\n"
            + "    rm -f \"\(controlDirectoryPath)/memory_ack\"\n"
            + "    emit_frame '{\"kind\":\"mlx_memory_limit_changed\","
            + "\"effective_mlx_memory_ceiling_bytes\":'\"$memory_bytes\"',"
            + "\"minimum_mlx_memory_ceiling_bytes\":1,\"expert_memory_mode\":\"resident\","
            + "\"mlx_memory_snapshot\":null,\"expert_residency\":null}'\n"
            + "    printf '%s' \"$memory_generation\" > \"\(controlDirectoryPath)/current_generation\"\n"
            + "  fi\n"
            + "  if [ -f \"\(controlDirectoryPath)/memory_reject\" ]; then\n"
            + "    rejected_bytes=$(cat \"\(controlDirectoryPath)/memory_reject\")\n"
            + "    rm -f \"\(controlDirectoryPath)/memory_reject\"\n"
            + "    emit_frame '{\"kind\":\"mlx_memory_limit_rejected\","
            + "\"requested_mlx_memory_ceiling_bytes\":'\"$rejected_bytes\"',"
            + "\"minimum_mlx_memory_ceiling_bytes\":1,\"machine_mlx_memory_ceiling_bytes\":40000000000,"
            + "\"reason\":\"fixture rejected the requested limit\"}'\n"
            + "  fi\n"
            + "  if [ -f \"\(controlDirectoryPath)/swap_model\" ]; then\n"
            + "    swapped_model_id=$(cat \"\(controlDirectoryPath)/swap_model\")\n"
            + "    rm -f \"\(controlDirectoryPath)/swap_model\"\n"
            + "    emit_frame '{\"kind\":\"model_swapped\",\"model_id\":\"'\"$swapped_model_id\"'\","
            + "\"capabilities\":{\"chat\":{\"supports_reasoning\":true,\"supports_tool_calls\":true,"
            + "\"has_vision\":false,\"max_input_tokens\":2047,\"max_output_tokens\":128,"
            + "\"context_window\":2048},\"image_generation\":null,\"embeddings\":null},"
            + "\"expert_memory_mode\":\"resident\",\"minimum_mlx_memory_ceiling_bytes\":3000000000}'\n"
            + "    current_generation=$(cat \"\(controlDirectoryPath)/current_generation\")\n"
            + "    if [ -n \"$current_generation\" ]; then\n"
            + "      emit_frame \"$(sed -e \"s|__GENERATION__|$current_generation|\" "
            + "-e \"s|__MODEL_ID__|$swapped_model_id|\" "
            + "\"\(controlDirectoryPath)/swap_runtime_policy\")\"\n"
            + "    fi\n"
            + "    while [ ! -f \"\(controlDirectoryPath)/complete_request\" ]; do sleep 0.02; done\n"
            + "    completion_request_id=$(cat \"\(controlDirectoryPath)/complete_request\")\n"
            + "    rm -f \"\(controlDirectoryPath)/complete_request\"\n"
            + "    emit_frame '{\"kind\":\"completed\",\"request_id\":'\"$completion_request_id\"',"
            + "\"prompt_token_count\":1,\"generated_token_count\":1,\"reasoning_token_count\":0,"
            + "\"cached_token_count\":0,\"persistent_prompt_cache_diagnostics\":null,"
            + "\"reason\":\"end_of_sequence\"}'\n"
            + "  fi\n"
            + "  sleep 0.02\n"
            + "done\n";
        let fixturePath: String = homeDirectoryUrl.appendingPathComponent("fixture-worker").path;
        try scriptText.write(toFile: fixturePath, atomically: true, encoding: .utf8);
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: fixturePath);
        return fixturePath;
    }

    /// The autoregressive policy the fixture worker acknowledges on swap.
    static func autoregressiveModelPolicy(modelId: String) -> RuntimeModelPolicy {
        return RuntimeModelPolicy(
            modelDirectory: FilePath(string: "/fictional/models/\(modelId)"),
            generationDefaults: RuntimeModelGenerationDefaults(
                maximumOutputTokens: 128,
                configuredMaximumOutputTokens: nil,
                temperatureThousandths: nil,
                topPThousandths: nil),
            configuredMaximumContextTokens: nil,
            defaultMaximumContextTokens: 2_048,
            configuredChunkingFields: ConfiguredChunkingFields.inactive(),
            workerModelConfiguration: WorkerModelConfiguration.autoregressive(
                WorkerAutoregressiveModelConfiguration(
                    modelId: modelId,
                    maximumContextTokens: 2_048,
                    maximumOutputTokens: 128,
                    chunking: WorkerChunkingConfiguration(
                        fixedPromptProcessingChunkSizeTokens: 256,
                        fixedSsdStreamingPromptProcessingChunkSizeTokens: 2_048,
                        fullAttentionKeyValueGrowthTokens: 256,
                        prefillGraphSubmissionLayerInterval: 0,
                        experimentalSsdPagingPrefillGraphSubmissionLayerInterval: 1,
                        experimentalSsdPagingGenerationGraphSubmissionLayerInterval: 3,
                        promptCacheBlockTokens: nil,
                        promptCacheCommonPrefixStrideBlocks: 4,
                        experimentalDecodeStageAttributionEnabled: false,
                        experimentalQuantizedKvCacheEnabled: false,
                        experimentalFusedMoeDecodeEnabled: false))));
    }
}

/// Typed failures of the live-worker reload journeys.
enum ReloadLiveWorkerJourneyFailure: Error {

    case routeMissing(String);
    case workerAcknowledgementTimedOut;
}

extension ReloadLiveWorkerJourney {

    /// One in-flight generation executed on its own thread, with its
    /// terminal outcome captured for the journey thread to inspect.
    final class GenerationOutcome: @unchecked Sendable {

        var workerThread: Thread = Thread();
        private let outcomeLock: NSLock;
        private var observedStreamEvents: Array<ChatGenerationStreamEvent>?;
        private var observedError: Error?;

        init() {
            self.outcomeLock = NSLock();
        }

        func record(streamEvents: Array<ChatGenerationStreamEvent>) -> Void {
            self.outcomeLock.lock();
            self.observedStreamEvents = streamEvents;
            self.outcomeLock.unlock();
        }

        func record(error: Error) -> Void {
            self.outcomeLock.lock();
            self.observedError = error;
            self.outcomeLock.unlock();
        }

        /// Bounded join: the generation must finish inside the window.
        func join(within timeoutSeconds: TimeInterval) throws -> Void {
            let joinDeadline: Date = Date().addingTimeInterval(timeoutSeconds);
            while Date() < joinDeadline {
                self.outcomeLock.lock();
                let hasFinished: Bool =
                    self.observedStreamEvents != nil || self.observedError != nil;
                self.outcomeLock.unlock();
                if hasFinished {
                    return;
                }
                Thread.sleep(forTimeInterval: 0.01);
            }
            throw ReloadLiveWorkerJourneyFailure.workerAcknowledgementTimedOut;
        }

        func requireCompletionReason() throws -> ChatGenerationCompletionReason {
            self.outcomeLock.lock();
            defer { self.outcomeLock.unlock(); }
            if let observedError = observedError {
                throw observedError;
            }
            guard let streamEvents = observedStreamEvents,
                  case let .completed(_, _, _, _, completionReason) = streamEvents.first else {
                throw ReloadLiveWorkerJourneyFailure.workerAcknowledgementTimedOut;
            }
            return completionReason;
        }
    }
}
