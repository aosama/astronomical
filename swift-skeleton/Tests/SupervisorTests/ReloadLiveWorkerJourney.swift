import Foundation;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;

@testable import Supervisor;

/// One live-worker reload journey: an executable fixture worker with
/// marker-armed memory, swap, and completion acknowledgements, a resolver
/// config home, and the reload plus memory-limit routes wired to the real
/// supervisor.
final class ReloadLiveWorkerJourney {

    let supervisor: WorkerSupervisor;
    let transitionState: ConfigTransitionState;
    let homeDirectoryUrl: URL;
    let configFileUrl: URL;
    /// The generation the resolver derives from the authored file before any
    /// journey override — exactly what a reload candidate will carry.
    let candidateGeneration: String;
    private let routeTable: RestRouteTable;
    private let controlDirectoryPath: String;

    private init(
        supervisor: WorkerSupervisor,
        transitionState: ConfigTransitionState,
        homeDirectoryUrl: URL,
        configFileUrl: URL,
        candidateGeneration: String,
        routeTable: RestRouteTable,
        controlDirectoryPath: String
    ) {
        self.supervisor = supervisor;
        self.transitionState = transitionState;
        self.homeDirectoryUrl = homeDirectoryUrl;
        self.configFileUrl = configFileUrl;
        self.candidateGeneration = candidateGeneration;
        self.routeTable = routeTable;
        self.controlDirectoryPath = controlDirectoryPath;
    }

    /// Launches the fixture supervisor. A `nil` startup configuration mirrors
    /// Rust's WorkerHandle::launch: the fixture reports idle without any
    /// runtime-policy acknowledgement, unless the journey pins an explicit
    /// acknowledged generation for a later replacement candidate to echo.
    static func launch(
        startupConfiguration: WorkerStartupConfiguration?,
        modelPolicyCatalog: Dictionary<String, RuntimeModelPolicy>,
        fixtureAcknowledgedGeneration: String? = nil,
        reloadableConfigOverride: ((inout ResolvedRuntimeConfig) -> Void)? = nil
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
            startupConfiguration: startupConfiguration,
            fixtureAcknowledgedGeneration: fixtureAcknowledgedGeneration
                ?? startupConfiguration?.configurationGeneration
                ?? "");
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            FilePath(string: homeDirectoryUrl.path),
            runtimeInstance: AstronomicalRuntimeInstance.development);
        let resolver: ResolvedRuntimeConfigResolver = ResolvedRuntimeConfigResolver(
            instancePaths: instancePaths,
            fallbackWorkerExecutablePath: FilePath(string: fixtureWorkerPath));
        let resolvedFromFileGeneration: String = try resolver.load().configurationGeneration;
        var initialResolvedConfig: ResolvedRuntimeConfig = try resolver.load();
        reloadableConfigOverride?(&initialResolvedConfig);
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
            candidateGeneration: resolvedFromFileGeneration,
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

    func getStatusDocument() throws -> [String: Any] {
        let routeOutcome: RestRouteOutcome = self.routeTable.outcome(method: "GET", path: "/v1/status");
        guard case .handler(let routeHandler) = routeOutcome else {
            throw ReloadLiveWorkerJourneyFailure.routeMissing("status");
        }
        let statusResponse: RestHttpResponse = try routeHandler(WorkerReplacementJourney.emptyRequest(
            method: "GET",
            path: "/v1/status"));
        return try ConfigReloadJourney.decodeObject(statusResponse);
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

    /// Pins the generation the fixture acknowledges at startup — the poke a
    /// no-startup-policy journey issues once it knows the reload candidate's
    /// generation.
    func pokeCurrentGeneration(_ configurationGeneration: String) -> Void {
        try? configurationGeneration.write(
            toFile: self.controlDirectoryPath + "/current_generation",
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
        startupConfiguration: WorkerStartupConfiguration?,
        fixtureAcknowledgedGeneration: String
    ) throws -> String {
        let initialGeneration: String = fixtureAcknowledgedGeneration;
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
            + "if [ -n \"\(initialGeneration)\" ]; then printf '%s' \"\(initialGeneration)\" "
            + "> \"\(controlDirectoryPath)/current_generation\"; fi\n"
            + "startup_generation=$(cat \"\(controlDirectoryPath)/current_generation\" 2>/dev/null)\n"
            + "if [ -n \"$startup_generation\" ]; then\n"
            + "  emit_frame \"{\\\"kind\\\":\\\"runtime_feature_configuration_applied\\\","
            + "\\\"worker_runtime_feature_configuration\\\":{\\\"configuration_generation\\\":"
            + "\\\"$startup_generation\\\",\\\"persistent_prompt_cache_enabled\\\":true,"
            + "\\\"prompt_cache_maximum_size_bytes\\\":1073741824,\\\"loaded_model\\\":null}}\"\n"
            + "fi\n"
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
