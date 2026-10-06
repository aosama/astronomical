import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import JourneyCategories;

@testable import Supervisor;

/**
 * Acceptance journey for the instance status REST endpoint: an HTTP client on
 * the loopback asks the live daemon GET /v1/status and receives the full
 * status document — the application identity, the configured/resolved/
 * effective configuration generations with the effectiveness and restart
 * verdicts, discovery diagnostics and unmatched configured identities, the
 * ready model's effective policy, the prompt cache and memory summaries, the
 * worker acknowledgements, the expert residency, and the MLX memory
 * observations — so an operator can decide whether the instance serves the
 * configuration they authored. Every answer stays path-free: no model
 * directory and no cache location may leak into the document. Activity,
 * serving-session, and per-request progress sections join when their
 * worker-event sources land (E2).
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class RestStatusEndpointTests {

    private static let CONFIGURED_GENERATION: String = String(repeating: "a", count: 64);
    private static let WORKER_GENERATION: String = String(repeating: "b", count: 64);

    private let journeySupport: RestStatusJourneySupport = RestStatusJourneySupport();

    @Test
    func should_answer_the_fresh_instance_configuration_state_through_the_status_journey() throws {
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forExplicitStateDirectory(
            FilePath(string: "/status-journey-state"),
            defaultBindAddress: SocketEndpoint.loopback(port: 0));
        let server: RestHttpServer = try self.startServingServer(
            resolvedGeneration: RestStatusEndpointTests.CONFIGURED_GENERATION,
            workerHealthState: WorkerHealthState(),
            instancePaths: instancePaths);

        let (statusCode, envelope): (Int, [String: Any]) = try self.journeySupport.exchangeObject(
            port: server.boundEndpoint.port, requestTarget: "/v1/status");
        #expect(statusCode == 200);

        let application: [String: Any] = try self.journeySupport.requireObject(
            envelope["application"], failure: .missingApplicationSection);
        #expect(application["version"] as? String == "7.2.1-test");
        #expect(application["build_number"] as? UInt64 == 418);
        #expect(application["commit"] as? String == "cafebabe");
        #expect(application["is_dirty"] as? Bool == false);
        #expect(application["channel"] as? String == "development");
        #expect(application["channel_display_name"] as? String == "Development");
        #expect(application["state_directory"] as? String == "custom");

        #expect(envelope["status"] as? String == "loading");
        #expect(envelope["worker_runtime_feature_configuration_applied"] as? Bool == false);
        if envelope["worker_runtime_feature_configuration"] != nil
            && !(envelope["worker_runtime_feature_configuration"] is NSNull) {
            Issue.record("a fresh instance has no worker acknowledgement to echo");
        }

        #expect(envelope["configured_generation"] as? String == RestStatusEndpointTests.CONFIGURED_GENERATION);
        #expect(envelope["resolved_generation"] as? String == RestStatusEndpointTests.CONFIGURED_GENERATION);
        if envelope["effective_generation"] != nil
            && !(envelope["effective_generation"] is NSNull) {
            Issue.record("no worker has acknowledged a generation yet");
        }

        let configuration: [String: Any] = try self.journeySupport.requireObject(
            envelope["configuration"], failure: .missingConfigurationSection);
        #expect(configuration["configured_generation"] as? String == RestStatusEndpointTests.CONFIGURED_GENERATION);
        #expect(configuration["resolved_generation"] as? String == RestStatusEndpointTests.CONFIGURED_GENERATION);
        #expect(configuration["is_effective"] as? Bool == false);
        #expect(configuration["restart_required"] as? Bool == false);
        if configuration["validation_error"] != nil && !(configuration["validation_error"] is NSNull) {
            Issue.record("a clean resolution carries no validation error");
        }
        #expect((configuration["model_discovery_diagnostics"] as? [Any])?.count == 0);
        #expect((configuration["unmatched_model_config_ids"] as? [String])?.count == 0);
        if configuration["ready_model"] != nil && !(configuration["ready_model"] is NSNull) {
            Issue.record("no worker model is ready yet");
        }

        let promptCache: [String: Any] = try self.journeySupport.requireObject(
            configuration["prompt_cache"], failure: .missingPromptCacheSummary);
        let enabledTriple: [String: Any] = try self.journeySupport.requireObject(
            promptCache["enabled"], failure: .missingConfigurationTriple);
        #expect(enabledTriple["configured"] as? Bool == nil);
        #expect(enabledTriple["default"] as? Bool == true);
        #expect(enabledTriple["effective"] as? Bool == nil);
        let capacityTriple: [String: Any] = try self.journeySupport.requireObject(
            promptCache["capacity_bytes"], failure: .missingConfigurationTriple);
        #expect(capacityTriple["configured"] as? UInt64 == 50_000_000_000);
        #expect(capacityTriple["default"] as? UInt64 == 50_000_000_000);
        #expect(capacityTriple["effective"] as? UInt64 == nil);

        let memory: [String: Any] = try self.journeySupport.requireObject(
            configuration["memory"], failure: .missingMemorySummary);
        #expect(memory["configured_maximum_bytes"] as? UInt64 == nil);
        #expect(memory["pending_maximum_bytes"] as? UInt64 == nil);
        #expect(memory["error"] as? String == nil);
        server.stop();
    }

    @Test
    func should_require_a_restart_when_a_ready_worker_runs_a_stale_generation() throws {
        let workerHealthState: WorkerHealthState = WorkerHealthState();
        workerHealthState.publish(self.readyWorkerSnapshot(
            acknowledgedGeneration: RestStatusEndpointTests.WORKER_GENERATION,
            loadedModel: nil));
        let server: RestHttpServer = try self.startServingServer(
            resolvedGeneration: RestStatusEndpointTests.CONFIGURED_GENERATION,
            workerHealthState: workerHealthState);

        let (statusCode, envelope): (Int, [String: Any]) = try self.journeySupport.exchangeObject(
            port: server.boundEndpoint.port, requestTarget: "/v1/status");
        #expect(statusCode == 200);
        #expect(envelope["status"] as? String == "ready");
        #expect(envelope["configured_generation"] as? String == RestStatusEndpointTests.CONFIGURED_GENERATION);
        #expect(envelope["effective_generation"] as? String == RestStatusEndpointTests.WORKER_GENERATION);
        #expect(envelope["worker_runtime_feature_configuration_applied"] as? Bool == true);
        #expect(envelope["ready_model_id"] as? String == "synthetic-chat-model");

        let acknowledgedConfiguration: [String: Any] = try self.journeySupport.requireObject(
            envelope["worker_runtime_feature_configuration"], failure: .missingWorkerAcknowledgement);
        #expect(acknowledgedConfiguration["configuration_generation"] as? String == RestStatusEndpointTests.WORKER_GENERATION);

        let configuration: [String: Any] = try self.journeySupport.requireObject(
            envelope["configuration"], failure: .missingConfigurationSection);
        #expect(configuration["is_effective"] as? Bool == false);
        #expect(configuration["restart_required"] as? Bool == true);

        let promptCache: [String: Any] = try self.journeySupport.requireObject(
            configuration["prompt_cache"], failure: .missingPromptCacheSummary);
        let enabledTriple: [String: Any] = try self.journeySupport.requireObject(
            promptCache["enabled"], failure: .missingConfigurationTriple);
        #expect(enabledTriple["effective"] as? Bool == true);
        let capacityTriple: [String: Any] = try self.journeySupport.requireObject(
            promptCache["capacity_bytes"], failure: .missingConfigurationTriple);
        #expect(capacityTriple["effective"] as? UInt64 == 50_000_000_000);
        server.stop();
    }

    @Test
    func should_mark_a_matching_generation_effective_without_a_restart() throws {
        let workerHealthState: WorkerHealthState = WorkerHealthState();
        workerHealthState.publish(self.readyWorkerSnapshot(
            acknowledgedGeneration: RestStatusEndpointTests.CONFIGURED_GENERATION,
            loadedModel: nil));
        let server: RestHttpServer = try self.startServingServer(
            resolvedGeneration: RestStatusEndpointTests.CONFIGURED_GENERATION,
            workerHealthState: workerHealthState);

        let (statusCode, envelope): (Int, [String: Any]) = try self.journeySupport.exchangeObject(
            port: server.boundEndpoint.port, requestTarget: "/v1/status");
        #expect(statusCode == 200);
        let configuration: [String: Any] = try self.journeySupport.requireObject(
            envelope["configuration"], failure: .missingConfigurationSection);
        #expect(configuration["is_effective"] as? Bool == true);
        #expect(configuration["restart_required"] as? Bool == false);
        server.stop();
    }

    @Test
    func should_summarize_the_ready_model_policy_through_the_status_journey() throws {
        let loadedModel: WorkerLoadedModelRuntimeConfiguration = WorkerLoadedModelRuntimeConfiguration.autoregressive(
            WorkerLoadedAutoregressiveModelRuntimeConfiguration(
                modelId: "synthetic-chat-model",
                maximumContextTokens: 8_192,
                maximumOutputTokens: 1_024,
                chunking: WorkerChunkingConfiguration(
                    fixedPromptProcessingChunkSizeTokens: 2_048,
                    fixedSsdStreamingPromptProcessingChunkSizeTokens: 2_048,
                    fullAttentionKeyValueGrowthTokens: 256,
                    prefillGraphSubmissionLayerInterval: 0,
                    experimentalSsdPagingPrefillGraphSubmissionLayerInterval: 1,
                    experimentalSsdPagingGenerationGraphSubmissionLayerInterval: 3,
                    promptCacheBlockTokens: nil,
                    promptCacheCommonPrefixStrideBlocks: 4,
                    experimentalDecodeStageAttributionEnabled: false,
                    experimentalQuantizedKvCacheEnabled: false,
                    experimentalFusedMoeDecodeEnabled: false)));
        let workerHealthState: WorkerHealthState = WorkerHealthState();
        workerHealthState.publish(self.readyWorkerSnapshot(
            acknowledgedGeneration: RestStatusEndpointTests.CONFIGURED_GENERATION,
            loadedModel: loadedModel));
        let server: RestHttpServer = try self.startServingServer(
            resolvedGeneration: RestStatusEndpointTests.CONFIGURED_GENERATION,
            workerHealthState: workerHealthState);

        let (statusCode, envelope): (Int, [String: Any]) = try self.journeySupport.exchangeObject(
            port: server.boundEndpoint.port, requestTarget: "/v1/status");
        #expect(statusCode == 200);
        #expect(envelope["ready_model_id"] as? String == "synthetic-chat-model");
        #expect(envelope["ready_model_size_bytes"] as? UInt64 == 400_000_000);

        let configuration: [String: Any] = try self.journeySupport.requireObject(
            envelope["configuration"], failure: .missingConfigurationSection);
        let readyModel: [String: Any] = try self.journeySupport.requireObject(
            configuration["ready_model"], failure: .missingReadyModelSummary);
        #expect(readyModel["model_id"] as? String == "synthetic-chat-model");

        let contextTriple: [String: Any] = try self.journeySupport.requireObject(
            readyModel["maximum_context_tokens"], failure: .missingConfigurationTriple);
        #expect(contextTriple["configured"] as? UInt32 == nil);
        #expect(contextTriple["default"] as? UInt32 == 4_096);
        #expect(contextTriple["effective"] as? UInt32 == 8_192);

        let outputTriple: [String: Any] = try self.journeySupport.requireObject(
            readyModel["maximum_output_default_tokens"], failure: .missingConfigurationTriple);
        #expect(outputTriple["configured"] as? UInt32 == nil);
        #expect(outputTriple["default"] as? UInt32 == 4_095);
        // The resolver's own policy caps the output default one below the
        // artifact context (min(20_480, 4_096 - 1)), not the raw 20_480.
        #expect(outputTriple["effective"] as? UInt32 == 4_095);

        let temperatureTriple: [String: Any] = try self.journeySupport.requireObject(
            readyModel["temperature"], failure: .missingConfigurationTriple);
        #expect(temperatureTriple["configured"] as? Double == nil);
        #expect(temperatureTriple["default"] as? Double == nil);
        #expect(temperatureTriple["effective"] as? Double == nil);

        let chunking: [String: Any] = try self.journeySupport.requireObject(
            readyModel["chunking"], failure: .missingChunkingSummary);
        let fixedChunkTriple: [String: Any] = try self.journeySupport.requireObject(
            chunking["fixed_prompt_processing_chunk_size_tokens"], failure: .missingConfigurationTriple);
        // The first-run document authors the fixed chunk quantities explicitly
        // (UserConfigFile.minimal), so the empty-config journey reports them
        // as configured rather than falling back to the built-in defaults.
        #expect(fixedChunkTriple["configured"] as? UInt32 == 2_048);
        #expect(fixedChunkTriple["default"] as? UInt32 == 2_048);
        #expect(fixedChunkTriple["effective"] as? UInt32 == 2_048);
        let blockTokensTriple: [String: Any] = try self.journeySupport.requireObject(
            chunking["prompt_cache_block_tokens"], failure: .missingConfigurationTriple);
        #expect(blockTokensTriple["is_configured"] as? Bool == false);
        #expect(blockTokensTriple["configured"] as? UInt32 == nil);
        #expect(blockTokensTriple["default"] as? UInt32 == nil);
        #expect(blockTokensTriple["effective"] as? UInt32 == nil);
        let strideTriple: [String: Any] = try self.journeySupport.requireObject(
            chunking["prompt_cache_common_prefix_stride_blocks"], failure: .missingConfigurationTriple);
        #expect(strideTriple["default"] as? UInt32 == 4);
        #expect(strideTriple["effective"] as? UInt32 == 4);
        server.stop();
    }

    @Test
    func should_report_the_authored_model_policy_and_stay_path_free() throws {
        // The authored policy mirrors the Rust status contract test: an
        // explicitly configured context, output default, temperature, top-p,
        // and one chunking field, with the remaining chunk quantities left to
        // the built-in defaults.
        let authoredChunking: WorkerChunkingConfiguration = WorkerChunkingConfiguration(
            fixedPromptProcessingChunkSizeTokens: 2_048,
            fixedSsdStreamingPromptProcessingChunkSizeTokens: 256,
            fullAttentionKeyValueGrowthTokens: 256,
            prefillGraphSubmissionLayerInterval: 0,
            experimentalSsdPagingPrefillGraphSubmissionLayerInterval: 1,
            experimentalSsdPagingGenerationGraphSubmissionLayerInterval: 0,
            promptCacheBlockTokens: 128,
            promptCacheCommonPrefixStrideBlocks: 4,
            experimentalDecodeStageAttributionEnabled: false,
            experimentalQuantizedKvCacheEnabled: false,
            experimentalFusedMoeDecodeEnabled: false);
        let authoredPolicy: RuntimeModelPolicy = RuntimeModelPolicy(
            modelDirectory: FilePath(string: "/fictional/private/target"),
            generationDefaults: RuntimeModelGenerationDefaults(
                maximumOutputTokens: 1_024,
                configuredMaximumOutputTokens: 1_024,
                temperatureThousandths: 700,
                topPThousandths: 900),
            configuredMaximumContextTokens: 16_384,
            defaultMaximumContextTokens: 32_768,
            configuredChunkingFields: ConfiguredChunkingFields(
                fixedPromptProcessingChunkSizeTokens: true,
                fixedSsdStreamingPromptProcessingChunkSizeTokens: false,
                fullAttentionKeyValueGrowthTokens: false,
                prefillGraphSubmissionLayerInterval: false,
                experimentalSsdPagingPrefillGraphSubmissionLayerInterval: false,
                experimentalSsdPagingGenerationGraphSubmissionLayerInterval: false,
                promptCacheBlockTokens: false,
                promptCacheCommonPrefixStrideBlocks: false,
                experimentalDecodeStageAttributionEnabled: false,
                experimentalQuantizedKvCacheEnabled: false,
                experimentalFusedMoeDecodeEnabled: false),
            workerModelConfiguration: WorkerModelConfiguration.autoregressive(
                WorkerAutoregressiveModelConfiguration(
                    modelId: "synthetic-chat-model",
                    maximumContextTokens: 16_384,
                    maximumOutputTokens: 4_096,
                    chunking: authoredChunking)));
        let workerHealthState: WorkerHealthState = WorkerHealthState();
        workerHealthState.publish(self.readyWorkerSnapshot(
            acknowledgedGeneration: RestStatusEndpointTests.WORKER_GENERATION,
            loadedModel: authoredPolicy.workerModelConfiguration.runtimeConfiguration()));
        var resolvedConfig: ResolvedRuntimeConfig = try self.journeySupport.makeResolvedConfig(
            resolvedGeneration: RestStatusEndpointTests.CONFIGURED_GENERATION);
        resolvedConfig.modelPolicyCatalog["synthetic-chat-model"] = authoredPolicy;
        let server: RestHttpServer = try self.startServingServer(
            resolvedRuntimeConfig: resolvedConfig,
            workerHealthState: workerHealthState);

        let (statusCode, envelope): (Int, [String: Any]) = try self.journeySupport.exchangeObject(
            port: server.boundEndpoint.port, requestTarget: "/v1/status");
        #expect(statusCode == 200);
        #expect(envelope["configured_generation"] as? String == RestStatusEndpointTests.CONFIGURED_GENERATION);
        #expect(envelope["effective_generation"] as? String == RestStatusEndpointTests.WORKER_GENERATION);

        let configuration: [String: Any] = try self.journeySupport.requireObject(
            envelope["configuration"], failure: .missingConfigurationSection);
        #expect(configuration["restart_required"] as? Bool == true);
        let readyModel: [String: Any] = try self.journeySupport.requireObject(
            configuration["ready_model"], failure: .missingReadyModelSummary);
        let contextTriple: [String: Any] = try self.journeySupport.requireObject(
            readyModel["maximum_context_tokens"], failure: .missingConfigurationTriple);
        #expect(contextTriple["configured"] as? UInt32 == 16_384);
        #expect(contextTriple["default"] as? UInt32 == 32_768);
        #expect(contextTriple["effective"] as? UInt32 == 16_384);
        let outputTriple: [String: Any] = try self.journeySupport.requireObject(
            readyModel["maximum_output_default_tokens"], failure: .missingConfigurationTriple);
        #expect(outputTriple["configured"] as? UInt32 == 1_024);
        #expect(outputTriple["default"] as? UInt32 == 16_383);
        #expect(outputTriple["effective"] as? UInt32 == 1_024);
        let temperatureTriple: [String: Any] = try self.journeySupport.requireObject(
            readyModel["temperature"], failure: .missingConfigurationTriple);
        #expect(temperatureTriple["configured"] as? Double == 0.7);
        let topPTriple: [String: Any] = try self.journeySupport.requireObject(
            readyModel["top_p"], failure: .missingConfigurationTriple);
        #expect(topPTriple["configured"] as? Double == 0.9);
        let chunking: [String: Any] = try self.journeySupport.requireObject(
            readyModel["chunking"], failure: .missingChunkingSummary);
        let fixedChunkTriple: [String: Any] = try self.journeySupport.requireObject(
            chunking["fixed_prompt_processing_chunk_size_tokens"], failure: .missingConfigurationTriple);
        #expect(fixedChunkTriple["configured"] as? UInt32 == 2_048);
        let fullAttentionTriple: [String: Any] = try self.journeySupport.requireObject(
            chunking["full_attention_key_value_growth_tokens"], failure: .missingConfigurationTriple);
        #expect(fullAttentionTriple["configured"] as? UInt32 == nil);
        #expect(fullAttentionTriple["effective"] as? UInt32 == 256);
        let generationIntervalTriple: [String: Any] = try self.journeySupport.requireObject(
            chunking["experimental_ssd_paging_generation_graph_submission_layer_interval"], failure: .missingConfigurationTriple);
        #expect(generationIntervalTriple["configured"] as? UInt32 == nil);
        #expect(generationIntervalTriple["effective"] as? UInt32 == 0);
        let blockTokensTriple: [String: Any] = try self.journeySupport.requireObject(
            chunking["prompt_cache_block_tokens"], failure: .missingConfigurationTriple);
        #expect(blockTokensTriple["is_configured"] as? Bool == false);
        #expect(blockTokensTriple["configured"] as? UInt32 == nil);
        #expect(blockTokensTriple["effective"] as? UInt32 == 128);

        let statusText: String = try self.journeySupport.requireResponseText(RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /v1/status HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"));
        #expect(!statusText.contains("/fictional/private"), "no model directory may leak into the status document");
        #expect(!statusText.contains("prompt-cache"), "no cache location may leak into the status document");
        server.stop();
    }

    @Test
    func should_surface_diagnostics_unmatched_ids_and_memory_observations_through_the_status_journey() throws {
        let workerHealthState: WorkerHealthState = WorkerHealthState();
        var observedWorkerSnapshot: WorkerHealthSnapshot = self.readyWorkerSnapshot(
            acknowledgedGeneration: RestStatusEndpointTests.WORKER_GENERATION,
            loadedModel: nil);
        observedWorkerSnapshot.machineMlxMemoryCeilingBytes = 128_000_000_000;
        observedWorkerSnapshot.effectiveMlxMemoryCeilingBytes = 90_000_000_000;
        observedWorkerSnapshot.minimumMlxMemoryCeilingBytes = 8_000_000_000;
        observedWorkerSnapshot.pendingMlxMemoryCeilingBytes = 64_000_000_000;
        observedWorkerSnapshot.mlxMemoryLimitError = "the authored ceiling cannot hold the resident model";
        observedWorkerSnapshot.expertMemoryMode = ExpertMemoryMode.paged;
        observedWorkerSnapshot.expertResidency = WorkerExpertResidencySnapshot(
            totalLayerCount: 80,
            residentExpertCount: 20,
            residentExpertPayloadBytes: 5_000_000_000);
        observedWorkerSnapshot.latestMlxMemorySnapshot = WorkerMlxMemorySnapshot(
            source: MlxMemorySnapshotSource.idlePoll,
            activeMemoryBytes: 1_000_000_000,
            allocatorCacheMemoryBytes: 2_000_000_000,
            peakMemoryBytes: 3_000_000_000,
            expertPayloadBytes: 4_000_000_000,
            modelCorePayloadBytes: 5_000_000_000,
            contextStatePayloadBytes: 6_000_000_000,
            memoryCeilingUtilization: nil);
        workerHealthState.publish(observedWorkerSnapshot);

        var resolvedConfig: ResolvedRuntimeConfig = try self.journeySupport.makeResolvedConfig(
            resolvedGeneration: RestStatusEndpointTests.CONFIGURED_GENERATION);
        resolvedConfig.maximumMlxMemoryBytes = 90_000_000_000;
        resolvedConfig.modelDiscoveryDiagnostics = [
            DiscoveryModelDiscoveryDiagnostic.ambiguousModelIdentity(
                modelId: "duplicated-model",
                configuredRootNumbers: [1, 2]),
        ];
        resolvedConfig.unmatchedModelConfigIds = ["vendor/unmatched-model"];
        let server: RestHttpServer = try self.startServingServer(
            resolvedRuntimeConfig: resolvedConfig,
            workerHealthState: workerHealthState);

        let (statusCode, envelope): (Int, [String: Any]) = try self.journeySupport.exchangeObject(
            port: server.boundEndpoint.port, requestTarget: "/v1/status");
        #expect(statusCode == 200);
        #expect(envelope["configured_maximum_mlx_memory_gb"] as? UInt64 == 90);
        #expect(envelope["mlx_memory_ceiling_bytes"] as? UInt64 == 90_000_000_000);
        #expect(envelope["machine_mlx_memory_ceiling_bytes"] as? UInt64 == 128_000_000_000);
        #expect(envelope["minimum_mlx_memory_ceiling_bytes"] as? UInt64 == 8_000_000_000);
        #expect(envelope["pending_mlx_memory_ceiling_bytes"] as? UInt64 == 64_000_000_000);
        #expect(envelope["mlx_memory_limit_error"] as? String == "the authored ceiling cannot hold the resident model");
        #expect(envelope["expert_memory_mode"] as? String == "paged");
        let expertResidency: [String: Any] = try self.journeySupport.requireObject(
            envelope["expert_residency"], failure: .missingExpertResidency);
        #expect(expertResidency["total_layer_count"] as? UInt32 == 80);
        #expect(expertResidency["resident_expert_count"] as? UInt32 == 20);
        #expect(expertResidency["resident_expert_payload_bytes"] as? UInt64 == 5_000_000_000);
        let mlxSnapshot: [String: Any] = try self.journeySupport.requireObject(
            envelope["mlx_memory_snapshot"], failure: .missingMlxSnapshot);
        #expect(mlxSnapshot["source"] as? String == "idle_poll");
        #expect(mlxSnapshot["active_memory_bytes"] as? UInt64 == 1_000_000_000);

        let configuration: [String: Any] = try self.journeySupport.requireObject(
            envelope["configuration"], failure: .missingConfigurationSection);
        #expect(configuration["is_effective"] as? Bool == false);
        // A pending memory-ceiling change is mid-flight, so the stale worker
        // generation must not demand a restart on top of it.
        #expect(configuration["restart_required"] as? Bool == false);

        let memory: [String: Any] = try self.journeySupport.requireObject(
            configuration["memory"], failure: .missingMemorySummary);
        #expect(memory["configured_maximum_bytes"] as? UInt64 == 90_000_000_000);
        #expect(memory["effective_maximum_bytes"] as? UInt64 == 90_000_000_000);
        #expect(memory["pending_maximum_bytes"] as? UInt64 == 64_000_000_000);
        #expect(memory["error"] as? String == "the authored ceiling cannot hold the resident model");

        let diagnostics: Array<Any> = try self.journeySupport.requireArray(
            configuration["model_discovery_diagnostics"], failure: .missingDiagnostics);
        #expect(diagnostics.count == 1);
        let diagnostic: [String: Any] = try self.journeySupport.requireObject(
            diagnostics[0], failure: .missingDiagnostics);
        #expect(diagnostic["code"] as? String == "ambiguous_model_identity");
        #expect(diagnostic["model_id"] as? String == "duplicated-model");
        #expect(diagnostic["configured_root_numbers"] as? [Int] == [1, 2]);

        #expect(configuration["unmatched_model_config_ids"] as? [String] == ["vendor/unmatched-model"]);
        server.stop();
    }

    @Test
    func should_surface_the_unavailable_directory_diagnostic_without_paths() throws {
        var resolvedConfig: ResolvedRuntimeConfig = try self.journeySupport.makeResolvedConfig(
            resolvedGeneration: RestStatusEndpointTests.CONFIGURED_GENERATION);
        resolvedConfig.modelDiscoveryDiagnostics = [
            DiscoveryModelDiscoveryDiagnostic.unavailableModelDirectory(configuredRootNumber: 4),
        ];
        let server: RestHttpServer = try self.startServingServer(
            resolvedRuntimeConfig: resolvedConfig,
            workerHealthState: WorkerHealthState());

        let (statusCode, envelope): (Int, [String: Any]) = try self.journeySupport.exchangeObject(
            port: server.boundEndpoint.port, requestTarget: "/v1/status");
        #expect(statusCode == 200);
        let configuration: [String: Any] = try self.journeySupport.requireObject(
            envelope["configuration"], failure: .missingConfigurationSection);
        let diagnostics: Array<Any> = try self.journeySupport.requireArray(
            configuration["model_discovery_diagnostics"], failure: .missingDiagnostics);
        #expect(diagnostics.count == 1);
        let diagnostic: [String: Any] = try self.journeySupport.requireObject(
            diagnostics[0], failure: .missingDiagnostics);
        #expect(diagnostic["code"] as? String == "unavailable_model_directory");
        #expect(diagnostic["model_id"] as? String == "");
        #expect(diagnostic["configured_root_numbers"] as? [Int] == [4]);

        let statusText: String = try self.journeySupport.requireResponseText(RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /v1/status HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"));
        #expect(!statusText.contains("/models/synthetic-chat-model"), "no model directory may leak into the status document");
        server.stop();
    }

    @Test
    func should_suppress_the_restart_verdict_while_a_validation_failure_stands() throws {
        let workerHealthState: WorkerHealthState = WorkerHealthState();
        workerHealthState.publish(self.readyWorkerSnapshot(
            acknowledgedGeneration: RestStatusEndpointTests.WORKER_GENERATION,
            loadedModel: nil));
        let server: RestHttpServer = try self.startServingServer(
            resolvedGeneration: RestStatusEndpointTests.CONFIGURED_GENERATION,
            workerHealthState: workerHealthState,
            configurationValidationError: "the configuration document was rejected: unknown field");

        let (statusCode, envelope): (Int, [String: Any]) = try self.journeySupport.exchangeObject(
            port: server.boundEndpoint.port, requestTarget: "/v1/status");
        #expect(statusCode == 200);
        let configuration: [String: Any] = try self.journeySupport.requireObject(
            envelope["configuration"], failure: .missingConfigurationSection);
        #expect(configuration["validation_error"] as? String == "the configuration document was rejected: unknown field");
        #expect(configuration["is_effective"] as? Bool == false);
        #expect(configuration["restart_required"] as? Bool == false);
        server.stop();
    }

    // MARK: - Journey helpers

    private func startServingServer(
        resolvedGeneration: String,
        workerHealthState: WorkerHealthState,
        instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forExplicitStateDirectory(
            FilePath(string: "/status-journey-state"),
            defaultBindAddress: SocketEndpoint.loopback(port: 0)),
        configurationValidationError: String? = nil
    ) throws -> RestHttpServer {
        let resolvedConfig: ResolvedRuntimeConfig = try self.journeySupport.makeResolvedConfig(
            resolvedGeneration: resolvedGeneration);
        return try self.startServingServer(
            resolvedRuntimeConfig: resolvedConfig,
            workerHealthState: workerHealthState,
            instancePaths: instancePaths,
            configurationValidationError: configurationValidationError);
    }

    private func startServingServer(
        resolvedRuntimeConfig: ResolvedRuntimeConfig,
        workerHealthState: WorkerHealthState,
        instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forExplicitStateDirectory(
            FilePath(string: "/status-journey-state"),
            defaultBindAddress: SocketEndpoint.loopback(port: 0)),
        configurationValidationError: String? = nil
    ) throws -> RestHttpServer {
        let routeTable: RestRouteTable = RestEndpointRoutes.servingRouteTable(
            resolvedRuntimeConfig: resolvedRuntimeConfig,
            workerHealthState: workerHealthState,
            instancePaths: instancePaths,
            buildIdentity: ApplicationBuildIdentity(
                version: "7.2.1-test",
                buildNumber: 418,
                commit: "cafebabe",
                isDirty: false),
            configurationValidationError: configurationValidationError);
        return try RestHttpServer.start(
            bindEndpoint: SocketEndpoint.loopback(port: 0),
            routeTable: routeTable);
    }

    private func readyWorkerSnapshot(
        acknowledgedGeneration: String,
        loadedModel: WorkerLoadedModelRuntimeConfiguration?
    ) -> WorkerHealthSnapshot {
        var workerSnapshot: WorkerHealthSnapshot = WorkerHealthSnapshot.readyWithModel(
            modelId: "synthetic-chat-model",
            capabilities: WorkerModelCapabilities.from(
                chatCapabilities: ChatModelCapabilities(
                    supportsReasoning: false,
                    supportsToolCalls: false,
                    hasVision: false,
                    maxInputTokens: 4_095,
                    maxOutputTokens: 1_024,
                    contextWindow: 4_096)));
        workerSnapshot.workerRuntimeFeatureConfiguration = WorkerRuntimeFeatureConfiguration(
            configurationGeneration: acknowledgedGeneration,
            persistentPromptCacheEnabled: true,
            promptCacheMaximumSizeBytes: 50_000_000_000,
            loadedModel: loadedModel);
        return workerSnapshot;
    }
}
