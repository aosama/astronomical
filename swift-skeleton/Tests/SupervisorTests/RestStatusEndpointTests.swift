import Foundation;
import XCTest;

import AstronomicalConfig;
import IpcProtocol;

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
final class RestStatusEndpointTests: XCTestCase {

    private static let CONFIGURED_GENERATION: String = String(repeating: "a", count: 64);
    private static let WORKER_GENERATION: String = String(repeating: "b", count: 64);

    private let journeySupport: RestStatusJourneySupport = RestStatusJourneySupport();

    func testStatusJourneyAnswersTheFreshInstanceConfigurationState() throws {
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forExplicitStateDirectory(
            FilePath(string: "/status-journey-state"),
            defaultBindAddress: SocketEndpoint.loopback(port: 0));
        let server: RestHttpServer = try self.startServingServer(
            resolvedGeneration: RestStatusEndpointTests.CONFIGURED_GENERATION,
            workerHealthState: WorkerHealthState(),
            instancePaths: instancePaths);

        let (statusCode, envelope): (Int, [String: Any]) = try self.journeySupport.exchangeObject(
            port: server.boundEndpoint.port, requestTarget: "/v1/status");
        XCTAssertEqual(statusCode, 200);

        let application: [String: Any] = try self.journeySupport.requireObject(
            envelope["application"], failure: .missingApplicationSection);
        XCTAssertEqual(application["version"] as? String, "7.2.1-test");
        XCTAssertEqual(application["build_number"] as? UInt64, 418);
        XCTAssertEqual(application["commit"] as? String, "cafebabe");
        XCTAssertEqual(application["is_dirty"] as? Bool, false);
        XCTAssertEqual(application["channel"] as? String, "development");
        XCTAssertEqual(application["channel_display_name"] as? String, "Development");
        XCTAssertEqual(application["state_directory"] as? String, "custom");

        XCTAssertEqual(envelope["status"] as? String, "loading");
        XCTAssertEqual(envelope["worker_runtime_feature_configuration_applied"] as? Bool, false);
        if envelope["worker_runtime_feature_configuration"] != nil
            && !(envelope["worker_runtime_feature_configuration"] is NSNull) {
            XCTFail("a fresh instance has no worker acknowledgement to echo");
        }

        XCTAssertEqual(envelope["configured_generation"] as? String, RestStatusEndpointTests.CONFIGURED_GENERATION);
        XCTAssertEqual(envelope["resolved_generation"] as? String, RestStatusEndpointTests.CONFIGURED_GENERATION);
        if envelope["effective_generation"] != nil
            && !(envelope["effective_generation"] is NSNull) {
            XCTFail("no worker has acknowledged a generation yet");
        }

        let configuration: [String: Any] = try self.journeySupport.requireObject(
            envelope["configuration"], failure: .missingConfigurationSection);
        XCTAssertEqual(configuration["configured_generation"] as? String, RestStatusEndpointTests.CONFIGURED_GENERATION);
        XCTAssertEqual(configuration["resolved_generation"] as? String, RestStatusEndpointTests.CONFIGURED_GENERATION);
        XCTAssertEqual(configuration["is_effective"] as? Bool, false);
        XCTAssertEqual(configuration["restart_required"] as? Bool, false);
        if configuration["validation_error"] != nil && !(configuration["validation_error"] is NSNull) {
            XCTFail("a clean resolution carries no validation error");
        }
        XCTAssertEqual((configuration["model_discovery_diagnostics"] as? [Any])?.count, 0);
        XCTAssertEqual((configuration["unmatched_model_config_ids"] as? [String])?.count, 0);
        if configuration["ready_model"] != nil && !(configuration["ready_model"] is NSNull) {
            XCTFail("no worker model is ready yet");
        }

        let promptCache: [String: Any] = try self.journeySupport.requireObject(
            configuration["prompt_cache"], failure: .missingPromptCacheSummary);
        let enabledTriple: [String: Any] = try self.journeySupport.requireObject(
            promptCache["enabled"], failure: .missingConfigurationTriple);
        XCTAssertEqual(enabledTriple["configured"] as? Bool, nil);
        XCTAssertEqual(enabledTriple["default"] as? Bool, true);
        XCTAssertEqual(enabledTriple["effective"] as? Bool, nil);
        let capacityTriple: [String: Any] = try self.journeySupport.requireObject(
            promptCache["capacity_bytes"], failure: .missingConfigurationTriple);
        XCTAssertEqual(capacityTriple["configured"] as? UInt64, 50_000_000_000);
        XCTAssertEqual(capacityTriple["default"] as? UInt64, 50_000_000_000);
        XCTAssertEqual(capacityTriple["effective"] as? UInt64, nil);

        let memory: [String: Any] = try self.journeySupport.requireObject(
            configuration["memory"], failure: .missingMemorySummary);
        XCTAssertEqual(memory["configured_maximum_bytes"] as? UInt64, nil);
        XCTAssertEqual(memory["pending_maximum_bytes"] as? UInt64, nil);
        XCTAssertEqual(memory["error"] as? String, nil);
        server.stop();
    }

    func testStatusJourneyRequiresRestartWhenAReadyWorkerRunsAStaleGeneration() throws {
        let workerHealthState: WorkerHealthState = WorkerHealthState();
        workerHealthState.publish(self.readyWorkerSnapshot(
            acknowledgedGeneration: RestStatusEndpointTests.WORKER_GENERATION,
            loadedModel: nil));
        let server: RestHttpServer = try self.startServingServer(
            resolvedGeneration: RestStatusEndpointTests.CONFIGURED_GENERATION,
            workerHealthState: workerHealthState);

        let (statusCode, envelope): (Int, [String: Any]) = try self.journeySupport.exchangeObject(
            port: server.boundEndpoint.port, requestTarget: "/v1/status");
        XCTAssertEqual(statusCode, 200);
        XCTAssertEqual(envelope["status"] as? String, "ready");
        XCTAssertEqual(envelope["configured_generation"] as? String, RestStatusEndpointTests.CONFIGURED_GENERATION);
        XCTAssertEqual(envelope["effective_generation"] as? String, RestStatusEndpointTests.WORKER_GENERATION);
        XCTAssertEqual(envelope["worker_runtime_feature_configuration_applied"] as? Bool, true);
        XCTAssertEqual(envelope["ready_model_id"] as? String, "synthetic-chat-model");

        let acknowledgedConfiguration: [String: Any] = try self.journeySupport.requireObject(
            envelope["worker_runtime_feature_configuration"], failure: .missingWorkerAcknowledgement);
        XCTAssertEqual(acknowledgedConfiguration["configuration_generation"] as? String, RestStatusEndpointTests.WORKER_GENERATION);

        let configuration: [String: Any] = try self.journeySupport.requireObject(
            envelope["configuration"], failure: .missingConfigurationSection);
        XCTAssertEqual(configuration["is_effective"] as? Bool, false);
        XCTAssertEqual(configuration["restart_required"] as? Bool, true);

        let promptCache: [String: Any] = try self.journeySupport.requireObject(
            configuration["prompt_cache"], failure: .missingPromptCacheSummary);
        let enabledTriple: [String: Any] = try self.journeySupport.requireObject(
            promptCache["enabled"], failure: .missingConfigurationTriple);
        XCTAssertEqual(enabledTriple["effective"] as? Bool, true);
        let capacityTriple: [String: Any] = try self.journeySupport.requireObject(
            promptCache["capacity_bytes"], failure: .missingConfigurationTriple);
        XCTAssertEqual(capacityTriple["effective"] as? UInt64, 50_000_000_000);
        server.stop();
    }

    func testStatusJourneyMarksAMatchingGenerationEffectiveWithoutRestart() throws {
        let workerHealthState: WorkerHealthState = WorkerHealthState();
        workerHealthState.publish(self.readyWorkerSnapshot(
            acknowledgedGeneration: RestStatusEndpointTests.CONFIGURED_GENERATION,
            loadedModel: nil));
        let server: RestHttpServer = try self.startServingServer(
            resolvedGeneration: RestStatusEndpointTests.CONFIGURED_GENERATION,
            workerHealthState: workerHealthState);

        let (statusCode, envelope): (Int, [String: Any]) = try self.journeySupport.exchangeObject(
            port: server.boundEndpoint.port, requestTarget: "/v1/status");
        XCTAssertEqual(statusCode, 200);
        let configuration: [String: Any] = try self.journeySupport.requireObject(
            envelope["configuration"], failure: .missingConfigurationSection);
        XCTAssertEqual(configuration["is_effective"] as? Bool, true);
        XCTAssertEqual(configuration["restart_required"] as? Bool, false);
        server.stop();
    }

    func testStatusJourneySummarizesTheReadyModelPolicy() throws {
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
        XCTAssertEqual(statusCode, 200);
        XCTAssertEqual(envelope["ready_model_id"] as? String, "synthetic-chat-model");
        XCTAssertEqual(envelope["ready_model_size_bytes"] as? UInt64, 400_000_000);

        let configuration: [String: Any] = try self.journeySupport.requireObject(
            envelope["configuration"], failure: .missingConfigurationSection);
        let readyModel: [String: Any] = try self.journeySupport.requireObject(
            configuration["ready_model"], failure: .missingReadyModelSummary);
        XCTAssertEqual(readyModel["model_id"] as? String, "synthetic-chat-model");

        let contextTriple: [String: Any] = try self.journeySupport.requireObject(
            readyModel["maximum_context_tokens"], failure: .missingConfigurationTriple);
        XCTAssertEqual(contextTriple["configured"] as? UInt32, nil);
        XCTAssertEqual(contextTriple["default"] as? UInt32, 4_096);
        XCTAssertEqual(contextTriple["effective"] as? UInt32, 8_192);

        let outputTriple: [String: Any] = try self.journeySupport.requireObject(
            readyModel["maximum_output_default_tokens"], failure: .missingConfigurationTriple);
        XCTAssertEqual(outputTriple["configured"] as? UInt32, nil);
        XCTAssertEqual(outputTriple["default"] as? UInt32, 4_095);
        // The resolver's own policy caps the output default one below the
        // artifact context (min(20_480, 4_096 - 1)), not the raw 20_480.
        XCTAssertEqual(outputTriple["effective"] as? UInt32, 4_095);

        let temperatureTriple: [String: Any] = try self.journeySupport.requireObject(
            readyModel["temperature"], failure: .missingConfigurationTriple);
        XCTAssertEqual(temperatureTriple["configured"] as? Double, nil);
        XCTAssertEqual(temperatureTriple["default"] as? Double, nil);
        XCTAssertEqual(temperatureTriple["effective"] as? Double, nil);

        let chunking: [String: Any] = try self.journeySupport.requireObject(
            readyModel["chunking"], failure: .missingChunkingSummary);
        let fixedChunkTriple: [String: Any] = try self.journeySupport.requireObject(
            chunking["fixed_prompt_processing_chunk_size_tokens"], failure: .missingConfigurationTriple);
        // The first-run document authors the fixed chunk quantities explicitly
        // (UserConfigFile.minimal), so the empty-config journey reports them
        // as configured rather than falling back to the built-in defaults.
        XCTAssertEqual(fixedChunkTriple["configured"] as? UInt32, 2_048);
        XCTAssertEqual(fixedChunkTriple["default"] as? UInt32, 2_048);
        XCTAssertEqual(fixedChunkTriple["effective"] as? UInt32, 2_048);
        let blockTokensTriple: [String: Any] = try self.journeySupport.requireObject(
            chunking["prompt_cache_block_tokens"], failure: .missingConfigurationTriple);
        XCTAssertEqual(blockTokensTriple["is_configured"] as? Bool, false);
        XCTAssertEqual(blockTokensTriple["configured"] as? UInt32, nil);
        XCTAssertEqual(blockTokensTriple["default"] as? UInt32, nil);
        XCTAssertEqual(blockTokensTriple["effective"] as? UInt32, nil);
        let strideTriple: [String: Any] = try self.journeySupport.requireObject(
            chunking["prompt_cache_common_prefix_stride_blocks"], failure: .missingConfigurationTriple);
        XCTAssertEqual(strideTriple["default"] as? UInt32, 4);
        XCTAssertEqual(strideTriple["effective"] as? UInt32, 4);
        server.stop();
    }

    func testStatusJourneyReportsTheAuthoredModelPolicyAndStaysPathFree() throws {
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
        XCTAssertEqual(statusCode, 200);
        XCTAssertEqual(envelope["configured_generation"] as? String, RestStatusEndpointTests.CONFIGURED_GENERATION);
        XCTAssertEqual(envelope["effective_generation"] as? String, RestStatusEndpointTests.WORKER_GENERATION);

        let configuration: [String: Any] = try self.journeySupport.requireObject(
            envelope["configuration"], failure: .missingConfigurationSection);
        XCTAssertEqual(configuration["restart_required"] as? Bool, true);
        let readyModel: [String: Any] = try self.journeySupport.requireObject(
            configuration["ready_model"], failure: .missingReadyModelSummary);
        let contextTriple: [String: Any] = try self.journeySupport.requireObject(
            readyModel["maximum_context_tokens"], failure: .missingConfigurationTriple);
        XCTAssertEqual(contextTriple["configured"] as? UInt32, 16_384);
        XCTAssertEqual(contextTriple["default"] as? UInt32, 32_768);
        XCTAssertEqual(contextTriple["effective"] as? UInt32, 16_384);
        let outputTriple: [String: Any] = try self.journeySupport.requireObject(
            readyModel["maximum_output_default_tokens"], failure: .missingConfigurationTriple);
        XCTAssertEqual(outputTriple["configured"] as? UInt32, 1_024);
        XCTAssertEqual(outputTriple["default"] as? UInt32, 16_383);
        XCTAssertEqual(outputTriple["effective"] as? UInt32, 1_024);
        let temperatureTriple: [String: Any] = try self.journeySupport.requireObject(
            readyModel["temperature"], failure: .missingConfigurationTriple);
        XCTAssertEqual(temperatureTriple["configured"] as? Double, 0.7);
        let topPTriple: [String: Any] = try self.journeySupport.requireObject(
            readyModel["top_p"], failure: .missingConfigurationTriple);
        XCTAssertEqual(topPTriple["configured"] as? Double, 0.9);
        let chunking: [String: Any] = try self.journeySupport.requireObject(
            readyModel["chunking"], failure: .missingChunkingSummary);
        let fixedChunkTriple: [String: Any] = try self.journeySupport.requireObject(
            chunking["fixed_prompt_processing_chunk_size_tokens"], failure: .missingConfigurationTriple);
        XCTAssertEqual(fixedChunkTriple["configured"] as? UInt32, 2_048);
        let fullAttentionTriple: [String: Any] = try self.journeySupport.requireObject(
            chunking["full_attention_key_value_growth_tokens"], failure: .missingConfigurationTriple);
        XCTAssertEqual(fullAttentionTriple["configured"] as? UInt32, nil);
        XCTAssertEqual(fullAttentionTriple["effective"] as? UInt32, 256);
        let generationIntervalTriple: [String: Any] = try self.journeySupport.requireObject(
            chunking["experimental_ssd_paging_generation_graph_submission_layer_interval"], failure: .missingConfigurationTriple);
        XCTAssertEqual(generationIntervalTriple["configured"] as? UInt32, nil);
        XCTAssertEqual(generationIntervalTriple["effective"] as? UInt32, 0);
        let blockTokensTriple: [String: Any] = try self.journeySupport.requireObject(
            chunking["prompt_cache_block_tokens"], failure: .missingConfigurationTriple);
        XCTAssertEqual(blockTokensTriple["is_configured"] as? Bool, false);
        XCTAssertEqual(blockTokensTriple["configured"] as? UInt32, nil);
        XCTAssertEqual(blockTokensTriple["effective"] as? UInt32, 128);

        let statusText: String = try self.journeySupport.requireResponseText(RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /v1/status HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"));
        XCTAssertFalse(statusText.contains("/fictional/private"), "no model directory may leak into the status document");
        XCTAssertFalse(statusText.contains("prompt-cache"), "no cache location may leak into the status document");
        server.stop();
    }

    func testStatusJourneySurfacesDiagnosticsUnmatchedIdsAndMemoryObservations() throws {
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
        XCTAssertEqual(statusCode, 200);
        XCTAssertEqual(envelope["configured_maximum_mlx_memory_gb"] as? UInt64, 90);
        XCTAssertEqual(envelope["mlx_memory_ceiling_bytes"] as? UInt64, 90_000_000_000);
        XCTAssertEqual(envelope["machine_mlx_memory_ceiling_bytes"] as? UInt64, 128_000_000_000);
        XCTAssertEqual(envelope["minimum_mlx_memory_ceiling_bytes"] as? UInt64, 8_000_000_000);
        XCTAssertEqual(envelope["pending_mlx_memory_ceiling_bytes"] as? UInt64, 64_000_000_000);
        XCTAssertEqual(envelope["mlx_memory_limit_error"] as? String, "the authored ceiling cannot hold the resident model");
        XCTAssertEqual(envelope["expert_memory_mode"] as? String, "paged");
        let expertResidency: [String: Any] = try self.journeySupport.requireObject(
            envelope["expert_residency"], failure: .missingExpertResidency);
        XCTAssertEqual(expertResidency["total_layer_count"] as? UInt32, 80);
        XCTAssertEqual(expertResidency["resident_expert_count"] as? UInt32, 20);
        XCTAssertEqual(expertResidency["resident_expert_payload_bytes"] as? UInt64, 5_000_000_000);
        let mlxSnapshot: [String: Any] = try self.journeySupport.requireObject(
            envelope["mlx_memory_snapshot"], failure: .missingMlxSnapshot);
        XCTAssertEqual(mlxSnapshot["source"] as? String, "idle_poll");
        XCTAssertEqual(mlxSnapshot["active_memory_bytes"] as? UInt64, 1_000_000_000);

        let configuration: [String: Any] = try self.journeySupport.requireObject(
            envelope["configuration"], failure: .missingConfigurationSection);
        XCTAssertEqual(configuration["is_effective"] as? Bool, false);
        // A pending memory-ceiling change is mid-flight, so the stale worker
        // generation must not demand a restart on top of it.
        XCTAssertEqual(configuration["restart_required"] as? Bool, false);

        let memory: [String: Any] = try self.journeySupport.requireObject(
            configuration["memory"], failure: .missingMemorySummary);
        XCTAssertEqual(memory["configured_maximum_bytes"] as? UInt64, 90_000_000_000);
        XCTAssertEqual(memory["effective_maximum_bytes"] as? UInt64, 90_000_000_000);
        XCTAssertEqual(memory["pending_maximum_bytes"] as? UInt64, 64_000_000_000);
        XCTAssertEqual(memory["error"] as? String, "the authored ceiling cannot hold the resident model");

        let diagnostics: Array<Any> = try self.journeySupport.requireArray(
            configuration["model_discovery_diagnostics"], failure: .missingDiagnostics);
        XCTAssertEqual(diagnostics.count, 1);
        let diagnostic: [String: Any] = try self.journeySupport.requireObject(
            diagnostics[0], failure: .missingDiagnostics);
        XCTAssertEqual(diagnostic["code"] as? String, "ambiguous_model_identity");
        XCTAssertEqual(diagnostic["model_id"] as? String, "duplicated-model");
        XCTAssertEqual(diagnostic["configured_root_numbers"] as? [Int], [1, 2]);

        XCTAssertEqual(configuration["unmatched_model_config_ids"] as? [String], ["vendor/unmatched-model"]);
        server.stop();
    }

    func testStatusJourneySurfacesTheUnavailableDirectoryDiagnosticWithoutPaths() throws {
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
        XCTAssertEqual(statusCode, 200);
        let configuration: [String: Any] = try self.journeySupport.requireObject(
            envelope["configuration"], failure: .missingConfigurationSection);
        let diagnostics: Array<Any> = try self.journeySupport.requireArray(
            configuration["model_discovery_diagnostics"], failure: .missingDiagnostics);
        XCTAssertEqual(diagnostics.count, 1);
        let diagnostic: [String: Any] = try self.journeySupport.requireObject(
            diagnostics[0], failure: .missingDiagnostics);
        XCTAssertEqual(diagnostic["code"] as? String, "unavailable_model_directory");
        XCTAssertEqual(diagnostic["model_id"] as? String, "");
        XCTAssertEqual(diagnostic["configured_root_numbers"] as? [Int], [4]);

        let statusText: String = try self.journeySupport.requireResponseText(RawLoopbackHttpClient.exchange(
            port: server.boundEndpoint.port,
            requestText: "GET /v1/status HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"));
        XCTAssertFalse(statusText.contains("/models/synthetic-chat-model"), "no model directory may leak into the status document");
        server.stop();
    }

    func testStatusJourneySuppressesRestartWhileAValidationFailureStands() throws {
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
        XCTAssertEqual(statusCode, 200);
        let configuration: [String: Any] = try self.journeySupport.requireObject(
            envelope["configuration"], failure: .missingConfigurationSection);
        XCTAssertEqual(configuration["validation_error"] as? String, "the configuration document was rejected: unknown field");
        XCTAssertEqual(configuration["is_effective"] as? Bool, false);
        XCTAssertEqual(configuration["restart_required"] as? Bool, false);
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
