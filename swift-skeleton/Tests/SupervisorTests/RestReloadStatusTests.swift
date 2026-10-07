import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;

@testable import Supervisor;

/**
 * Reload status journeys, migrating
 * apps/supervisor/tests/rest_api/config_reload/reload_status.rs: a prompt
 * cache policy change reaches the worker through transactional replacement,
 * restart-required verdicts persist until a real restart, a mixed reload
 * never reports the memory-only state effective, worker and logging fields
 * hold still while an application restart is pending, and the discovery
 * snapshot one reload installs is the same one the listing and routing
 * surfaces serve.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class RestReloadStatusTests {

    /// A prompt-cache policy change replaces the worker transactionally and
    /// the status triple reports the applied configuration.
    @Test
    func should_apply_prompt_cache_policy_after_config_file_reload() throws {
        let journey: ReloadLiveWorkerJourney = try ReloadLiveWorkerJourney.launch(
            startupConfiguration: nil,
            modelPolicyCatalog: Dictionary(),
            fixtureAcknowledgedGeneration: nil,
            reloadableConfigOverride: { (resolvedConfig: inout ResolvedRuntimeConfig) -> Void in
                resolvedConfig.persistentPromptCacheEnabled = false;
                resolvedConfig.configuredPersistentPromptCacheEnabled = false;
            });
        defer { journey.dispose() }
        // The replacement candidate must acknowledge the resolver-derived
        // candidate generation; the journey pins it before the reload.
        journey.pokeCurrentGeneration(journey.candidateGeneration);

        let reloadResponse: RestHttpResponse = try journey.postConfigReload();

        let reloadDocument: [String: Any] = try ConfigReloadJourney.decodeObject(reloadResponse);
        #expect(reloadResponse.statusCode == 200, Comment(rawValue: String(decoding: reloadResponse.bodyBytes, as: UTF8.self)));
        #expect(reloadDocument["status"] as? String == "reloaded");
        #expect(reloadDocument["worker_restart_completed"] as? Bool == true);
        #expect(reloadDocument["candidate_generation"] as? String == journey.candidateGeneration);
        #expect(reloadDocument["effective_generation"] as? String == journey.candidateGeneration);
        let acknowledgedConfiguration: [String: Any] = try RestReloadStatusTests.requireObject(
            reloadDocument, field: "worker_runtime_feature_configuration");
        #expect(
            acknowledgedConfiguration["configuration_generation"] as? String
                == journey.candidateGeneration);
        #expect(acknowledgedConfiguration["persistent_prompt_cache_enabled"] as? Bool == true);

        let statusDocument: [String: Any] = try journey.getStatusDocument();
        #expect(statusDocument["configured_generation"] as? String == journey.candidateGeneration);
        #expect(statusDocument["effective_generation"] as? String == journey.candidateGeneration);
        #expect(statusDocument["worker_runtime_feature_configuration_applied"] as? Bool == true);
        let statusConfiguration: [String: Any] = try RestReloadStatusTests.requireObject(
            statusDocument, field: "worker_runtime_feature_configuration");
        #expect(statusConfiguration["persistent_prompt_cache_enabled"] as? Bool == true);
        #expect(journey.transitionState.currentReloadableConfig().persistentPromptCacheEnabled);
    }

    /// Restart-required verdicts persist across repeated reloads until the
    /// listener is actually restarted.
    @Test
    func should_keep_reporting_restart_required_until_server_is_restarted() throws {
        let journey: ReloadStatusJourney = try ReloadStatusJourney.launch();
        defer { journey.dispose() }
        ConfigReloadJourney.writeConfigFile(
            journey.homeDirectoryUrl,
            configuredFieldsJson: "{\"diagnostics\":{\"log_level\":\"info\"}}");

        for reloadAttemptNumber: Int in 1...2 {
            let reloadResponse: RestHttpResponse = try journey.postConfigReload();
            let reloadDocument: [String: Any] = try ConfigReloadJourney.decodeObject(reloadResponse);
            #expect(
                reloadDocument["status"] as? String == "restart_required",
                Comment(rawValue: "reload attempt \(reloadAttemptNumber) must remain restart-required "
                    + "until the listener is actually restarted"));
        }
    }

    /// A mixed reload that applied only memory must not report the effective
    /// state as the configured candidate.
    @Test
    func should_not_report_a_mixed_reload_effective_when_only_memory_applied() throws {
        let initialResolvedConfig: ResolvedRuntimeConfig = try RestChatJourneySupport.makeResolvedConfig();
        let journey: ReloadLiveWorkerJourney = try ReloadLiveWorkerJourney.launch(
            startupConfiguration: initialResolvedConfig.workerStartupConfiguration(),
            modelPolicyCatalog: initialResolvedConfig.modelPolicyCatalog);
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

        let statusDocument: [String: Any] = try journey.getStatusDocument();
        let configurationDocument: [String: Any] = try RestReloadStatusTests.requireObject(
            statusDocument, field: "configuration");
        #expect(configurationDocument["is_effective"] as? Bool == false);
        #expect(configurationDocument["restart_required"] as? Bool == true);
        #expect(
            configurationDocument["resolved_generation"] as? String
                != configurationDocument["configured_generation"] as? String);
        #expect(
            configurationDocument["effective_generation"] as? String
                == configurationDocument["resolved_generation"] as? String);
        #expect(
            configurationDocument["effective_generation"] as? String
                != configurationDocument["configured_generation"] as? String);
    }

    /// Worker and logging fields hold still while an application restart is
    /// required; only the logging verdict is reported.
    @Test
    func should_keep_worker_and_logging_fields_unchanged_when_application_restart_is_required() throws {
        let journey: ReloadStatusJourney = try ReloadStatusJourney.launch();
        defer { journey.dispose() }
        ConfigReloadJourney.writeConfigFile(
            journey.homeDirectoryUrl,
            configuredFieldsJson: "{\"chunking\":{\"fixed_prompt_processing_chunk_size_tokens\":4096},"
                + "\"diagnostics\":{\"log_level\":\"info\"}}");

        let reloadResponse: RestHttpResponse = try journey.postConfigReload();

        #expect(reloadResponse.statusCode == 200);
        let reloadDocument: [String: Any] = try ConfigReloadJourney.decodeObject(reloadResponse);
        #expect(reloadDocument["status"] as? String == "restart_required");
        #expect(reloadDocument["reloaded_fields"] as? [String] == []);
        #expect(reloadDocument["restart_required_fields"] as? [String] == ["logging"]);
        let liveConfig: ResolvedRuntimeConfig = journey.transitionState.currentReloadableConfig();
        #expect(liveConfig.loggingConfig.level == LogLevel.warn);
        #expect(liveConfig.bindAddress == journey.resolverDefaultBindAddress);
    }

    /// Every discovered model stays listed and routable regardless of
    /// configured policies.
    @Test
    func should_keep_all_discovered_models_listed_and_routable() throws {
        let journey: DiscoveryServingJourney = try DiscoveryServingJourney.launch(
            discoveredModelIds: [
                RestReloadStatusTests.configuredTargetModelId,
                RestReloadStatusTests.unconfiguredModelId,
            ],
            policyModelId: RestReloadStatusTests.configuredTargetModelId);
        defer { journey.dispose() }

        let modelListText: String = try journey.getModelListText();
        #expect(modelListText.contains(RestReloadStatusTests.configuredTargetModelId));
        #expect(modelListText.contains(RestReloadStatusTests.unconfiguredModelId));

        for modelId: String in [
            RestReloadStatusTests.configuredTargetModelId,
            RestReloadStatusTests.unconfiguredModelId,
        ] {
            let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
                routeTable: journey.routeTable,
                requestBody: "{\"model\":\"\(modelId)\",\"messages\":[{\"role\":\"user\","
                    + "\"content\":\"hello\"}],\"stream\":true}");
            #expect(chatResponse.statusCode == 200);
        }
        let receivedCommands: Array<ChatGenerationCommand> = journey.executor.receivedCommands;
        #expect(receivedCommands.count == 2);
        #expect(receivedCommands[0].model == RestReloadStatusTests.configuredTargetModelId);
        #expect(receivedCommands[1].model == RestReloadStatusTests.unconfiguredModelId);
    }

    /// The reloaded discovery snapshot reaches listing and routing together.
    @Test
    func should_use_the_same_reloaded_discovery_snapshot_for_listing_and_routing() throws {
        let journey: DiscoveryServingJourney = try DiscoveryServingJourney.launch(
            discoveredModelIds: [],
            policyModelId: nil);
        defer { journey.dispose() }
        var reloadedConfig: ResolvedRuntimeConfig =
            journey.transitionState.currentReloadableConfig();
        reloadedConfig.discoveredModels = [
            DiscoveryServingJourney.discoveredChatModel(
                modelId: RestReloadStatusTests.reloadedModelId,
                modelFamily: .qwen35),
        ];
        journey.transitionState.replaceReloadableConfig(reloadedConfig);

        let modelListText: String = try journey.getModelListText();
        #expect(modelListText.contains(RestReloadStatusTests.reloadedModelId));

        let generationResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: journey.routeTable,
            requestBody: "{\"model\":\"\(RestReloadStatusTests.reloadedModelId)\","
                + "\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":true}");
        #expect(generationResponse.statusCode == 200);
        let receivedCommands: Array<ChatGenerationCommand> = journey.executor.receivedCommands;
        #expect(receivedCommands.count == 1);
        #expect(receivedCommands[0].model == RestReloadStatusTests.reloadedModelId);
    }

    /// A worker-restart reload without worker control fails without
    /// mutating the live state and the instance still reports ready.
    @Test
    func should_reject_prompt_cache_reload_when_worker_replacement_is_unavailable() throws {
        let journey: ReloadStatusJourney = try ReloadStatusJourney.launch();
        defer { journey.dispose() }
        ConfigReloadJourney.writeConfigFile(
            journey.homeDirectoryUrl,
            configuredFieldsJson: "{\"prompt_cache\":{\"enabled\":false}}");

        let reloadResponse: RestHttpResponse = try journey.postConfigReload();

        #expect(reloadResponse.statusCode == 500);
        let reloadDocument: [String: Any] = try ConfigReloadJourney.decodeObject(reloadResponse);
        #expect(reloadDocument["worker_restart_completed"] as? Bool == false);
        #expect(reloadDocument["reloaded_fields"] as? [String] == []);
        #expect(reloadDocument["status"] as? String == "failed");
        let statusDocument: [String: Any] = try journey.getStatusDocument();
        #expect(statusDocument["status"] as? String == "ready");
    }

    static let configuredTargetModelId: String = "astronomical/application-test-model";
    static let unconfiguredModelId: String = "astronomical/another-test-model";
    static let reloadedModelId: String = "astronomical/reloaded-model";

    private static func requireObject(
        _ document: [String: Any],
        field: String
    ) throws -> [String: Any] {
        guard let nestedObject: [String: Any] = document[field] as? [String: Any] else {
            throw ReloadStatusJourneyFailure.objectMissing(field);
        }
        return nestedObject;
    }
}

/// Typed failures of the reload-status journeys.
enum ReloadStatusJourneyFailure: Error {

    case objectMissing(String);
}

/// One no-worker reload-status journey: a resolver config home with the live
/// snapshot resolved from the authored defaults and the reload route wired
/// without worker control.
final class ReloadStatusJourney {

    let transitionState: ConfigTransitionState;
    let homeDirectoryUrl: URL;
    let resolverDefaultBindAddress: String;
    private let routeTable: RestRouteTable;

    private init(
        transitionState: ConfigTransitionState,
        homeDirectoryUrl: URL,
        resolverDefaultBindAddress: String,
        routeTable: RestRouteTable
    ) {
        self.transitionState = transitionState;
        self.homeDirectoryUrl = homeDirectoryUrl;
        self.resolverDefaultBindAddress = resolverDefaultBindAddress;
        self.routeTable = routeTable;
    }

    static func launch() throws -> ReloadStatusJourney {
        let homeDirectoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("astronomical-reload-status-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(at: homeDirectoryUrl, withIntermediateDirectories: true);
        ConfigReloadJourney.writeConfigFile(homeDirectoryUrl, configuredFieldsJson: "{}");
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            FilePath(string: homeDirectoryUrl.path),
            runtimeInstance: AstronomicalRuntimeInstance.development);
        let resolver: ResolvedRuntimeConfigResolver = ResolvedRuntimeConfigResolver(
            instancePaths: instancePaths,
            fallbackWorkerExecutablePath: try RestChatJourneySupport.makeResolvedConfig().workerExecutablePath);
        let initialResolvedConfig: ResolvedRuntimeConfig = try resolver.load();
        let transitionState: ConfigTransitionState = ConfigTransitionState(
            reloadableConfig: initialResolvedConfig,
            configuredConfigSnapshot: initialResolvedConfig);
        let workerHealthState: WorkerHealthState = WorkerHealthState();
        workerHealthState.publish(WorkerHealthSnapshot.readyWithoutModel(
            machineMlxMemoryCeilingBytes: 40_000_000_000,
            effectiveMlxMemoryCeilingBytes: 8_000_000_000,
            minimumMlxMemoryCeilingBytes: 1));
        let routeTable: RestRouteTable = RestEndpointRoutes.servingRouteTable(
            resolvedRuntimeConfig: initialResolvedConfig,
            workerHealthState: workerHealthState,
            instancePaths: instancePaths,
            buildIdentity: RestChatJourneySupport.journeyBuildIdentity(),
            configReloadContext: RestConfigReloadRouteContext(
                transitionState: transitionState,
                runtimeConfigResolver: resolver,
                workerControl: nil,
                workerHealthState: WorkerHealthState(),
                generationActivityIdleProvider: { return true }));
        return ReloadStatusJourney(
            transitionState: transitionState,
            homeDirectoryUrl: homeDirectoryUrl,
            resolverDefaultBindAddress: initialResolvedConfig.bindAddress,
            routeTable: routeTable);
    }

    func dispose() -> Void {
        try? FileManager.default.removeItem(atPath: self.homeDirectoryUrl.path);
    }

    func postConfigReload() throws -> RestHttpResponse {
        let routeOutcome: RestRouteOutcome = self.routeTable.outcome(
            method: RestConfigReloadEndpoint.routeMethod,
            path: RestConfigReloadEndpoint.routePath);
        guard case .handler(let routeHandler) = routeOutcome else {
            throw ReloadStatusJourneyFailure.objectMissing("config reload route");
        }
        return try routeHandler(WorkerReplacementJourney.emptyRequest(
            method: RestConfigReloadEndpoint.routeMethod,
            path: RestConfigReloadEndpoint.routePath));
    }

    func getStatusDocument() throws -> [String: Any] {
        let routeOutcome: RestRouteOutcome = self.routeTable.outcome(method: "GET", path: "/v1/status");
        guard case .handler(let routeHandler) = routeOutcome else {
            throw ReloadStatusJourneyFailure.objectMissing("status route");
        }
        let statusResponse: RestHttpResponse = try routeHandler(WorkerReplacementJourney.emptyRequest(
            method: "GET",
            path: "/v1/status"));
        return try ConfigReloadJourney.decodeObject(statusResponse);
    }
}

/// One discovery-serving journey: a scripted chat executor, a resolved
/// configuration with the given discovered models and policies, and the live
/// snapshot provider wired to the transition state like the daemon does.
final class DiscoveryServingJourney {

    let executor: ScriptedChatExecutor;
    let routeTable: RestRouteTable;
    let transitionState: ConfigTransitionState;
    let homeDirectoryUrl: URL;

    private init(
        executor: ScriptedChatExecutor,
        routeTable: RestRouteTable,
        transitionState: ConfigTransitionState,
        homeDirectoryUrl: URL
    ) {
        self.executor = executor;
        self.routeTable = routeTable;
        self.transitionState = transitionState;
        self.homeDirectoryUrl = homeDirectoryUrl;
    }

    static func launch(
        discoveredModelIds: Array<String>,
        policyModelId: String?
    ) throws -> DiscoveryServingJourney {
        let homeDirectoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("astronomical-discovery-serving-\(UUID().uuidString)", isDirectory: true);
        try FileManager.default.createDirectory(at: homeDirectoryUrl, withIntermediateDirectories: true);
        let instancePaths: AstronomicalInstancePaths = AstronomicalInstancePaths.forHomeDirectory(
            FilePath(string: homeDirectoryUrl.path),
            runtimeInstance: AstronomicalRuntimeInstance.development);
        var resolvedConfig: ResolvedRuntimeConfig = try RestChatJourneySupport.makeResolvedConfig();
        resolvedConfig.discoveredModels = discoveredModelIds.map({ (modelId: String) -> DiscoveryDiscoveredModel in
            return DiscoveryServingJourney.discoveredChatModel(modelId: modelId, modelFamily: .qwen35);
        });
        var modelPolicyCatalog: Dictionary<String, RuntimeModelPolicy> = Dictionary();
        if let policyModelId = policyModelId {
            modelPolicyCatalog[policyModelId] = FakeWorkerJourneyHarness.modelPolicy(modelId: policyModelId);
        }
        resolvedConfig.modelPolicyCatalog = modelPolicyCatalog;
        let transitionState: ConfigTransitionState = ConfigTransitionState(
            reloadableConfig: resolvedConfig,
            configuredConfigSnapshot: resolvedConfig);
        let executor: ScriptedChatExecutor = ScriptedChatExecutor(
            healthSnapshot: WorkerHealthSnapshot.readyWithModel(
                modelId: RestReloadStatusTests.configuredTargetModelId,
                capabilities: RestChatJourneySupport.readyChatCapabilities()),
            streamEvents: [
                .completed(
                    promptTokenCount: 1,
                    generatedTokenCount: 1,
                    reasoningTokenCount: 0,
                    cachedTokenCount: 0,
                    reason: .endOfSequence),
            ]);
        let routeTable: RestRouteTable = RestEndpointRoutes.servingRouteTable(
            resolvedRuntimeConfig: resolvedConfig,
            workerHealthState: WorkerHealthState(),
            instancePaths: instancePaths,
            buildIdentity: RestChatJourneySupport.journeyBuildIdentity(),
            chatContext: RestChatRouteContext(
                chatExecutor: executor,
                requestIdAllocator: ChatRequestIdAllocator(),
                resolvedRuntimeConfig: resolvedConfig,
                instancePaths: instancePaths,
                liveResolvedRuntimeConfigProvider: {
                    return transitionState.currentReloadableConfig();
                }),
            configReloadContext: RestConfigReloadRouteContext(
                transitionState: transitionState,
                runtimeConfigResolver: ResolvedRuntimeConfigResolver(
                    instancePaths: instancePaths,
                    fallbackWorkerExecutablePath: try RestChatJourneySupport.makeResolvedConfig().workerExecutablePath),
                workerControl: nil,
                workerHealthState: WorkerHealthState(),
                generationActivityIdleProvider: { return true }));
        return DiscoveryServingJourney(
            executor: executor,
            routeTable: routeTable,
            transitionState: transitionState,
            homeDirectoryUrl: homeDirectoryUrl);
    }

    func dispose() -> Void {
        try? FileManager.default.removeItem(atPath: self.homeDirectoryUrl.path);
    }

    func getModelListText() throws -> String {
        let routeOutcome: RestRouteOutcome = self.routeTable.outcome(method: "GET", path: "/v1/models");
        guard case .handler(let routeHandler) = routeOutcome else {
            throw ReloadStatusJourneyFailure.objectMissing("models route");
        }
        let modelListResponse: RestHttpResponse = try routeHandler(WorkerReplacementJourney.emptyRequest(
            method: "GET",
            path: "/v1/models"));
        return String(decoding: modelListResponse.bodyBytes, as: UTF8.self);
    }

    static func discoveredChatModel(
        modelId: String,
        modelFamily: ModelFamily
    ) -> DiscoveryDiscoveredModel {
        return DiscoveryDiscoveredModel(
            modelId: modelId,
            providerModelId: nil,
            modelFamily: modelFamily,
            revision: "test-revision",
            modelDirectory: FilePath(string: "/fictional/models/\(modelId)"),
            capabilities: .chat(DiscoveryChatModelCapabilities(
                contextWindowTokens: 2_048,
                maximumInputTokens: 1_024,
                maximumOutputTokens: 128,
                supportsVision: false,
                supportsReasoning: true,
                supportsToolCalls: true)),
            license: nil,
            modelSizeBytes: 0);
    }
}
