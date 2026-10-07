import Foundation;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;

/**
 * The context the daemon hands the reload route: the shared transition
 * state, the config resolver, the live worker control when one exists, the
 * worker health state, and the executor-side activity probe used while no
 * worker control exists, or nil when reload is unsupported and the route
 * must advertise itself as absent.
 */
public struct RestConfigReloadRouteContext: @unchecked Sendable {

    let transitionState: ConfigTransitionState;
    let runtimeConfigResolver: ResolvedRuntimeConfigResolver;
    let workerControl: WorkerSupervisor?;
    let workerHealthState: WorkerHealthState;
    let generationActivityIdleProvider: @Sendable () -> Bool;

    public init(
        transitionState: ConfigTransitionState,
        runtimeConfigResolver: ResolvedRuntimeConfigResolver,
        workerControl: WorkerSupervisor?,
        workerHealthState: WorkerHealthState,
        generationActivityIdleProvider: @escaping @Sendable () -> Bool
    ) {
        self.transitionState = transitionState;
        self.runtimeConfigResolver = runtimeConfigResolver;
        self.workerControl = workerControl;
        self.workerHealthState = workerHealthState;
        self.generationActivityIdleProvider = generationActivityIdleProvider;
    }
}

/// Reloads config without cancelling active or queued generation work,
/// migrating apps/supervisor/src/config_reload_endpoint.rs:
/// POST /v1/config/reload answers 200 reloaded, 400 invalid_config with
/// path-safe feedback, 409 while a generation or memory transition is in
/// flight, and 500 when the worker rejects the reload or cannot be replaced.
public enum RestConfigReloadEndpoint {

    public static let routeMethod: String = "POST";
    public static let routePath: String = "/v1/config/reload";

    public static func handle(
        _ request: RestHttpRequest,
        reloadContext: RestConfigReloadRouteContext?
    ) -> RestHttpResponse {
        guard let reloadContext = reloadContext else {
            return RestHttpResponse.text(statusCode: 404, body: "reload not supported");
        }
        return reloadContext.transitionState.withTransitionGuard({ () -> RestHttpResponse in
            return RestConfigReloadEndpoint.reloadConfig(reloadContext: reloadContext);
        });
    }

    private static func reloadConfig(
        reloadContext: RestConfigReloadRouteContext
    ) -> RestHttpResponse {
        if reloadContext.transitionState.currentPendingMemoryConfigGeneration() != nil {
            return RestConfigReloadEndpoint.jsonResponse(409, .busy());
        }
        if reloadContext.workerControl == nil
            && !reloadContext.generationActivityIdleProvider() {
            return RestConfigReloadEndpoint.jsonResponse(409, .busy());
        }

        let candidateResolvedConfig: ResolvedRuntimeConfig;
        do {
            candidateResolvedConfig = try reloadContext.runtimeConfigResolver.load();
        } catch {
            let validationMessage: String =
                "Configuration is invalid; correct the local configuration file and retry";
            reloadContext.transitionState.setConfigurationValidationError(validationMessage);
            return RestConfigReloadEndpoint.jsonResponse(
                400, .invalidConfig(validationMessage));
        }
        reloadContext.transitionState.setConfigurationValidationError(nil);
        reloadContext.transitionState.replaceConfiguredConfigSnapshot(candidateResolvedConfig);
        let candidateGeneration: String = candidateResolvedConfig.configurationGeneration;
        if let discoveryDiagnostic: DiscoveryModelDiscoveryDiagnostic = candidateResolvedConfig.modelDiscoveryDiagnostics
            .first(where: { (diagnostic: DiscoveryModelDiscoveryDiagnostic) -> Bool in
                return diagnostic.code == DiscoveryModelDiscoveryDiagnosticCode.ambiguousModelIdentity;
            }) {
            let configuredRootNumbers: String = diagnosticRootNumbers(diagnostic: discoveryDiagnostic);
            return RestConfigReloadEndpoint.jsonResponse(
                400,
                .invalidConfig(
                    "Model '\(discoveryDiagnostic.modelId)' appears in model_directories entries "
                        + "\(configuredRootNumbers); remove one duplicate root and retry")
                    .withGenerations(
                        candidate: candidateGeneration,
                        effective: RestConfigReloadEndpoint.effectiveWorkerGeneration(reloadContext)));
        }
        let currentResolvedConfig: ResolvedRuntimeConfig = reloadContext.transitionState.currentReloadableConfig();
        let reloadDecision: ConfigReloadDecision = ConfigReloadDiff.compare(
            current: currentResolvedConfig,
            candidate: candidateResolvedConfig);
        let discoveredModelCount: Int = reloadDecision.discoveredModelCount;
        let reloadedFields: Array<String> = reloadDecision.reloadedFields;
        let memoryLimitChanged: Bool = currentResolvedConfig.maximumMlxMemoryBytes
            != candidateResolvedConfig.maximumMlxMemoryBytes;
        let memoryEffectiveGeneration: String;
        if memoryLimitChanged, case .restApiRestartRequired = reloadDecision {
            memoryEffectiveGeneration = ResolvedConfigurationGeneration.deriveMemoryOnlyTransition(
                priorResolvedGeneration: currentResolvedConfig.configurationGeneration,
                maximumMlxMemoryBytes: candidateResolvedConfig.maximumMlxMemoryBytes);
        } else {
            memoryEffectiveGeneration = candidateGeneration;
        }
        let isGenerationBusy: Bool;
        if let workerControl = reloadContext.workerControl {
            isGenerationBusy = !workerControl.isGenerationIdleForControlAction();
        } else {
            isGenerationBusy = !reloadContext.generationActivityIdleProvider();
        }
        if isGenerationBusy && reloadDecision.workerRestartRequired {
            return RestConfigReloadEndpoint.jsonResponse(
                409,
                .busy().withGenerations(
                    candidate: candidateGeneration,
                    effective: RestConfigReloadEndpoint.effectiveWorkerGeneration(reloadContext)));
        }
        if memoryLimitChanged && !reloadDecision.workerRestartRequired,
            let workerControl = reloadContext.workerControl {
            let memoryUpdateResponse: RestHttpResponse? = RestConfigReloadEndpoint.reloadLiveMemoryLimit(
                memoryEffectiveGeneration: memoryEffectiveGeneration,
                candidateResolvedConfig: candidateResolvedConfig,
                currentResolvedConfig: currentResolvedConfig,
                candidateGeneration: candidateGeneration,
                discoveredModelCount: discoveredModelCount,
                workerControl: workerControl,
                reloadContext: reloadContext);
            if let memoryUpdateResponse = memoryUpdateResponse {
                return memoryUpdateResponse;
            }
        }

        switch (reloadDecision) {
        case .noWorkerRestart:
            reloadContext.transitionState.replaceReloadableConfig(candidateResolvedConfig);
            return RestConfigReloadEndpoint.jsonResponse(
                200,
                .reloaded(
                    reloadedFields: reloadedFields,
                    discoveredModelCount: discoveredModelCount)
                    .withGenerations(
                        candidate: candidateGeneration,
                        effective: RestConfigReloadEndpoint.effectiveWorkerGeneration(reloadContext)));
        case .restApiRestartRequired(_, let restartRequiredFields, _):
            var liveConfig: ResolvedRuntimeConfig = reloadContext.transitionState.currentReloadableConfig();
            // Copy only fields that are safe to apply without a worker or
            // REST restart. Worker and listener settings remain live-old
            // until the requested restart succeeds.
            liveConfig.maximumMlxMemoryBytes = candidateResolvedConfig.maximumMlxMemoryBytes;
            if memoryLimitChanged {
                liveConfig.configurationGeneration = memoryEffectiveGeneration;
            }
            reloadContext.transitionState.replaceReloadableConfig(liveConfig);
            return RestConfigReloadEndpoint.jsonResponse(
                200,
                .restartRequired(
                    reloadedFields: reloadedFields,
                    restartRequiredFields: restartRequiredFields,
                    discoveredModelCount: discoveredModelCount)
                    .withGenerations(
                        candidate: candidateGeneration,
                        effective: RestConfigReloadEndpoint.effectiveWorkerGeneration(reloadContext)));
        case .restartWorker:
            return RestConfigReloadEndpoint.restartServingWorker(
                reloadContext: reloadContext,
                candidateResolvedConfig: candidateResolvedConfig,
                reloadedFields: reloadedFields,
                discoveredModelCount: discoveredModelCount);
        }
    }

    /// Replaces the serving worker with a candidate resolved from the reload
    /// candidate, mirroring config_reload_endpoint.rs's restart_worker: the
    /// acknowledged runtime configuration becomes the new effective
    /// generation, a busy race stays 409, and any other replacement failure
    /// keeps the previous worker serving untouched.
    private static func restartServingWorker(
        reloadContext: RestConfigReloadRouteContext,
        candidateResolvedConfig: ResolvedRuntimeConfig,
        reloadedFields: Array<String>,
        discoveredModelCount: Int
    ) -> RestHttpResponse {
        let candidateGeneration: String = candidateResolvedConfig.configurationGeneration;
        guard let workerControl: WorkerSupervisor = reloadContext.workerControl else {
            return RestConfigReloadEndpoint.jsonResponse(
                500,
                .failed(
                    "Config reload cannot replace this application's worker",
                    discoveredModelCount: discoveredModelCount)
                    .withGenerations(
                        candidate: candidateGeneration,
                        effective: RestConfigReloadEndpoint.effectiveWorkerGeneration(reloadContext)));
        }
        let acknowledgedConfiguration: WorkerRuntimeFeatureConfiguration;
        do {
            acknowledgedConfiguration = try workerControl.restartWorkerWithStartupConfiguration(
                candidateWorkerExecutablePath: candidateResolvedConfig.workerExecutablePath.string,
                candidateWorkerArguments: Array<String>(),
                candidateModelPolicyCatalog: candidateResolvedConfig.modelPolicyCatalog,
                candidateStartupConfiguration: candidateResolvedConfig.workerStartupConfiguration());
        } catch WorkerControlError.generationBusy {
            return RestConfigReloadEndpoint.jsonResponse(409, .busy());
        } catch {
            return RestConfigReloadEndpoint.jsonResponse(
                500,
                .failed(
                    "Config was valid, but worker replacement failed; inspect local diagnostics and retry",
                    discoveredModelCount: discoveredModelCount)
                    .withGenerations(
                        candidate: candidateGeneration,
                        effective: RestConfigReloadEndpoint.effectiveWorkerGeneration(reloadContext)));
        }
        reloadContext.transitionState.replaceReloadableConfig(candidateResolvedConfig);
        return RestConfigReloadEndpoint.jsonResponse(
            200,
            .workerRestartCompleted(
                reloadedFields: reloadedFields,
                discoveredModelCount: discoveredModelCount,
                acknowledgedConfiguration: acknowledgedConfiguration)
                .withGenerations(candidate: candidateGeneration, effective: candidateGeneration));
    }

    /// Applies the memory-only slice of a reload against the live worker.
    /// Returns nil when the reload may continue; otherwise the response the
    /// endpoint must return for the memory outcome.
    private static func reloadLiveMemoryLimit(
        memoryEffectiveGeneration: String,
        candidateResolvedConfig: ResolvedRuntimeConfig,
        currentResolvedConfig: ResolvedRuntimeConfig,
        candidateGeneration: String,
        discoveredModelCount: Int,
        workerControl: WorkerSupervisor,
        reloadContext: RestConfigReloadRouteContext
    ) -> RestHttpResponse? {
        let workerHealthSnapshot: WorkerHealthSnapshot = workerControl.workerHealthSnapshot();
        let effectiveMlxMemoryCeilingBytes: UInt64 = candidateResolvedConfig.maximumMlxMemoryBytes
            ?? workerHealthSnapshot.machineMlxMemoryCeilingBytes;
        if effectiveMlxMemoryCeilingBytes == 0
            || effectiveMlxMemoryCeilingBytes > workerHealthSnapshot.machineMlxMemoryCeilingBytes
            || effectiveMlxMemoryCeilingBytes < workerHealthSnapshot.minimumMlxMemoryCeilingBytes {
            return RestConfigReloadEndpoint.jsonResponse(
                400,
                .invalidConfig("maximum_mlx_memory_gb is outside the worker's reported limits")
                    .withGenerations(
                        candidate: candidateGeneration,
                        effective: RestConfigReloadEndpoint.effectiveWorkerGeneration(reloadContext)));
        }
        reloadContext.transitionState.setPendingMemoryConfigGeneration(memoryEffectiveGeneration);
        workerControl.stageMemoryConfigurationGeneration(memoryEffectiveGeneration);
        // A restart-required candidate is not wholly effective yet. Sending
        // its full generation would make the worker's next model
        // acknowledgement disagree with the supervisor's memory-only live
        // configuration.
        let updateOutcomeResult: Result<MlxMemoryLimitUpdateOutcome, Error>;
        do {
            updateOutcomeResult = .success(try workerControl.updateMlxMemoryLimit(
                effectiveMlxMemoryCeilingBytes,
                configurationGeneration: memoryEffectiveGeneration));
        } catch {
            updateOutcomeResult = .failure(error);
        }
        switch (updateOutcomeResult) {
        case .success(let updateOutcome):
            if updateOutcome == .rejected {
                reloadContext.transitionState.setPendingMemoryConfigGeneration(nil);
                workerControl.recordMemoryConfigurationGeneration(
                    memoryEffectiveGeneration, .rejected);
                let rejectedHealthSnapshot: WorkerHealthSnapshot = workerControl.workerHealthSnapshot();
                return RestConfigReloadEndpoint.jsonResponse(
                    400,
                    .invalidConfig(
                        rejectedHealthSnapshot.mlxMemoryLimitError
                            ?? "worker rejected the MLX memory limit")
                        .withGenerations(
                            candidate: candidateGeneration,
                            effective: RestConfigReloadEndpoint.effectiveWorkerGeneration(reloadContext)));
            }
            workerControl.recordMemoryConfigurationGeneration(
                memoryEffectiveGeneration, updateOutcome);
            if updateOutcome == .queued {
                let reconcileSupervisor: WorkerSupervisor = workerControl;
                let reconcileTransitionState: ConfigTransitionState = reloadContext.transitionState;
                let reconcileThread: Thread = Thread {
                    QueuedMemoryReload.reconcileReloadedMemoryConfig(
                        supervisor: reconcileSupervisor,
                        transitionState: reconcileTransitionState,
                        effectiveMemoryGeneration: memoryEffectiveGeneration,
                        priorResolvedConfig: currentResolvedConfig);
                };
                reconcileThread.name = "queued-memory-reload-reconcile";
                reconcileThread.start();
            } else {
                reloadContext.transitionState.setPendingMemoryConfigGeneration(nil);
            }
            return nil;
        case .failure:
            reloadContext.transitionState.setPendingMemoryConfigGeneration(nil);
            workerControl.recordMemoryConfigurationGeneration(
                memoryEffectiveGeneration, .rejected);
            return RestConfigReloadEndpoint.jsonResponse(
                500,
                .failed(
                    "Could not apply the MLX memory limit; inspect local diagnostics and retry",
                    discoveredModelCount: discoveredModelCount)
                    .withGenerations(
                        candidate: candidateGeneration,
                        effective: RestConfigReloadEndpoint.effectiveWorkerGeneration(reloadContext)));
        }
    }

    private static func effectiveWorkerGeneration(
        _ reloadContext: RestConfigReloadRouteContext
    ) -> String? {
        return reloadContext.workerHealthState.currentSnapshot().workerRuntimeFeatureConfiguration?
            .configurationGeneration;
    }

    private static func diagnosticRootNumbers(diagnostic: DiscoveryModelDiscoveryDiagnostic) -> String {
        return diagnostic.configuredRootNumbers
            .map({ (configuredRootNumber: Int) -> String in
                return String(configuredRootNumber);
            })
            .joined(separator: ", ");
    }

    private static func jsonResponse(
        _ statusCode: Int,
        _ reloadResponse: RestConfigReloadResponse
    ) -> RestHttpResponse {
        guard let reloadHttpResponse: RestHttpResponse = try? RestHttpResponse.json(
            statusCode: statusCode,
            wireValue: reloadResponse.wireValue()) else {
            return RestHttpResponse.text(statusCode: statusCode, body: reloadResponse.messageText);
        }
        return reloadHttpResponse;
    }
}
