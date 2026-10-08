import Foundation;

import AstronomicalConfig;

/// Result of comparing the current resolved config with a candidate.
///
/// Migrates the decision half of apps/supervisor/src/config_reload.rs.
public enum ConfigReloadDecision: Equatable {

    /// All changed fields were applied in place without restarting the worker.
    case noWorkerRestart(reloadedFields: Array<String>, discoveredModelCount: Int);

    /// At least one worker-startup field changed; the worker must be replaced.
    case restartWorker(reloadedFields: Array<String>, discoveredModelCount: Int);

    /// At least one REST-API-only field changed; a full restart is required.
    case restApiRestartRequired(
        reloadedFields: Array<String>,
        restartRequiredFields: Array<String>,
        discoveredModelCount: Int);

    public var discoveredModelCount: Int {
        switch (self) {
        case .noWorkerRestart(_, let discoveredModelCount): return discoveredModelCount;
        case .restartWorker(_, let discoveredModelCount): return discoveredModelCount;
        case .restApiRestartRequired(_, _, let discoveredModelCount): return discoveredModelCount;
        }
    }

    public var reloadedFields: Array<String> {
        switch (self) {
        case .noWorkerRestart(let reloadedFields, _): return reloadedFields;
        case .restartWorker(let reloadedFields, _): return reloadedFields;
        case .restApiRestartRequired(let reloadedFields, _, _): return reloadedFields;
        }
    }

    public var workerRestartRequired: Bool {
        switch (self) {
        case .noWorkerRestart: return false;
        case .restartWorker: return true;
        case .restApiRestartRequired: return false;
        }
    }
}

/// Pure comparison between two resolved configs.
public enum ConfigReloadDiff {

    /// Compares `current` against `candidate` and returns the reload decision.
    public static func compare(
        current: ResolvedRuntimeConfig,
        candidate: ResolvedRuntimeConfig
    ) -> ConfigReloadDecision {
        var inPlaceReloadedFields: Array<String> = Array<String>();
        var workerRestartReloadedFields: Array<String> = Array<String>();
        var restartRequiredFields: Array<String> = Array<String>();
        var workerRestartRequired: Bool = false;

        if current.configuredModelDirectories != candidate.configuredModelDirectories {
            workerRestartReloadedFields.append("model_directories");
            workerRestartRequired = true;
        }
        if current.discoveredModels != candidate.discoveredModels {
            workerRestartReloadedFields.append("discovered_model_artifacts");
            workerRestartRequired = true;
        }
        if current.modelPolicyCatalog != candidate.modelPolicyCatalog {
            workerRestartReloadedFields.append("model_policies");
            workerRestartRequired = true;
        }
        if current.unmatchedModelConfigIds != candidate.unmatchedModelConfigIds {
            // Replacement keeps the worker acknowledgement aligned with the
            // complete resolved generation while the unmatched policy itself
            // remains dormant and non-blocking.
            workerRestartReloadedFields.append("dormant_model_policies");
            workerRestartRequired = true;
        }
        if current.maximumMlxMemoryBytes != candidate.maximumMlxMemoryBytes {
            inPlaceReloadedFields.append("maximum_mlx_memory_gb");
        }
        if current.performanceAttributionEnabled != candidate.performanceAttributionEnabled {
            // Supervisor and worker attribution share one setting, while the
            // supervisor writer is opened before the listener binds. An
            // application restart keeps both owners aligned.
            restartRequiredFields.append("diagnostics.performance_attribution_enabled");
        }
        if current.persistentPromptCacheEnabled != candidate.persistentPromptCacheEnabled {
            workerRestartReloadedFields.append("persistent_prompt_cache_enabled");
            workerRestartRequired = true;
        }
        if current.promptCacheConfig != candidate.promptCacheConfig {
            workerRestartReloadedFields.append("prompt_cache");
            workerRestartRequired = true;
        }
        if current.bindAddress != candidate.bindAddress {
            restartRequiredFields.append("supervisor.bind_address");
        }
        if current.loggingConfig != candidate.loggingConfig {
            restartRequiredFields.append("logging");
        }
        if current.configurationGeneration != candidate.configurationGeneration
            && inPlaceReloadedFields.isEmpty
            && workerRestartReloadedFields.isEmpty
            && restartRequiredFields.isEmpty {
            workerRestartReloadedFields.append("resolved_configuration");
            workerRestartRequired = true;
        }

        let discoveredModelCount: Int = candidate.discoveredModels.count;

        if !restartRequiredFields.isEmpty {
            return .restApiRestartRequired(
                reloadedFields: inPlaceReloadedFields,
                restartRequiredFields: restartRequiredFields,
                discoveredModelCount: discoveredModelCount);
        }
        if workerRestartRequired {
            var mergedReloadedFields: Array<String> = inPlaceReloadedFields;
            mergedReloadedFields.append(contentsOf: workerRestartReloadedFields);
            return .restartWorker(
                reloadedFields: mergedReloadedFields,
                discoveredModelCount: discoveredModelCount);
        }
        return .noWorkerRestart(
            reloadedFields: inPlaceReloadedFields,
            discoveredModelCount: discoveredModelCount);
    }
}
