import Foundation;

import AstronomicalConfig;

/// Resolves startup and reload config from one user configuration directory.
///
/// Migrates the resolver half of apps/supervisor/src/config_reload.rs.
public struct ResolvedRuntimeConfigResolver {

    private let instancePaths: AstronomicalInstancePaths;
    private let fallbackWorkerExecutablePath: FilePath;

    public init(
        instancePaths: AstronomicalInstancePaths,
        fallbackWorkerExecutablePath: FilePath
    ) {
        self.instancePaths = instancePaths;
        self.fallbackWorkerExecutablePath = fallbackWorkerExecutablePath;
    }

    public var stateDirectory: FilePath {
        return self.instancePaths.stateDirectory;
    }

    public var resolvedInstancePaths: AstronomicalInstancePaths {
        return self.instancePaths;
    }

    /// Loads and resolves the current config file.
    public func load() throws -> ResolvedRuntimeConfig {
        let userConfig: AstronomicalConfig = try AstronomicalConfig.loadFromInstancePaths(self.instancePaths);
        return try self.resolve(userConfig: userConfig);
    }

    /// Resolves one already-loaded config using startup-equivalent precedence.
    public func resolve(userConfig: AstronomicalConfig) throws -> ResolvedRuntimeConfig {
        let supervisorBindAddress: SocketEndpoint = try self.instancePaths.validateConfiguredBindAddress(
            try userConfig.supervisorBindAddress());
        let configuredModelDirectories: Array<FilePath> = userConfig.modelDirectories;
        let effectiveDiscovery: EffectiveModelDiscovery.Outcome = try EffectiveModelDiscovery.discover(
            automaticModelsDirectory: self.instancePaths.modelsDirectory,
            configuredModelDirectories: configuredModelDirectories);
        var discoveredModels: Array<DiscoveryDiscoveredModel> = effectiveDiscovery.discoveredModels;
        let modelDiscoveryDiagnostics: Array<DiscoveryModelDiscoveryDiagnostic> = effectiveDiscovery.diagnostics;
        let discoveredModelIds: Set<String> = Set<String>(discoveredModels.map { (discoveredModel: DiscoveryDiscoveredModel) -> String in
            return discoveredModel.modelId;
        });
        var artifactContextWindows: Dictionary<String, UInt32> = Dictionary<String, UInt32>();
        for discoveredModel: DiscoveryDiscoveredModel in discoveredModels {
            guard case let .chat(chatCapabilities) = discoveredModel.capabilities else {
                continue;
            }
            artifactContextWindows[discoveredModel.modelId] = chatCapabilities.contextWindowTokens;
        }
        let unmatchedModelConfigIds: Array<String> = userConfig.configuredModelIds.filter { (configuredModelId: String) -> Bool in
            return !discoveredModelIds.contains(configuredModelId);
        };
        for discoveredModelIndex: Int in discoveredModels.indices {
            guard case let .chat(chatCapabilities) = discoveredModels[discoveredModelIndex].capabilities else {
                continue;
            }
            let resolvedModelConfig: ResolvedModelConfig = try userConfig.resolvedModelConfig(
                modelId: discoveredModels[discoveredModelIndex].modelId,
                artifactMaximumContextTokens: chatCapabilities.contextWindowTokens);
            discoveredModels[discoveredModelIndex] = ResolvedRuntimeConfigResolver.applyEffectiveModelLimits(
                discoveredModel: discoveredModels[discoveredModelIndex],
                resolvedModelConfig: resolvedModelConfig);
        }
        let modelPolicyCatalog: Dictionary<String, RuntimeModelPolicy> = try ResolvedModelPolicyCatalog.resolve(
            userConfig: userConfig,
            discoveredModels: discoveredModels,
            artifactContextWindows: artifactContextWindows);
        let promptCacheConfig: PromptCacheConfig = try userConfig.promptCache();
        let loggingConfig: LoggingConfig = userConfig.logging();
        let configurationGeneration: String = try ResolvedConfigurationGeneration.derive(
            documentGeneration: try userConfig.generation(),
            discoveredModels: discoveredModels,
            modelPolicyCatalog: modelPolicyCatalog,
            unmatchedModelConfigIds: unmatchedModelConfigIds);
        return ResolvedRuntimeConfig(
            configurationGeneration: configurationGeneration,
            workerExecutablePath: self.fallbackWorkerExecutablePath,
            discoveredModels: discoveredModels,
            modelDiscoveryDiagnostics: modelDiscoveryDiagnostics,
            configuredModelDirectories: configuredModelDirectories,
            modelPolicyCatalog: modelPolicyCatalog,
            unmatchedModelConfigIds: unmatchedModelConfigIds,
            maximumMlxMemoryBytes: try userConfig.maximumMlxMemoryBytes(),
            performanceAttributionEnabled: userConfig.performanceAttributionEnabled(),
            completionAttributionEnabled: userConfig.completionAttributionEnabled(),
            experimentalQwenThinkingChannelSeedEnabled: userConfig.experimentalQwenThinkingChannelSeedEnabled(),
            persistentPromptCacheEnabled: userConfig.persistentPromptCacheEnabled(),
            configuredPersistentPromptCacheEnabled: userConfig.configuredPersistentPromptCacheEnabled(),
            configuredPromptCacheMaximumSizeBytes: try userConfig.configuredPromptCacheMaximumSizeBytes(),
            promptCacheConfig: promptCacheConfig,
            bindAddress: supervisorBindAddress.description,
            bindEndpoint: supervisorBindAddress,
            loggingConfig: loggingConfig);
    }

    /// Config may narrow a chat model's capability, never widen it: the
    /// context ceiling applies, input keeps one prompt token, and output is
    /// the u16 ceiling clamped under the effective context.
    private static func applyEffectiveModelLimits(
        discoveredModel: DiscoveryDiscoveredModel,
        resolvedModelConfig: ResolvedModelConfig
    ) -> DiscoveryDiscoveredModel {
        guard case let .chat(chatCapabilities) = discoveredModel.capabilities else {
            return discoveredModel;
        }
        let effectiveMaximumContextTokens: UInt32 = resolvedModelConfig.maximumContextTokens()
            ?? chatCapabilities.contextWindowTokens;
        let saturatedContextMinusOne: UInt32 = effectiveMaximumContextTokens > 0
            ? effectiveMaximumContextTokens - 1
            : 0;
        let narrowedCapabilities: DiscoveryChatModelCapabilities = DiscoveryChatModelCapabilities(
            contextWindowTokens: effectiveMaximumContextTokens,
            maximumInputTokens: saturatedContextMinusOne,
            maximumOutputTokens: min(UInt32(UInt16.max), saturatedContextMinusOne),
            supportsVision: chatCapabilities.supportsVision,
            supportsReasoning: chatCapabilities.supportsReasoning,
            supportsToolCalls: chatCapabilities.supportsToolCalls);
        return DiscoveryDiscoveredModel(
            modelId: discoveredModel.modelId,
            providerModelId: discoveredModel.providerModelId,
            modelFamily: discoveredModel.modelFamily,
            revision: discoveredModel.revision,
            modelDirectory: discoveredModel.modelDirectory,
            capabilities: .chat(narrowedCapabilities),
            license: discoveredModel.license,
            modelSizeBytes: discoveredModel.modelSizeBytes);
    }
}
