import Foundation;

import AstronomicalConfig;
import IpcProtocol;

/// Builds the complete runtime policy catalog from resolved user
/// configuration.
///
/// Keeping tagged chat and image worker policy construction together ensures
/// discovery, configuration generations, and worker replacement compare the
/// same immutable execution policy.
public enum ResolvedModelPolicyCatalog {

    /// Resolves every discovered model into the exact policy sent to the
    /// worker, keyed by canonical requestable model identity.
    public static func resolve(
        userConfig: AstronomicalConfig,
        discoveredModels: Array<DiscoveryDiscoveredModel>,
        artifactContextWindows: Dictionary<String, UInt32>
    ) throws -> Dictionary<String, RuntimeModelPolicy> {
        var modelPolicies: Dictionary<String, RuntimeModelPolicy> = Dictionary<String, RuntimeModelPolicy>();
        for discoveredModel: DiscoveryDiscoveredModel in discoveredModels {
            let runtimeModelPolicy: RuntimeModelPolicy;
            switch (discoveredModel.capabilities) {
            case let .chat(chatCapabilities):
                runtimeModelPolicy = try ResolvedModelPolicyCatalog.chatPolicy(
                    userConfig: userConfig,
                    discoveredModel: discoveredModel,
                    chatCapabilities: chatCapabilities,
                    artifactContextWindows: artifactContextWindows);
            case .imageGeneration:
                runtimeModelPolicy = ResolvedModelPolicyCatalog.imagePolicy(discoveredModel: discoveredModel);
            case let .embeddings(embeddingCapabilities):
                runtimeModelPolicy = ResolvedModelPolicyCatalog.embeddingsPolicy(
                    discoveredModel: discoveredModel,
                    embeddingCapabilities: embeddingCapabilities);
            }
            modelPolicies[discoveredModel.modelId] = runtimeModelPolicy;
        }
        return modelPolicies;
    }

    private static func chatPolicy(
        userConfig: AstronomicalConfig,
        discoveredModel: DiscoveryDiscoveredModel,
        chatCapabilities: DiscoveryChatModelCapabilities,
        artifactContextWindows: Dictionary<String, UInt32>
    ) throws -> RuntimeModelPolicy {
        let resolvedModelConfig: ResolvedModelConfig = try userConfig.resolvedModelConfig(
            modelId: discoveredModel.modelId,
            artifactMaximumContextTokens: chatCapabilities.contextWindowTokens);
        let workerModelConfiguration: WorkerModelConfiguration = ResolvedModelPolicyCatalog.workerModelConfiguration(
            discoveredModel: discoveredModel,
            chatCapabilities: chatCapabilities,
            resolvedModelConfig: resolvedModelConfig);
        return RuntimeModelPolicy(
            modelDirectory: discoveredModel.modelDirectory,
            generationDefaults: RuntimeModelPolicy.generationDefaults(from: resolvedModelConfig),
            configuredMaximumContextTokens: resolvedModelConfig.maximumContextTokens(),
            // The pre-config artifact window stays the routing default.
            defaultMaximumContextTokens: artifactContextWindows[discoveredModel.modelId]
                ?? chatCapabilities.contextWindowTokens,
            configuredChunkingFields: resolvedModelConfig.configuredChunkingFields(),
            workerModelConfiguration: workerModelConfiguration);
    }

    private static func workerModelConfiguration(
        discoveredModel: DiscoveryDiscoveredModel,
        chatCapabilities: DiscoveryChatModelCapabilities,
        resolvedModelConfig: ResolvedModelConfig
    ) -> WorkerModelConfiguration {
        return WorkerModelConfiguration.autoregressive(WorkerAutoregressiveModelConfiguration(
            modelId: discoveredModel.modelId,
            // Worker policy carries model capability rather than a request default.
            maximumContextTokens: chatCapabilities.contextWindowTokens,
            maximumOutputTokens: chatCapabilities.maximumOutputTokens,
            chunking: RuntimeModelPolicy.workerChunkingConfiguration(
                from: resolvedModelConfig.chunking())));
    }

    private static func imagePolicy(discoveredModel: DiscoveryDiscoveredModel) -> RuntimeModelPolicy {
        let workerModelConfiguration: WorkerModelConfiguration;
        switch (discoveredModel.modelFamily) {
        case .qwenImage21:
            workerModelConfiguration = WorkerModelConfiguration.qwenImage21(WorkerQwenImage21ModelConfiguration(
                modelId: discoveredModel.modelId,
                modelFamily: .qwenImage21,
                artifactRevision: discoveredModel.revision));
        // An image capability requires one of the image families above; a
        // discovered directory that classifies otherwise fails at worker
        // selection instead of silently receiving a Flux identity it never
        // verified.
        case .flux2Klein, .qwen35, .k2HorizonMova, .modernbert:
            workerModelConfiguration = WorkerModelConfiguration.flux2Klein(WorkerFlux2KleinModelConfiguration(
                modelId: discoveredModel.modelId,
                modelFamily: .flux2Klein,
                artifactRevision: discoveredModel.revision));
        }
        return RuntimeModelPolicy(
            modelDirectory: discoveredModel.modelDirectory,
            generationDefaults: RuntimeModelGenerationDefaults.inert(),
            configuredMaximumContextTokens: nil,
            defaultMaximumContextTokens: 0,
            configuredChunkingFields: ResolvedModelPolicyCatalog.emptyChunkingFields(),
            workerModelConfiguration: workerModelConfiguration);
    }

    private static func embeddingsPolicy(
        discoveredModel: DiscoveryDiscoveredModel,
        embeddingCapabilities: DiscoveryEmbeddingModelCapabilities
    ) -> RuntimeModelPolicy {
        return RuntimeModelPolicy(
            modelDirectory: discoveredModel.modelDirectory,
            generationDefaults: RuntimeModelGenerationDefaults.inert(),
            configuredMaximumContextTokens: nil,
            defaultMaximumContextTokens: 0,
            configuredChunkingFields: ResolvedModelPolicyCatalog.emptyChunkingFields(),
            workerModelConfiguration: WorkerModelConfiguration.embeddings(WorkerEmbeddingModelConfiguration(
                modelId: discoveredModel.modelId,
                modelFamily: .modernBert,
                artifactRevision: discoveredModel.revision,
                vectorWidth: embeddingCapabilities.vectorWidth,
                maximumInputTokens: embeddingCapabilities.maximumInputTokens)));
    }

    private static func emptyChunkingFields() -> ConfiguredChunkingFields {
        return ConfiguredChunkingFields(chunkingConfigFile: ChunkingConfigFile.defaultFile());
    }
}
