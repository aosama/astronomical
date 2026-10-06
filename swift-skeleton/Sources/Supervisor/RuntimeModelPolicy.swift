import Foundation;

import AstronomicalConfig;
import IpcProtocol;

/// Request defaults resolved from one model's configuration.
public struct RuntimeModelGenerationDefaults: Equatable, Sendable {

    public let maximumOutputTokens: UInt16;
    public let configuredMaximumOutputTokens: UInt16?;
    public let temperatureThousandths: UInt16?;
    public let topPThousandths: UInt16?;

    public init(
        maximumOutputTokens: UInt16,
        configuredMaximumOutputTokens: UInt16?,
        temperatureThousandths: UInt16?,
        topPThousandths: UInt16?
    ) {
        self.maximumOutputTokens = maximumOutputTokens;
        self.configuredMaximumOutputTokens = configuredMaximumOutputTokens;
        self.temperatureThousandths = temperatureThousandths;
        self.topPThousandths = topPThousandths;
    }

    /// Inert defaults for typed image and embedding worker policies, which
    /// carry no chat request defaults at all.
    public static func inert() -> RuntimeModelGenerationDefaults {
        return RuntimeModelGenerationDefaults(
            maximumOutputTokens: 0,
            configuredMaximumOutputTokens: nil,
            temperatureThousandths: nil,
            topPThousandths: nil);
    }
}

/// One canonical requestable model's directory and fully resolved execution
/// policy, used for routing and worker swaps.
public struct RuntimeModelPolicy: Equatable, Sendable {

    public let modelDirectory: FilePath;
    public let generationDefaults: RuntimeModelGenerationDefaults;
    public let configuredMaximumContextTokens: UInt32?;
    public let defaultMaximumContextTokens: UInt32;
    public let configuredChunkingFields: ConfiguredChunkingFields;
    public let workerModelConfiguration: WorkerModelConfiguration;

    public init(
        modelDirectory: FilePath,
        generationDefaults: RuntimeModelGenerationDefaults,
        configuredMaximumContextTokens: UInt32?,
        defaultMaximumContextTokens: UInt32,
        configuredChunkingFields: ConfiguredChunkingFields,
        workerModelConfiguration: WorkerModelConfiguration
    ) {
        self.modelDirectory = modelDirectory;
        self.generationDefaults = generationDefaults;
        self.configuredMaximumContextTokens = configuredMaximumContextTokens;
        self.defaultMaximumContextTokens = defaultMaximumContextTokens;
        self.configuredChunkingFields = configuredChunkingFields;
        self.workerModelConfiguration = workerModelConfiguration;
    }

    /// Request defaults resolved from one model's configuration.
    public static func generationDefaults(
        from resolvedModelConfig: ResolvedModelConfig
    ) -> RuntimeModelGenerationDefaults {
        return RuntimeModelGenerationDefaults(
            maximumOutputTokens: UInt16(clamping: resolvedModelConfig.maximumOutputTokens()),
            configuredMaximumOutputTokens: resolvedModelConfig.configuredMaximumOutputTokens().map { (configuredMaximumOutputTokens: UInt32) -> UInt16 in
                return UInt16(clamping: configuredMaximumOutputTokens);
            },
            temperatureThousandths: resolvedModelConfig.temperature().map { (temperature: Float) -> UInt16 in
                return RuntimeModelPolicy.samplingParameterThousandths(temperature);
            },
            topPThousandths: resolvedModelConfig.topP().map { (topP: Float) -> UInt16 in
                return RuntimeModelPolicy.samplingParameterThousandths(topP);
            });
    }

    /// The worker chunking policy after global and model inheritance.
    public static func workerChunkingConfiguration(
        from chunking: ChunkingConfig
    ) -> WorkerChunkingConfiguration {
        return WorkerChunkingConfiguration(
            fixedPromptProcessingChunkSizeTokens: chunking.fixedPromptProcessingChunkSizeTokens(),
            fixedSsdStreamingPromptProcessingChunkSizeTokens: chunking.fixedSsdStreamingPromptProcessingChunkSizeTokens(),
            fullAttentionKeyValueGrowthTokens: chunking.fullAttentionKeyValueGrowthTokens(),
            prefillGraphSubmissionLayerInterval: chunking.prefillGraphSubmissionLayerInterval(),
            experimentalSsdPagingPrefillGraphSubmissionLayerInterval: chunking.experimentalSsdPagingPrefillGraphSubmissionLayerInterval(),
            experimentalSsdPagingGenerationGraphSubmissionLayerInterval: chunking.experimentalSsdPagingGenerationGraphSubmissionLayerInterval(),
            promptCacheBlockTokens: chunking.promptCacheBlockTokens(),
            promptCacheCommonPrefixStrideBlocks: chunking.promptCacheCommonPrefixStrideBlocks(),
            experimentalDecodeStageAttributionEnabled: chunking.experimentalDecodeStageAttributionEnabled(),
            experimentalQuantizedKvCacheEnabled: chunking.experimentalQuantizedKvCacheEnabled(),
            experimentalFusedMoeDecodeEnabled: chunking.experimentalFusedMoeDecodeEnabled());
    }

    /// Sampling settings travel as thousandths so JSON rounding stays exact.
    private static func samplingParameterThousandths(_ samplingParameter: Float) -> UInt16 {
        return UInt16((samplingParameter * 1_000.0).rounded());
    }
}
