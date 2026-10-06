import Foundation;

/// Complete config-owned policy for one canonical model after global and
/// per-model inheritance, porting crates/config/src/resolved_model_config.rs.
/// Context comes from the discovered artifact; config may only narrow it.
public struct ResolvedModelConfig: Equatable {

    /// Internal output default used when a model has no configured preference.
    public static let defaultMaximumOutputTokens: UInt32 = 20_480;

    private let maximumContextTokensValue: UInt32?;
    private let maximumOutputTokensValue: UInt32;
    private let configuredMaximumOutputTokensValue: UInt32?;
    private let temperatureValue: Float?;
    private let topPValue: Float?;
    private let chunkingValue: ChunkingConfig;
    private let configuredChunkingFieldsValue: ConfiguredChunkingFields;

    /// Resolves one model's inherited limits, generation defaults, and
    /// chunking. The configured context may only narrow the artifact's
    /// capability, and the configured output default must stay smaller than
    /// the effective context so one prompt token always remains.
    public static func resolve(
        modelId: String,
        artifactMaximumContextTokens: UInt32,
        globalChunkingFile: ChunkingConfigFile,
        configuredModel: ModelConfigFile?
    ) throws -> ResolvedModelConfig {
        let maximumContextTokens: UInt32? = configuredModel?.limits?.maximumContextTokens;
        if let configuredMaximumContextTokens: UInt32 = maximumContextTokens,
           configuredMaximumContextTokens > artifactMaximumContextTokens {
            throw AstronomicalConfigError.configuredContextExceedsArtifact(
                modelId: modelId,
                configuredMaximumContextTokens: configuredMaximumContextTokens,
                artifactMaximumContextTokens: artifactMaximumContextTokens);
        }
        let generationDefaults: GenerationDefaultsConfigFile? = configuredModel?.generationDefaults;
        let effectiveMaximumContextTokens: UInt32 = maximumContextTokens ?? artifactMaximumContextTokens;
        let configuredMaximumOutputTokens: UInt32? = generationDefaults?.maximumOutputTokens;
        if let configuredMaximumOutput: UInt32 = configuredMaximumOutputTokens,
           configuredMaximumOutput >= effectiveMaximumContextTokens {
            throw AstronomicalConfigError.configuredOutputNotSmallerThanContext(
                modelId: modelId,
                configuredMaximumOutputTokens: configuredMaximumOutput,
                effectiveMaximumContextTokens: effectiveMaximumContextTokens);
        }
        let effectiveChunkingFile: ChunkingConfigFile = ChunkingConfigFile.merged(
            globalChunkingFile: globalChunkingFile,
            modelChunkingFile: configuredModel?.chunking);
        // The internal default is policy, not an explicit user demand, so tiny
        // artifacts retain one prompt token instead of becoming undiscoverable.
        let internalMaximumOutputTokens: UInt32 = ResolvedModelConfig.defaultMaximumOutputTokens;
        let saturationSufficientContext: UInt32 = effectiveMaximumContextTokens > 0
            ? effectiveMaximumContextTokens - 1
            : 0;
        let maximumOutputTokens: UInt32 = configuredMaximumOutputTokens
            ?? min(internalMaximumOutputTokens, saturationSufficientContext);
        return ResolvedModelConfig(
            maximumContextTokensValue: maximumContextTokens,
            maximumOutputTokensValue: maximumOutputTokens,
            configuredMaximumOutputTokensValue: configuredMaximumOutputTokens,
            temperatureValue: generationDefaults?.temperature,
            topPValue: generationDefaults?.topP,
            chunkingValue: try ChunkingConfig.resolve(configuredChunkingFile: effectiveChunkingFile),
            configuredChunkingFieldsValue: effectiveChunkingFile.configuredFields());
    }

    private init(
        maximumContextTokensValue: UInt32?,
        maximumOutputTokensValue: UInt32,
        configuredMaximumOutputTokensValue: UInt32?,
        temperatureValue: Float?,
        topPValue: Float?,
        chunkingValue: ChunkingConfig,
        configuredChunkingFieldsValue: ConfiguredChunkingFields
    ) {
        self.maximumContextTokensValue = maximumContextTokensValue;
        self.maximumOutputTokensValue = maximumOutputTokensValue;
        self.configuredMaximumOutputTokensValue = configuredMaximumOutputTokensValue;
        self.temperatureValue = temperatureValue;
        self.topPValue = topPValue;
        self.chunkingValue = chunkingValue;
        self.configuredChunkingFieldsValue = configuredChunkingFieldsValue;
    }

    /// The configured operational context ceiling, if one was supplied.
    public func maximumContextTokens() -> UInt32? {
        return self.maximumContextTokensValue;
    }

    /// The configured output default or Astronomical's internal default.
    public func maximumOutputTokens() -> UInt32 {
        return self.maximumOutputTokensValue;
    }

    /// Distinguishes a user-authored generation default from internal fallback policy.
    public func hasExplicitMaximumOutputTokens() -> Bool {
        return self.configuredMaximumOutputTokensValue != nil;
    }

    /// The configured sampling temperature without inventing a default.
    public func temperature() -> Float? {
        return self.temperatureValue;
    }

    /// The configured nucleus-sampling probability without inventing a default.
    public func topP() -> Float? {
        return self.topPValue;
    }

    /// The complete chunking policy after global and model inheritance.
    public func chunking() -> ChunkingConfig {
        return self.chunkingValue;
    }

    /// Identifies each authored inherited field without labelling sibling
    /// defaults as configured.
    public func configuredChunkingFields() -> ConfiguredChunkingFields {
        return self.configuredChunkingFieldsValue;
    }
}
