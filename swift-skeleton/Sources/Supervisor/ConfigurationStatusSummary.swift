import Foundation;

import AstronomicalConfig;
import IpcProtocol;

/// One human-readable discovery outcome reduced to its status fields.
public struct ModelDiscoveryDiagnosticSummary {

    public let code: String;
    public let modelId: String;
    public let configuredRootNumbers: Array<Int>;

    public init(code: String, modelId: String, configuredRootNumbers: Array<Int>) {
        self.code = code;
        self.modelId = modelId;
        self.configuredRootNumbers = configuredRootNumbers;
    }

    public func wireValue() -> JsonWireValue {
        var wireObject: JsonWireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "code", value: .string(self.code));
        wireObject.appendEntry(key: "model_id", value: .string(self.modelId));
        wireObject.appendEntry(
            key: "configured_root_numbers",
            value: .array(self.configuredRootNumbers.map { (rootNumber: Int) -> JsonWireValue in
                return .signedInteger(Int64(rootNumber));
            }));
        return .object(wireObject);
    }
}

/// The ready model's effective execution policy summary.
public struct ReadyModelConfigurationSummary {

    public let modelId: String;
    public let maximumContextTokens: ConfigurationValue<UInt32>;
    public let maximumOutputDefaultTokens: ConfigurationValue<UInt32>;
    public let temperature: ConfigurationValue<Double>;
    public let topP: ConfigurationValue<Double>;
    public let chunking: ChunkingConfigurationSummary;

    public init(
        modelId: String,
        maximumContextTokens: ConfigurationValue<UInt32>,
        maximumOutputDefaultTokens: ConfigurationValue<UInt32>,
        temperature: ConfigurationValue<Double>,
        topP: ConfigurationValue<Double>,
        chunking: ChunkingConfigurationSummary
    ) {
        self.modelId = modelId;
        self.maximumContextTokens = maximumContextTokens;
        self.maximumOutputDefaultTokens = maximumOutputDefaultTokens;
        self.temperature = temperature;
        self.topP = topP;
        self.chunking = chunking;
    }

    public func wireValue() -> JsonWireValue {
        var wireObject: JsonWireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "model_id", value: .string(self.modelId));
        wireObject.appendEntry(key: "maximum_context_tokens", value: self.maximumContextTokens.wireValue());
        wireObject.appendEntry(key: "maximum_output_default_tokens", value: self.maximumOutputDefaultTokens.wireValue());
        wireObject.appendEntry(key: "temperature", value: self.temperature.wireValue());
        wireObject.appendEntry(key: "top_p", value: self.topP.wireValue());
        wireObject.appendEntry(key: "chunking", value: self.chunking.wireValue());
        return .object(wireObject);
    }
}

/// The ready model's chunking policy across every tunable chunk quantity.
public struct ChunkingConfigurationSummary {

    public let fixedPromptProcessingChunkSizeTokens: ConfigurationValue<UInt32>;
    public let fixedSsdStreamingPromptProcessingChunkSizeTokens: ConfigurationValue<UInt32>;
    public let fullAttentionKeyValueGrowthTokens: ConfigurationValue<UInt32>;
    public let prefillGraphSubmissionLayerInterval: ConfigurationValue<UInt32>;
    public let experimentalSsdPagingPrefillGraphSubmissionLayerInterval: ConfigurationValue<UInt32>;
    public let experimentalSsdPagingGenerationGraphSubmissionLayerInterval: ConfigurationValue<UInt32>;
    public let promptCacheBlockTokens: NullableConfigurationValue<UInt32>;
    public let promptCacheCommonPrefixStrideBlocks: ConfigurationValue<UInt32>;

    public init(
        fixedPromptProcessingChunkSizeTokens: ConfigurationValue<UInt32>,
        fixedSsdStreamingPromptProcessingChunkSizeTokens: ConfigurationValue<UInt32>,
        fullAttentionKeyValueGrowthTokens: ConfigurationValue<UInt32>,
        prefillGraphSubmissionLayerInterval: ConfigurationValue<UInt32>,
        experimentalSsdPagingPrefillGraphSubmissionLayerInterval: ConfigurationValue<UInt32>,
        experimentalSsdPagingGenerationGraphSubmissionLayerInterval: ConfigurationValue<UInt32>,
        promptCacheBlockTokens: NullableConfigurationValue<UInt32>,
        promptCacheCommonPrefixStrideBlocks: ConfigurationValue<UInt32>
    ) {
        self.fixedPromptProcessingChunkSizeTokens = fixedPromptProcessingChunkSizeTokens;
        self.fixedSsdStreamingPromptProcessingChunkSizeTokens = fixedSsdStreamingPromptProcessingChunkSizeTokens;
        self.fullAttentionKeyValueGrowthTokens = fullAttentionKeyValueGrowthTokens;
        self.prefillGraphSubmissionLayerInterval = prefillGraphSubmissionLayerInterval;
        self.experimentalSsdPagingPrefillGraphSubmissionLayerInterval = experimentalSsdPagingPrefillGraphSubmissionLayerInterval;
        self.experimentalSsdPagingGenerationGraphSubmissionLayerInterval = experimentalSsdPagingGenerationGraphSubmissionLayerInterval;
        self.promptCacheBlockTokens = promptCacheBlockTokens;
        self.promptCacheCommonPrefixStrideBlocks = promptCacheCommonPrefixStrideBlocks;
    }

    public func wireValue() -> JsonWireValue {
        var wireObject: JsonWireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "fixed_prompt_processing_chunk_size_tokens", value: self.fixedPromptProcessingChunkSizeTokens.wireValue());
        wireObject.appendEntry(key: "fixed_ssd_streaming_prompt_processing_chunk_size_tokens", value: self.fixedSsdStreamingPromptProcessingChunkSizeTokens.wireValue());
        wireObject.appendEntry(key: "full_attention_key_value_growth_tokens", value: self.fullAttentionKeyValueGrowthTokens.wireValue());
        wireObject.appendEntry(key: "prefill_graph_submission_layer_interval", value: self.prefillGraphSubmissionLayerInterval.wireValue());
        wireObject.appendEntry(key: "experimental_ssd_paging_prefill_graph_submission_layer_interval", value: self.experimentalSsdPagingPrefillGraphSubmissionLayerInterval.wireValue());
        wireObject.appendEntry(key: "experimental_ssd_paging_generation_graph_submission_layer_interval", value: self.experimentalSsdPagingGenerationGraphSubmissionLayerInterval.wireValue());
        wireObject.appendEntry(key: "prompt_cache_block_tokens", value: self.promptCacheBlockTokens.wireValue());
        wireObject.appendEntry(key: "prompt_cache_common_prefix_stride_blocks", value: self.promptCacheCommonPrefixStrideBlocks.wireValue());
        return .object(wireObject);
    }
}

/// The persistent prompt cache configuration triple.
public struct PromptCacheConfigurationSummary {

    public let enabled: ConfigurationValue<Bool>;
    public let capacityBytes: ConfigurationValue<UInt64>;

    public init(enabled: ConfigurationValue<Bool>, capacityBytes: ConfigurationValue<UInt64>) {
        self.enabled = enabled;
        self.capacityBytes = capacityBytes;
    }

    public func wireValue() -> JsonWireValue {
        var wireObject: JsonWireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "enabled", value: self.enabled.wireValue());
        wireObject.appendEntry(key: "capacity_bytes", value: self.capacityBytes.wireValue());
        return .object(wireObject);
    }
}

/// The MLX (machine learning acceleration) memory ceiling configuration.
public struct MemoryConfigurationSummary {

    public let configuredMaximumBytes: UInt64?;
    public let effectiveMaximumBytes: UInt64;
    public let pendingMaximumBytes: UInt64?;
    public let error: String?;

    public init(
        configuredMaximumBytes: UInt64?,
        effectiveMaximumBytes: UInt64,
        pendingMaximumBytes: UInt64?,
        error: String?
    ) {
        self.configuredMaximumBytes = configuredMaximumBytes;
        self.effectiveMaximumBytes = effectiveMaximumBytes;
        self.pendingMaximumBytes = pendingMaximumBytes;
        self.error = error;
    }

    public func wireValue() -> JsonWireValue {
        var wireObject: JsonWireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "configured_maximum_bytes", value: MemoryConfigurationSummary.optionalBytesWireValue(self.configuredMaximumBytes));
        wireObject.appendEntry(key: "effective_maximum_bytes", value: .unsignedInteger(self.effectiveMaximumBytes));
        wireObject.appendEntry(key: "pending_maximum_bytes", value: MemoryConfigurationSummary.optionalBytesWireValue(self.pendingMaximumBytes));
        wireObject.appendEntry(key: "error", value: MemoryConfigurationSummary.optionalTextWireValue(self.error));
        return .object(wireObject);
    }

    private static func optionalBytesWireValue(_ optionalBytes: UInt64?) -> JsonWireValue {
        guard let presentBytes: UInt64 = optionalBytes else {
            return .null;
        }
        return .unsignedInteger(presentBytes);
    }

    private static func optionalTextWireValue(_ optionalText: String?) -> JsonWireValue {
        guard let presentText: String = optionalText else {
            return .null;
        }
        return .string(presentText);
    }
}

/**
 * Builds the path-free configured, resolved, and worker-effective status
 * contract, porting apps/supervisor/src/configuration_status.rs.
 *
 * The configured snapshot carries what the operator authored, the resolved
 * snapshot what the daemon currently resolves, and the worker health
 * acknowledgement what the serving process actually applies; the generation
 * triple and the restart verdict come from comparing all three.
 */
public struct ConfigurationStatusSummary {

    public let configuredGeneration: String?;
    public let resolvedGeneration: String?;
    public let effectiveGeneration: String?;
    public let isEffective: Bool;
    public let restartRequired: Bool;
    public let validationError: String?;
    public let modelDiscoveryDiagnostics: Array<ModelDiscoveryDiagnosticSummary>;
    public let unmatchedModelConfigIds: Array<String>;
    public let readyModel: ReadyModelConfigurationSummary?;
    public let promptCache: PromptCacheConfigurationSummary;
    public let memory: MemoryConfigurationSummary;

    public static func fromParts(
        configuredRuntimeConfig: ResolvedRuntimeConfig?,
        resolvedRuntimeConfig: ResolvedRuntimeConfig?,
        workerHealthSnapshot: WorkerHealthSnapshot,
        configurationValidationError: String?
    ) -> ConfigurationStatusSummary {
        let workerConfiguration: WorkerRuntimeFeatureConfiguration? = workerHealthSnapshot.workerRuntimeFeatureConfiguration;
        let configuredGeneration: String? = configuredRuntimeConfig?.configurationGeneration;
        let resolvedGeneration: String? = resolvedRuntimeConfig?.configurationGeneration;
        let effectiveGeneration: String? = workerConfiguration?.configurationGeneration;
        let isEffective: Bool = configuredGeneration != nil
            && configurationValidationError == nil
            && configuredGeneration == resolvedGeneration
            && configuredGeneration == effectiveGeneration;
        let restartRequired: Bool = workerHealthSnapshot.status != .loading
            && configurationValidationError == nil
            && configuredGeneration != nil
            && !isEffective
            && workerHealthSnapshot.pendingMlxMemoryCeilingBytes == nil;
        return ConfigurationStatusSummary(
            configuredGeneration: configuredGeneration,
            resolvedGeneration: resolvedGeneration,
            effectiveGeneration: effectiveGeneration,
            isEffective: isEffective,
            restartRequired: restartRequired,
            validationError: configurationValidationError,
            modelDiscoveryDiagnostics: ConfigurationStatusSummary.diagnosticSummaries(configuredRuntimeConfig),
            unmatchedModelConfigIds: configuredRuntimeConfig?.unmatchedModelConfigIds ?? Array<String>(),
            readyModel: ConfigurationStatusSummary.readyModelSummary(
                configuredRuntimeConfig: configuredRuntimeConfig,
                resolvedRuntimeConfig: resolvedRuntimeConfig,
                readyModelId: workerHealthSnapshot.readyModelId,
                workerConfiguration: workerConfiguration),
            promptCache: ConfigurationStatusSummary.promptCacheSummary(
                configuredRuntimeConfig: configuredRuntimeConfig,
                workerConfiguration: workerConfiguration),
            memory: MemoryConfigurationSummary(
                configuredMaximumBytes: configuredRuntimeConfig?.maximumMlxMemoryBytes,
                effectiveMaximumBytes: workerHealthSnapshot.effectiveMlxMemoryCeilingBytes,
                pendingMaximumBytes: workerHealthSnapshot.pendingMlxMemoryCeilingBytes,
                error: workerHealthSnapshot.mlxMemoryLimitError));
    }

    public func wireValue() -> JsonWireValue {
        var wireObject: JsonWireObject = JsonWireObject(entries: Array<(key: String, value: JsonWireValue)>());
        wireObject.appendEntry(key: "configured_generation", value: ConfigurationStatusSummary.optionalTextWireValue(self.configuredGeneration));
        wireObject.appendEntry(key: "resolved_generation", value: ConfigurationStatusSummary.optionalTextWireValue(self.resolvedGeneration));
        wireObject.appendEntry(key: "effective_generation", value: ConfigurationStatusSummary.optionalTextWireValue(self.effectiveGeneration));
        wireObject.appendEntry(key: "is_effective", value: .boolean(self.isEffective));
        wireObject.appendEntry(key: "restart_required", value: .boolean(self.restartRequired));
        wireObject.appendEntry(key: "validation_error", value: ConfigurationStatusSummary.optionalTextWireValue(self.validationError));
        wireObject.appendEntry(
            key: "model_discovery_diagnostics",
            value: .array(self.modelDiscoveryDiagnostics.map { (diagnostic: ModelDiscoveryDiagnosticSummary) -> JsonWireValue in
                return diagnostic.wireValue();
            }));
        wireObject.appendEntry(
            key: "unmatched_model_config_ids",
            value: .array(self.unmatchedModelConfigIds.map { (unmatchedModelConfigId: String) -> JsonWireValue in
                return .string(unmatchedModelConfigId);
            }));
        wireObject.appendEntry(
            key: "ready_model",
            value: self.readyModel.map { (readyModelSummary: ReadyModelConfigurationSummary) -> JsonWireValue in
                return readyModelSummary.wireValue();
            } ?? .null);
        wireObject.appendEntry(key: "prompt_cache", value: self.promptCache.wireValue());
        wireObject.appendEntry(key: "memory", value: self.memory.wireValue());
        return .object(wireObject);
    }

    private static func diagnosticSummaries(
        _ configuredRuntimeConfig: ResolvedRuntimeConfig?
    ) -> Array<ModelDiscoveryDiagnosticSummary> {
        guard let configuredRuntimeConfig: ResolvedRuntimeConfig = configuredRuntimeConfig else {
            return Array<ModelDiscoveryDiagnosticSummary>();
        }
        return configuredRuntimeConfig.modelDiscoveryDiagnostics.map { (diagnostic: DiscoveryModelDiscoveryDiagnostic) -> ModelDiscoveryDiagnosticSummary in
            return ModelDiscoveryDiagnosticSummary(
                code: diagnostic.code.rawValue,
                modelId: diagnostic.modelId,
                configuredRootNumbers: diagnostic.configuredRootNumbers);
        };
    }

    private static func readyModelSummary(
        configuredRuntimeConfig: ResolvedRuntimeConfig?,
        resolvedRuntimeConfig: ResolvedRuntimeConfig?,
        readyModelId: String?,
        workerConfiguration: WorkerRuntimeFeatureConfiguration?
    ) -> ReadyModelConfigurationSummary? {
        guard let readyModelId: String = readyModelId else {
            return nil;
        }
        let configuredPolicy: RuntimeModelPolicy? = configuredRuntimeConfig?.modelPolicyCatalog[readyModelId];
        let resolvedPolicy: RuntimeModelPolicy? = resolvedRuntimeConfig?.modelPolicyCatalog[readyModelId];
        let effectiveModel: WorkerLoadedModelRuntimeConfiguration? = workerConfiguration?.loadedModel.flatMap { (loadedModel: WorkerLoadedModelRuntimeConfiguration) -> WorkerLoadedModelRuntimeConfiguration? in
            return loadedModel.modelId() == readyModelId ? loadedModel : nil;
        };
        let effectiveAutoregressiveModel: WorkerLoadedAutoregressiveModelRuntimeConfiguration? = effectiveModel?.autoregressive();
        return ReadyModelConfigurationSummary(
            modelId: readyModelId,
            maximumContextTokens: ConfigurationValue<UInt32>(
                configured: configuredPolicy?.configuredMaximumContextTokens,
                defaultValue: configuredPolicy.map { (policy: RuntimeModelPolicy) -> UInt32 in
                    return policy.defaultMaximumContextTokens;
                },
                effective: effectiveAutoregressiveModel?.maximumContextTokens),
            maximumOutputDefaultTokens: ConfigurationValue<UInt32>(
                configured: configuredPolicy?.generationDefaults.configuredMaximumOutputTokens.map { (configuredMaximumOutputTokens: UInt16) -> UInt32 in
                    return UInt32(configuredMaximumOutputTokens);
                },
                defaultValue: configuredPolicy.map { (policy: RuntimeModelPolicy) -> UInt32 in
                    let contextCeilingTokens: UInt32 = policy.configuredMaximumContextTokens ?? policy.defaultMaximumContextTokens;
                    let oneBelowContextTokens: UInt32 = contextCeilingTokens > 0 ? contextCeilingTokens - 1 : 0;
                    return min(ResolvedModelConfig.defaultMaximumOutputTokens, oneBelowContextTokens);
                },
                effective: resolvedPolicy.map { (policy: RuntimeModelPolicy) -> UInt32 in
                    return UInt32(policy.generationDefaults.maximumOutputTokens);
                }),
            temperature: ConfigurationStatusSummary.samplingValue(
                configuredThousandths: configuredPolicy?.generationDefaults.temperatureThousandths,
                effectiveThousandths: resolvedPolicy?.generationDefaults.temperatureThousandths),
            topP: ConfigurationStatusSummary.samplingValue(
                configuredThousandths: configuredPolicy?.generationDefaults.topPThousandths,
                effectiveThousandths: resolvedPolicy?.generationDefaults.topPThousandths),
            chunking: ConfigurationStatusSummary.chunkingSummary(
                configuredPolicy: configuredPolicy,
                effectiveAutoregressiveModel: effectiveAutoregressiveModel));
    }

    private static func chunkingSummary(
        configuredPolicy: RuntimeModelPolicy?,
        effectiveAutoregressiveModel: WorkerLoadedAutoregressiveModelRuntimeConfiguration?
    ) -> ChunkingConfigurationSummary {
        let configuredFields: ConfiguredChunkingFields = configuredPolicy?.configuredChunkingFields ?? ConfiguredChunkingFields.inactive();
        let configuredChunking: WorkerChunkingConfiguration? = configuredPolicy?.workerModelConfiguration.autoregressive()?.chunking;
        let effectiveChunking: WorkerChunkingConfiguration? = effectiveAutoregressiveModel?.chunking;
        return ChunkingConfigurationSummary(
            fixedPromptProcessingChunkSizeTokens: ConfigurationStatusSummary.chunkTriple(
                isConfiguredField: configuredFields.fixedPromptProcessingChunkSizeTokens,
                configuredChunkTokens: configuredChunking?.fixedPromptProcessingChunkSizeTokens,
                effectiveChunkTokens: effectiveChunking?.fixedPromptProcessingChunkSizeTokens,
                defaultChunkTokens: ChunkingConfig.defaultFixedPromptProcessingChunkSizeTokens),
            fixedSsdStreamingPromptProcessingChunkSizeTokens: ConfigurationStatusSummary.chunkTriple(
                isConfiguredField: configuredFields.fixedSsdStreamingPromptProcessingChunkSizeTokens,
                configuredChunkTokens: configuredChunking?.fixedSsdStreamingPromptProcessingChunkSizeTokens,
                effectiveChunkTokens: effectiveChunking?.fixedSsdStreamingPromptProcessingChunkSizeTokens,
                defaultChunkTokens: ChunkingConfig.defaultFixedSsdStreamingPromptProcessingChunkSizeTokens),
            fullAttentionKeyValueGrowthTokens: ConfigurationStatusSummary.chunkTriple(
                isConfiguredField: configuredFields.fullAttentionKeyValueGrowthTokens,
                configuredChunkTokens: configuredChunking?.fullAttentionKeyValueGrowthTokens,
                effectiveChunkTokens: effectiveChunking?.fullAttentionKeyValueGrowthTokens,
                defaultChunkTokens: ChunkingConfig.defaultFullAttentionKeyValueGrowthTokens),
            prefillGraphSubmissionLayerInterval: ConfigurationStatusSummary.chunkTriple(
                isConfiguredField: configuredFields.prefillGraphSubmissionLayerInterval,
                configuredChunkTokens: configuredChunking?.prefillGraphSubmissionLayerInterval,
                effectiveChunkTokens: effectiveChunking?.prefillGraphSubmissionLayerInterval,
                defaultChunkTokens: ChunkingConfig.defaultPrefillGraphSubmissionLayerInterval),
            experimentalSsdPagingPrefillGraphSubmissionLayerInterval: ConfigurationStatusSummary.chunkTriple(
                isConfiguredField: configuredFields.experimentalSsdPagingPrefillGraphSubmissionLayerInterval,
                configuredChunkTokens: configuredChunking?.experimentalSsdPagingPrefillGraphSubmissionLayerInterval,
                effectiveChunkTokens: effectiveChunking?.experimentalSsdPagingPrefillGraphSubmissionLayerInterval,
                defaultChunkTokens: ChunkingConfig.defaultExperimentalSsdPagingPrefillGraphSubmissionLayerInterval),
            experimentalSsdPagingGenerationGraphSubmissionLayerInterval: ConfigurationStatusSummary.chunkTriple(
                isConfiguredField: configuredFields.experimentalSsdPagingGenerationGraphSubmissionLayerInterval,
                configuredChunkTokens: configuredChunking?.experimentalSsdPagingGenerationGraphSubmissionLayerInterval,
                effectiveChunkTokens: effectiveChunking?.experimentalSsdPagingGenerationGraphSubmissionLayerInterval,
                defaultChunkTokens: ChunkingConfig.defaultExperimentalSsdPagingGenerationGraphSubmissionLayerInterval),
            promptCacheBlockTokens: NullableConfigurationValue<UInt32>(
                isConfigured: configuredFields.promptCacheBlockTokens,
                configured: configuredFields.promptCacheBlockTokens ? configuredChunking?.promptCacheBlockTokens : nil,
                defaultValue: nil,
                effective: effectiveChunking?.promptCacheBlockTokens),
            promptCacheCommonPrefixStrideBlocks: ConfigurationStatusSummary.chunkTriple(
                isConfiguredField: configuredFields.promptCacheCommonPrefixStrideBlocks,
                configuredChunkTokens: configuredChunking?.promptCacheCommonPrefixStrideBlocks,
                effectiveChunkTokens: effectiveChunking?.promptCacheCommonPrefixStrideBlocks,
                defaultChunkTokens: ChunkingConfig.defaultPromptCacheCommonPrefixStrideBlocks));
    }

    private static func chunkTriple(
        isConfiguredField: Bool,
        configuredChunkTokens: UInt32?,
        effectiveChunkTokens: UInt32?,
        defaultChunkTokens: UInt32
    ) -> ConfigurationValue<UInt32> {
        return ConfigurationValue<UInt32>(
            configured: isConfiguredField ? configuredChunkTokens : nil,
            defaultValue: defaultChunkTokens,
            effective: effectiveChunkTokens);
    }

    private static func samplingValue(
        configuredThousandths: UInt16?,
        effectiveThousandths: UInt16?
    ) -> ConfigurationValue<Double> {
        return ConfigurationValue<Double>(
            configured: configuredThousandths.map { (configuredThousandths: UInt16) -> Double in
                return Double(configuredThousandths) / 1_000.0;
            },
            defaultValue: nil,
            effective: effectiveThousandths.map { (effectiveThousandths: UInt16) -> Double in
                return Double(effectiveThousandths) / 1_000.0;
            });
    }

    private static func promptCacheSummary(
        configuredRuntimeConfig: ResolvedRuntimeConfig?,
        workerConfiguration: WorkerRuntimeFeatureConfiguration?
    ) -> PromptCacheConfigurationSummary {
        return PromptCacheConfigurationSummary(
            enabled: ConfigurationValue<Bool>(
                configured: configuredRuntimeConfig?.configuredPersistentPromptCacheEnabled,
                defaultValue: true,
                effective: workerConfiguration?.persistentPromptCacheEnabled),
            capacityBytes: ConfigurationValue<UInt64>(
                configured: configuredRuntimeConfig?.configuredPromptCacheMaximumSizeBytes,
                defaultValue: PromptCacheConfig.defaultMaximumSizeBytes,
                effective: workerConfiguration?.promptCacheMaximumSizeBytes));
    }

    private static func optionalTextWireValue(_ optionalText: String?) -> JsonWireValue {
        guard let presentText: String = optionalText else {
            return .null;
        }
        return .string(presentText);
    }
}
