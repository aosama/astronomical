import Foundation;

import IpcProtocol;

import MLX;

/// Builds the matched MoE chat runtime from one validated artifact, with
/// the machine-adaptive expert-residency choice: a fitting artifact serves
/// with every routed expert resident, and a non-fitting artifact installs
/// the disk-materialized paged execution over an expert-free bind.
public enum Qwen35MoeChatRuntime {

    /// Builds the MoE runtime from the pre-validated artifact (the
    /// validator runs once in `Qwen35ChatRuntime.buildArtifactRuntime` and
    /// the artifact is consumed exactly once here). The expert residency
    /// comes from the override when one is given and otherwise from the
    /// machine-adaptive policy: a fitting artifact keeps every routed
    /// expert resident (the Rust `resident_sparse_moe` baseline
    /// conditions), and a non-fitting artifact binds no expert tensors at
    /// all and installs the disk-materialized paged execution over them.
    static func buildMoeRuntime(
        validatedArtifact: ValidatedQwen35Artifact,
        modelDirectory: String,
        autoregressiveConfiguration: WorkerAutoregressiveModelConfiguration,
        prefillChunkTokenCount: Int,
        performanceAttributionEnabled: Bool,
        persistentPromptCachePolicy: Qwen35MoePromptCacheSpawnPolicy? = nil,
        forceExpertPaging: Bool = false,
        mlxMemoryCeilingBytes: UInt64
    ) throws -> LoadedChatRuntime {
        let engine: Qwen35MoeEngine = Qwen35MoeEngine(
            attributionEnabled: performanceAttributionEnabled,
            prefillChunkTokenCount: prefillChunkTokenCount);
        let artifactExpertFootprint: (totalPayloadBytes: UInt64,
            largestGateUpFusionTransientBytes: UInt64) =
            Qwen35MoeChatRuntime.artifactExpertFootprint(validatedArtifact: validatedArtifact);
        let expertResidency: Qwen35MoeArtifactExpertResidency = forceExpertPaging
            ? .paged
            : Qwen35MoeChatRuntime.decidedExpertResidency(
                validatedArtifact: validatedArtifact,
                artifactExpertFootprint: artifactExpertFootprint,
                maximumContextTokenCount: autoregressiveConfiguration.maximumContextTokens,
                mlxMemoryCeilingBytes: mlxMemoryCeilingBytes,
                performanceAttributionEnabled: performanceAttributionEnabled);
        do {
            try engine.loadValidatedArtifact(
                validatedArtifact,
                bindRoutedExperts: expertResidency == .resident,
                residentExpertPayloadBytes: expertResidency == .resident
                    ? artifactExpertFootprint.totalPayloadBytes
                    : nil);
        } catch let engineError as InferenceEngineError {
            throw engineError;
        } catch {
            throw InferenceEngineError.modelLoad(
                reason: "the MoE model weights could not be streamed from the artifact");
        }
        if expertResidency == .paged {
            try Qwen35MoeChatRuntime.installPagedExpertExecution(
                engine: engine,
                validatedArtifact: validatedArtifact,
                modelDirectory: modelDirectory,
                performanceAttributionEnabled: performanceAttributionEnabled);
            try engine.materializePagedRuntimeCore();
        }
        // The prompt cache attaches before the runtime publishes: a store
        // that cannot open fails the model load fail-closed, exactly like
        // the Rust worker startup's store-open failure.
        if let persistentPromptCachePolicy: Qwen35MoePromptCacheSpawnPolicy =
            persistentPromptCachePolicy {
            try engine.attachPersistentPromptCache(
                persistentPromptCachePolicy.makeAttachment(
                    modelId: validatedArtifact.modelId(),
                    modelRevision: validatedArtifact.revision(),
                    configuredBlockTokenCount: autoregressiveConfiguration.chunking
                        .promptCacheBlockTokens.map({ (blockTokens: UInt32) -> Int in
                            return Int(blockTokens);
                        })));
        }
        return try Qwen35ChatRuntime.pairTokenizerWithProcessor(
            modelDirectory: modelDirectory,
            repositoryConfiguration: validatedArtifact.config(),
            autoregressiveConfiguration: autoregressiveConfiguration,
            engine: engine,
            performanceAttributionEnabled: performanceAttributionEnabled);
    }

    /// The machine-adaptive residency decision for one validated artifact:
    /// resident when the resident core payload, the complete expert
    /// payload, and one complete layer's fusion headroom fit the machine's
    /// MLX recommended working set; paged otherwise. Payload bytes come
    /// from the artifact's own dtype-aware tensor profiles, never a
    /// machine constant.
    private static func decidedExpertResidency(
        validatedArtifact: ValidatedQwen35Artifact,
        artifactExpertFootprint: (totalPayloadBytes: UInt64,
            largestGateUpFusionTransientBytes: UInt64),
        maximumContextTokenCount: UInt32,
        mlxMemoryCeilingBytes: UInt64,
        performanceAttributionEnabled: Bool
    ) -> Qwen35MoeArtifactExpertResidency {
        let residencyDecisionStart: ContinuousClock.Instant? = ServingPerformanceAttribution
            .startedOperation(
                operationName: "qwen35_moe_residency_decision",
                attributionEnabled: performanceAttributionEnabled);
        defer {
            ServingPerformanceAttribution.endedOperation(
                operationName: "qwen35_moe_residency_decision",
                operationStart: residencyDecisionStart,
                attributionEnabled: performanceAttributionEnabled);
        }
        let residentTensorNames: Set<String> = Set(Qwen3_5TensorSpec
            .qwen3_5ResidentLanguageTensorProfiles(qwen3_5Config: validatedArtifact.config())
            .map({ (tensorProfile: TensorProfile) -> String in
                return tensorProfile.name;
            }));
        guard let residentPayloadBytes: UInt64 = validatedArtifact.payloadByteCount(
            canonicalTensorNames: residentTensorNames) else {
            return .paged;
        }
        let contextWindowReserveBytes: UInt64? = Qwen35MoeArtifactExpertResidencyPolicy
            .contextWindowReserveBytes(
                fullAttentionLayerCount: UInt64(
                    validatedArtifact.config().fullAttentionDecoderLayerIndexes().count),
                keyValueHeadCount: UInt64(validatedArtifact.config().keyValueHeadCount()),
                headDimension: UInt64(validatedArtifact.config().headDimension()),
                bytesPerElement: validatedArtifact.config().activationDtype() == "float32"
                    ? 4 : 2,
                artifactMaximumPositionCount: UInt64(
                    validatedArtifact.config().maximumPositionCount()),
                maximumContextTokenCount: maximumContextTokenCount);
        guard let contextWindowReserveBytes: UInt64 = contextWindowReserveBytes else {
            return .paged;
        }
        return Qwen35MoeArtifactExpertResidencyPolicy.decide(
            residentPayloadBytes: residentPayloadBytes,
            expertPayloadBytes: artifactExpertFootprint.totalPayloadBytes,
            contextWindowReserveBytes: contextWindowReserveBytes,
            activationHeadroomBytes: 0,
            largestGateUpFusionTransientBytes:
                artifactExpertFootprint.largestGateUpFusionTransientBytes,
            mlxMemoryCeilingBytes: mlxMemoryCeilingBytes);
    }

    /// Captures the validated expert byte footprint once so residency
    /// selection, resident telemetry, and paging setup share the same
    /// header-derived evidence.
    private static func artifactExpertFootprint(
        validatedArtifact: ValidatedQwen35Artifact
    ) -> (totalPayloadBytes: UInt64, largestGateUpFusionTransientBytes: UInt64) {
        let expertTensorNames: Set<String> = Set(validatedArtifact.shardIndex()
            .languageTensorNameToShardFileName()
            .map({ (tensorLocation: (tensorName: String, shardFileName: String)) -> String in
                return tensorLocation.tensorName;
            })
            .filter({ (tensorName: String) -> Bool in
                return Qwen3_5MoeTensorSpec.isSparseSelectedExpertTensorName(tensorName: tensorName);
            }));
        return validatedArtifact.sparseExpertPayloadFootprint(
            canonicalTensorNames: expertTensorNames);
    }

    /// Plans every decoder layer's expert byte geometry from the shard
    /// headers and installs the disk-materialized paged execution.
    private static func installPagedExpertExecution(
        engine: Qwen35MoeEngine,
        validatedArtifact: ValidatedQwen35Artifact,
        modelDirectory: String,
        performanceAttributionEnabled: Bool
    ) throws {
        let installStart: ContinuousClock.Instant? = ServingPerformanceAttribution
            .startedOperation(
                operationName: "qwen35_moe_paging_install",
                attributionEnabled: performanceAttributionEnabled);
        defer {
            ServingPerformanceAttribution.endedOperation(
                operationName: "qwen35_moe_paging_install",
                operationStart: installStart,
                attributionEnabled: performanceAttributionEnabled);
        }
        let repositoryConfiguration: Qwen3_5Config = validatedArtifact.config();
        let decoderLayerCount: Int = Int(repositoryConfiguration.layerCount());
        let weightMap: [String: String] = Dictionary(
            uniqueKeysWithValues: validatedArtifact.shardIndex()
                .languageTensorNameToShardFileName());
        let modelDirectoryUrl: URL = URL(fileURLWithPath: modelDirectory);
        let layerPlans: [QuantizedExpertLayerPlan] = try QuantizedExpertLayerPlanBuilder
            .buildLayerPlans(
                modelDirectory: modelDirectoryUrl,
                weightMap: weightMap,
                layerPrefixes: (0..<decoderLayerCount).map(
                    { (decoderLayerIndex: Int) -> String in
                        return "language_model.model.layers.\(decoderLayerIndex).mlp";
                    }),
                config: repositoryConfiguration);
        let expertPageMaterializer: Qwen35MoeDiskExpertPageMaterializer =
            Qwen35MoeDiskExpertPageMaterializer(
                modelDirectory: modelDirectoryUrl,
                layerPlans: layerPlans,
                expertFileReadMetrics: nil,
                attributionEnabled: performanceAttributionEnabled);
        try engine.installPagedExpertExecution(
            retainedExpertIdsPerLayer: Array(repeating: [], count: decoderLayerCount),
            expertPageMaterializer: expertPageMaterializer);
    }
}
