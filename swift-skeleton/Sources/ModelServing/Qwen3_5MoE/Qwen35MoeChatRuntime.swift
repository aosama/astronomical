import Foundation;

import IpcProtocol;

/// Builds the paged MoE chat runtime from one validated artifact: the MoE
/// engine streams the shard weights, every decoder layer's expert geometry
/// is planned from the shard headers, and paged expert execution installs
/// over the loaded model so routed experts read from disk on demand.
///
/// The slice-one residency plan retains no experts — every routed expert
/// pages through the disk materializer — which the residency
/// classification treats as the first-class paged mode. The memory
/// governor track owns retained-set planning later.
public enum Qwen35MoeChatRuntime {

    /// Builds the paged runtime from the pre-validated artifact (the
    /// validator runs once in `Qwen35ChatRuntime.buildArtifactRuntime` and
    /// the artifact is consumed exactly once here).
    static func buildPagedRuntime(
        validatedArtifact: ValidatedQwen35Artifact,
        modelDirectory: String,
        autoregressiveConfiguration: WorkerAutoregressiveModelConfiguration,
        prefillChunkTokenCount: Int,
        performanceAttributionEnabled: Bool
    ) throws -> LoadedChatRuntime {
        let engine: Qwen35MoeEngine = Qwen35MoeEngine(
            attributionEnabled: performanceAttributionEnabled,
            prefillChunkTokenCount: prefillChunkTokenCount);
        do {
            try engine.loadValidatedArtifact(validatedArtifact);
        } catch let engineError as InferenceEngineError {
            throw engineError;
        } catch {
            throw InferenceEngineError.modelLoad(
                reason: "the MoE model weights could not be streamed from the artifact");
        }
        try Qwen35MoeChatRuntime.installPagedExpertExecution(
            engine: engine,
            validatedArtifact: validatedArtifact,
            modelDirectory: modelDirectory,
            performanceAttributionEnabled: performanceAttributionEnabled);
        return try Qwen35ChatRuntime.pairTokenizerWithProcessor(
            modelDirectory: modelDirectory,
            repositoryConfiguration: validatedArtifact.config(),
            autoregressiveConfiguration: autoregressiveConfiguration,
            engine: engine);
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
        let layerPlans: [QuantizedExpertLayerPlan] = try (0..<decoderLayerCount).map(
            { (decoderLayerIndex: Int) -> QuantizedExpertLayerPlan in
                return try QuantizedExpertLayerPlanBuilder.buildLayerPlan(
                    modelDirectory: modelDirectoryUrl,
                    weightMap: weightMap,
                    layerPrefix: "language_model.model.layers.\(decoderLayerIndex).mlp",
                    config: repositoryConfiguration);
            });
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
