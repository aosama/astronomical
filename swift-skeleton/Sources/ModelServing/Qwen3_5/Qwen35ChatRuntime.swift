import Foundation;

import IpcProtocol;
import MLX;
import MLXHuggingFace;
import MLXLMCommon;
import Tokenizers;

/// Builds the matched dense Qwen3.5 chat runtime from one model directory:
/// the artifact config drives the in-memory engine construction and the
/// directory's tokenizer files drive the bridged upstream tokenizer.
///
/// The in-memory builder serves journeys that exercise the tokenizer and
/// processor pairing without a packaged artifact; production loads go
/// through `buildArtifactRuntime`, which streams validated shard weights
/// into the engine before the runtime is published.
public enum Qwen35ChatRuntime {

    /// Builds the production runtime: the directory is validated end to
    /// end, the architecture routes the validated artifact onto the dense
    /// engine or the machine-adaptive resident/paged MoE engine, and the
    /// tokenizer bridges from the same directory. MoE expert residency
    /// comes from artifact payload and machine-ceiling evidence; callers
    /// may force the safe paged mode.
    public static func buildArtifactRuntime(
        modelDirectory: String,
        modelConfiguration: WorkerModelConfiguration,
        prefillChunkTokenCount: Int? = nil,
        performanceAttributionEnabled: Bool = false,
        persistentPromptCachePolicy: Qwen35MoePromptCacheSpawnPolicy? = nil,
        forceMoeExpertPaging: Bool = false,
        mlxMemoryCeilingBytes: UInt64? = nil
    ) throws -> LoadedChatRuntime {
        let runtimeCandidate: ChatRuntimeCandidate = try Qwen35ChatRuntime
            .prepareArtifactRuntime(
                modelDirectory: modelDirectory,
                modelConfiguration: modelConfiguration,
                prefillChunkTokenCount: prefillChunkTokenCount,
                performanceAttributionEnabled: performanceAttributionEnabled,
                persistentPromptCachePolicy: persistentPromptCachePolicy,
                forceMoeExpertPaging: forceMoeExpertPaging,
                mlxMemoryCeilingBytes: mlxMemoryCeilingBytes);
        return try runtimeCandidate.load();
    }

    /// Validates the artifact and resolves its model kind without allocating
    /// MLX weights. The candidate defers model construction and MLX work
    /// until the worker has retired the previous model. A nil
    /// `prefillChunkTokenCount` derives the engine chunk from the worker
    /// chunking configuration, so the serving chunk policy owns execution.
    public static func prepareArtifactRuntime(
        modelDirectory: String,
        modelConfiguration: WorkerModelConfiguration,
        prefillChunkTokenCount: Int? = nil,
        performanceAttributionEnabled: Bool = false,
        persistentPromptCachePolicy: Qwen35MoePromptCacheSpawnPolicy? = nil,
        forceMoeExpertPaging: Bool = false,
        mlxMemoryCeilingBytes: UInt64? = nil
    ) throws -> ChatRuntimeCandidate {
        guard let autoregressiveConfiguration = modelConfiguration.autoregressive() else {
            throw InferenceEngineError.modelLoad(
                reason: "the dense Qwen3.5 runtime requires an autoregressive model policy");
        }
        let effectiveMlxMemoryCeilingBytes: UInt64 = mlxMemoryCeilingBytes
            ?? Qwen35ChatRuntime.recommendedMlxMemoryCeilingBytes();
        guard effectiveMlxMemoryCeilingBytes > 0,
            effectiveMlxMemoryCeilingBytes <= UInt64(Int.max) else {
            throw InferenceEngineError.modelLoad(
                reason: "the effective MLX memory ceiling is unavailable or outside the platform range");
        }
        let validatedArtifact: ValidatedQwen35Artifact = try Qwen35ArtifactValidator()
            .validate(
                modelDirectory: modelDirectory,
                maxOutputTokens: autoregressiveConfiguration.maximumOutputTokens,
                performanceAttributionEnabled: performanceAttributionEnabled);
        return ChatRuntimeCandidate(validatedArtifact: validatedArtifact) {
            return try Qwen35ChatRuntime.buildValidatedArtifactRuntime(
                validatedArtifact: validatedArtifact,
                modelDirectory: modelDirectory,
                autoregressiveConfiguration: autoregressiveConfiguration,
                prefillChunkTokenCount: prefillChunkTokenCount,
                performanceAttributionEnabled: performanceAttributionEnabled,
                persistentPromptCachePolicy: persistentPromptCachePolicy,
                forceMoeExpertPaging: forceMoeExpertPaging,
                effectiveMlxMemoryCeilingBytes: effectiveMlxMemoryCeilingBytes);
        };
    }

    private static func buildValidatedArtifactRuntime(
        validatedArtifact: ValidatedQwen35Artifact,
        modelDirectory: String,
        autoregressiveConfiguration: WorkerAutoregressiveModelConfiguration,
        prefillChunkTokenCount: Int?,
        performanceAttributionEnabled: Bool,
        persistentPromptCachePolicy: Qwen35MoePromptCacheSpawnPolicy?,
        forceMoeExpertPaging: Bool,
        effectiveMlxMemoryCeilingBytes: UInt64
    ) throws -> LoadedChatRuntime {
        // The worker chunking configuration owns the serving chunk; the
        // engine's prefill loop must execute exactly the policy the
        // supervisor configured, so an explicit override stays the only way
        // to diverge from it.
        let enginePrefillChunkTokenCount: Int;
        if let prefillChunkTokenCount: Int = prefillChunkTokenCount {
            enginePrefillChunkTokenCount = prefillChunkTokenCount;
        } else {
            let configuredChunkTokenCount: UInt32 = autoregressiveConfiguration.chunking
                .fixedPromptProcessingChunkSizeTokens;
            guard configuredChunkTokenCount > 0 else {
                throw InferenceEngineError.modelLoad(
                    reason: "the prompt-processing chunk size must be positive");
            }
            enginePrefillChunkTokenCount = Int(configuredChunkTokenCount);
        }
        let memoryPreparationStart: ContinuousClock.Instant? = ServingPerformanceAttribution
            .startedOperation(
                operationName: "qwen35_mlx_pre_load_quiescence",
                attributionEnabled: performanceAttributionEnabled);
        MlxMemoryLimitPolicy.prepareForModelLoad(
            effectiveCeilingBytes: effectiveMlxMemoryCeilingBytes);
        ServingPerformanceAttribution.endedOperation(
            operationName: "qwen35_mlx_pre_load_quiescence",
            operationStart: memoryPreparationStart,
            attributionEnabled: performanceAttributionEnabled);
        switch validatedArtifact.config().feedForwardArchitecture() {
        case .dense:
            return try Qwen35ChatRuntime.buildDenseRuntime(
                validatedArtifact: validatedArtifact,
                modelDirectory: modelDirectory,
                autoregressiveConfiguration: autoregressiveConfiguration,
                prefillChunkTokenCount: enginePrefillChunkTokenCount,
                performanceAttributionEnabled: performanceAttributionEnabled);
        case .mixtureOfExperts:
            return try Qwen35MoeChatRuntime.buildMoeRuntime(
                validatedArtifact: validatedArtifact,
                modelDirectory: modelDirectory,
                autoregressiveConfiguration: autoregressiveConfiguration,
                prefillChunkTokenCount: enginePrefillChunkTokenCount,
                performanceAttributionEnabled: performanceAttributionEnabled,
                persistentPromptCachePolicy: persistentPromptCachePolicy,
                forceExpertPaging: forceMoeExpertPaging,
                mlxMemoryCeilingBytes: effectiveMlxMemoryCeilingBytes);
        }
    }

    /// Streams the validated dense artifact into the dense engine and pairs
    /// the runtime tail.
    private static func buildDenseRuntime(
        validatedArtifact: ValidatedQwen35Artifact,
        modelDirectory: String,
        autoregressiveConfiguration: WorkerAutoregressiveModelConfiguration,
        prefillChunkTokenCount: Int,
        performanceAttributionEnabled: Bool
    ) throws -> LoadedChatRuntime {
        let engine: Qwen35DenseEngine = Qwen35DenseEngine(
            attributionEnabled: performanceAttributionEnabled,
            prefillChunkTokenCount: prefillChunkTokenCount);
        do {
            try engine.loadValidatedArtifact(validatedArtifact);
        } catch let engineError as InferenceEngineError {
            throw engineError;
        } catch {
            throw InferenceEngineError.modelLoad(
                reason: "the dense model weights could not be streamed from the artifact");
        }
        return try Qwen35ChatRuntime.pairTokenizerWithProcessor(
            modelDirectory: modelDirectory,
            repositoryConfiguration: validatedArtifact.config(),
            autoregressiveConfiguration: autoregressiveConfiguration,
            engine: engine,
            performanceAttributionEnabled: performanceAttributionEnabled);
    }

    public static func buildInMemoryRuntime(
        modelDirectory: String,
        modelConfiguration: WorkerModelConfiguration,
        prefillChunkTokenCount: Int = 512
    ) throws -> LoadedChatRuntime {
        guard let autoregressiveConfiguration = modelConfiguration.autoregressive() else {
            throw InferenceEngineError.modelLoad(
                reason: "the dense Qwen3.5 runtime requires an autoregressive model policy");
        }
        let directoryUrl: URL = URL(fileURLWithPath: modelDirectory);
        let configBytes: Data;
        do {
            configBytes = try Data(contentsOf: directoryUrl.appendingPathComponent("config.json"));
        } catch {
            throw InferenceEngineError.modelLoad(
                reason: "the dense model configuration file could not be read");
        }
        let repositoryConfiguration: Qwen3_5Config;
        do {
            repositoryConfiguration = try Qwen3_5Config.fromJsonBytes(
                configBytes: Array(configBytes));
        } catch {
            throw InferenceEngineError.modelLoad(
                reason: "the dense model configuration could not be decoded");
        }
        let engine: Qwen35DenseEngine = Qwen35DenseEngine(
            prefillChunkTokenCount: prefillChunkTokenCount);
        do {
            try engine.loadInMemoryModel(configBytes: configBytes);
        } catch let engineError as InferenceEngineError {
            throw engineError;
        } catch {
            throw InferenceEngineError.modelLoad(
                reason: "the dense model could not be constructed");
        }
        return try Qwen35ChatRuntime.pairTokenizerWithProcessor(
            modelDirectory: modelDirectory,
            repositoryConfiguration: repositoryConfiguration,
            autoregressiveConfiguration: autoregressiveConfiguration,
            engine: engine);
    }

    /// Bridges the directory tokenizer and builds the matched processor
    /// around one loaded engine — the tail both runtime builders share.
    static func pairTokenizerWithProcessor(
        modelDirectory: String,
        repositoryConfiguration: Qwen3_5Config,
        autoregressiveConfiguration: WorkerAutoregressiveModelConfiguration,
        engine: any InferenceEngine,
        performanceAttributionEnabled: Bool = false
    ) throws -> LoadedChatRuntime {
        let directoryUrl: URL = URL(fileURLWithPath: modelDirectory);
        let bridgedTokenizer: any MLXLMCommon.Tokenizer;
        let tokenizerLoadStart: ContinuousClock.Instant? = ServingPerformanceAttribution
            .startedOperation(
                operationName: "qwen35_artifact_tokenizer_load",
                attributionEnabled: performanceAttributionEnabled);
        do {
            bridgedTokenizer = #adaptHuggingFaceTokenizer(
                try Qwen35ChatRuntime.blockingTokenizerLoad(directoryUrl: directoryUrl));
        } catch {
            ServingPerformanceAttribution.endedOperation(
                operationName: "qwen35_artifact_tokenizer_load",
                operationStart: tokenizerLoadStart,
                attributionEnabled: performanceAttributionEnabled);
            throw InferenceEngineError.modelLoad(
                reason: "the model tokenizer files could not be loaded: \(error)");
        }
        ServingPerformanceAttribution.endedOperation(
            operationName: "qwen35_artifact_tokenizer_load",
            operationStart: tokenizerLoadStart,
            attributionEnabled: performanceAttributionEnabled);
        let processor: Qwen35ChatProcessor = Qwen35ChatProcessor(
            tokenizer: bridgedTokenizer,
            modelId: autoregressiveConfiguration.modelId,
            endOfSequenceTokenIds: Set(
                repositoryConfiguration.endOfSequenceTokenIds()),
            capabilities: Qwen35ChatProcessor.capabilities(
                autoregressiveConfiguration: autoregressiveConfiguration,
                maximumPositionCount: repositoryConfiguration.maximumPositionCount()));
        return LoadedChatRuntime(processor: processor, engine: engine);
    }

    /// Blocks the synchronous factory boundary on the async upstream
    /// tokenizer load, exactly as the Rust factory's spawn_blocking boundary
    /// does; the worker loop stays the single MLX owner around it. The boxed
    /// result is the task's only shared state and is @unchecked Sendable by
    /// the semaphore handshake.
    private static func blockingTokenizerLoad(
        directoryUrl: URL
    ) throws -> any Tokenizers.Tokenizer {
        final class LoadOutcome: @unchecked Sendable {
            var result: Result<any Tokenizers.Tokenizer, Error>?
        }
        let loadOutcome: LoadOutcome = LoadOutcome();
        let loadSemaphore: DispatchSemaphore = DispatchSemaphore(value: 0);
        let detachedTask: Task<Void, Never> = Task.detached(priority: .userInitiated) {
            do {
                loadOutcome.result = .success(
                    try await Tokenizers.AutoTokenizer.from(modelFolder: directoryUrl));
            } catch {
                loadOutcome.result = .failure(error);
            }
            loadSemaphore.signal();
        }
        loadSemaphore.wait();
        detachedTask.cancel();
        guard let outcome: Result<any Tokenizers.Tokenizer, Error> = loadOutcome.result else {
            throw InferenceEngineError.modelLoad(
                reason: "the tokenizer loading task completed without a result");
        }
        return try outcome.get();
    }

    private static func recommendedMlxMemoryCeilingBytes() -> UInt64 {
        let recommendedWorkingSetBytes: Int? = MLX.GPU.maxRecommendedWorkingSetBytes();
        guard let recommendedWorkingSetBytes: Int = recommendedWorkingSetBytes,
            recommendedWorkingSetBytes > 0 else {
            return 0;
        }
        return UInt64(recommendedWorkingSetBytes);
    }

}
