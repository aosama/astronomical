import Foundation;

import IpcProtocol;
import MLX;
import MLXLMCommon;
import MLXLLM;
import MLXNN;

/// The prepared token-id prompt one dense Qwen3.5 generation consumes.
public final class Qwen35PreparedInferenceRequest: PreparedInferenceRequest {

    public let promptTokenIds: Array<UInt32>;
    public let samplingSettings: Qwen35SamplingSettings;
    /// The compiled enforced structured-generation constraint, present only
    /// when the request carried one; masked on visible tokens only.
    public let guidedConstraint: Qwen35GuidedConstraint?;
    /// Whether the rendered chat template opened the thinking channel, so
    /// generation starts inside `<think>` and the constraint stays dormant
    /// until the reasoning-end boundary commits.
    public let startsInsideThinking: Bool;
    /// The token ids that close the thinking channel naturally.
    public let naturalReasoningEndTokenIds: Set<UInt32>;
    /// The request-local hard reasoning-budget controller, present only when
    /// the request carries a thinking allowance.
    public let thinkingBudgetState: Qwen35ThinkingBudgetState?;

    public init(
        promptTokenIds: Array<UInt32>,
        samplingSettings: Qwen35SamplingSettings,
        guidedConstraint: Qwen35GuidedConstraint? = nil,
        startsInsideThinking: Bool = false,
        naturalReasoningEndTokenIds: Set<UInt32> = Set(),
        thinkingBudgetState: Qwen35ThinkingBudgetState? = nil
    ) {
        self.promptTokenIds = promptTokenIds;
        self.samplingSettings = samplingSettings;
        self.guidedConstraint = guidedConstraint;
        self.startsInsideThinking = startsInsideThinking;
        self.naturalReasoningEndTokenIds = naturalReasoningEndTokenIds;
        self.thinkingBudgetState = thinkingBudgetState;
    }

    public var promptTokenCount: Int {
        return self.promptTokenIds.count;
    }
}

/// The dense Qwen3.5 inference engine over the upstream mlx-swift-lm model.
///
/// Mirrors the serving behavior of the Rust qwen3_5 family runtime on the
/// engine adapter seam: chunked prompt processing with progress boundaries,
/// one seeded sampler per request, and one decode step per engine turn.
/// Weights load through the artifact streaming slice; this slice constructs
/// the model from the artifact config so journeys run a real forward pass
/// in-memory. Model-load, prefill-chunk, and decode-step attribution spans
/// switch through configuration.
public final class Qwen35DenseEngine: InferenceEngine {

    private let attributionEnabled: Bool;
    private let prefillChunkTokenCount: Int;
    private var denseModel: (any LanguageModel)?;
    private var totalLayerCount: UInt32 = 0;
    private var modelPayloadBytes: UInt64 = 0;
    private var activeCache: [KVCache]?;
    private var activeSampler: (any LogitSampler)?;
    private var promptTokenIds: Array<UInt32> = [];
    private var prefillNextTokenOffset: Int = 0;
    private var prefillStartClock: ContinuousClock.Instant?;
    private var prefillElapsedMillis: UInt64 = 0;
    private var lastLogits: MLXArray?;
    private var hasEmittedPreparation: Bool = false;
    private var hasEmittedFirstDecode: Bool = false;
    private var cancelledRequestIds: Set<RequestId> = [];
    private var activeThinkingBudgetState: Qwen35ThinkingBudgetState? = nil;
    private var activeGuidedConstraint: Qwen35GuidedConstraint?;
    private var isInsideThinking: Bool = false;
    private var activeReasoningEndTokenIds: Set<UInt32> = Set();

    public init(
        attributionEnabled: Bool = false,
        prefillChunkTokenCount: Int = 512
    ) {
        self.attributionEnabled = attributionEnabled;
        self.prefillChunkTokenCount = prefillChunkTokenCount;
    }

    /// Constructs the dense model from artifact config bytes with the
    /// in-memory path; journeys that need a real forward pass without a
    /// packaged artifact use this, while production loads stream through
    /// `loadValidatedArtifact`.
    public func loadInMemoryModel(configBytes: Data) throws {
        let upstreamConfiguration: Qwen35Configuration;
        do {
            upstreamConfiguration = try JSONDecoder().decode(
                Qwen35Configuration.self, from: configBytes);
        } catch {
            throw InferenceEngineError.modelLoad(
                reason: "the dense model configuration could not be decoded");
        }
        let modelLoadStart: ContinuousClock.Instant? =
            ServingPerformanceAttribution.startedOperation(
                operationName: "qwen35_model_load", attributionEnabled: self.attributionEnabled);
        let loadedModel: Qwen35Model = Qwen35Model(upstreamConfiguration);
        try loadedModel.prepare();
        ServingPerformanceAttribution.endedOperation(
            operationName: "qwen35_model_load",
            operationStart: modelLoadStart,
            attributionEnabled: self.attributionEnabled);
        // The upstream configuration keeps its text section internal, so the
        // layer count derives from the repository config contract, which is
        // also the structural-validity source the journeys assert against.
        let repositoryConfiguration: Qwen3_5Config = try Qwen3_5Config.fromJsonBytes(
            configBytes: Array(configBytes));
        self.denseModel = loadedModel;
        self.totalLayerCount = repositoryConfiguration.layerCount();
    }

    public func load() throws -> EngineLoadResult {
        guard self.denseModel != nil else {
            throw InferenceEngineError.modelLoad(
                reason: "no dense model is loaded");
        }
        return EngineLoadResult(
            minimumMlxMemoryCeilingBytes: 1, expertMemoryMode: nil);
    }

    /// Loads the dense model from a validated artifact's shard descriptors.
    ///
    /// The artifact is consumed: its shard sources transfer exactly once,
    /// so one validated artifact feeds exactly one engine load. Model-load,
    /// shard-read, and weight-bind attribution spans switch through
    /// configuration.
    public func loadValidatedArtifact(_ validatedArtifact: ValidatedQwen35Artifact) throws {
        let modelLoadStart: ContinuousClock.Instant? =
            ServingPerformanceAttribution.startedOperation(
                operationName: "qwen35_model_load", attributionEnabled: self.attributionEnabled);
        let loadedModel: Qwen35Model = try Qwen35ArtifactWeightLoading.loadArtifactBoundModel(
            validatedArtifact: validatedArtifact, attributionEnabled: self.attributionEnabled);
        ServingPerformanceAttribution.endedOperation(
            operationName: "qwen35_model_load",
            operationStart: modelLoadStart,
            attributionEnabled: self.attributionEnabled);
        self.denseModel = loadedModel;
        self.totalLayerCount = validatedArtifact.config().layerCount();
        self.modelPayloadBytes = validatedArtifact.totalPayloadBytes();
    }

    public func startGeneration(
        _ inferenceRequest: any PreparedInferenceRequest
    ) throws -> EngineGenerationStart {
        guard let denseModel = self.denseModel else {
            throw InferenceEngineError.invalidRequest(reason: "no dense model is loaded");
        }
        guard let preparedRequest = inferenceRequest as? Qwen35PreparedInferenceRequest else {
            throw InferenceEngineError.invalidRequest(
                reason: "the request does not belong to the dense Qwen3.5 engine");
        }
        if self.activeCache != nil {
            throw InferenceEngineError.engineBusy;
        }
        self.promptTokenIds = preparedRequest.promptTokenIds;
        self.prefillNextTokenOffset = 0;
        self.prefillStartClock = nil;
        self.prefillElapsedMillis = 0;
        self.lastLogits = nil;
        self.hasEmittedPreparation = false;
        self.hasEmittedFirstDecode = false;
        self.activeSampler = preparedRequest.samplingSettings.makeSampler();
        self.activeGuidedConstraint = preparedRequest.guidedConstraint;
        self.isInsideThinking = preparedRequest.startsInsideThinking;
        self.activeReasoningEndTokenIds = preparedRequest.naturalReasoningEndTokenIds;
        self.activeThinkingBudgetState = preparedRequest.thinkingBudgetState;
        self.activeCache = try denseModel.newCache(parameters: nil);
        return EngineGenerationStart(
            cachedTokenCount: 0,
            restoredPromptPrefixTokenCount: 0,
            expertMemoryMode: nil,
            promptProcessingPhase: .target);
    }

    public func decodeNextToken(requestId: RequestId) throws -> GeneratedToken {
        guard let denseModel = self.denseModel, let activeCache = self.activeCache else {
            throw InferenceEngineError.invalidRequest(
                reason: "the engine holds no active dense request");
        }
        if self.cancelledRequestIds.contains(requestId) {
            throw InferenceEngineError.invalidRequest(
                reason: "the generation was already cancelled");
        }
        if self.prefillNextTokenOffset < self.promptTokenIds.count {
            return try self.decodeNextPrefillChunk(activeCache: activeCache);
        }
        if self.hasEmittedPreparation == false {
            self.hasEmittedPreparation = true;
            return .generationPreparationStarted(
                totalLayerCount: self.totalLayerCount,
                residentExpertCount: 0,
                residentExpertPayloadBytes: 0,
                mlxMemorySnapshot: nil);
        }
        guard let activeSampler = self.activeSampler, let lastLogits = self.lastLogits else {
            throw InferenceEngineError.fatalExecution(
                reason: "the dense engine lost its sampler or logits before decode");
        }
        // The forced transition token is selected before ordinary sampling and
        // fed through the decoder exactly; the budget state observes the
        // commit and reports whether the committed text stays in reasoning.
        let selectedTokenId: Int;
        if let forcedTokenId = try self.nextForcedThinkingTransitionTokenId() {
            selectedTokenId = Int(forcedTokenId);
        } else {
            selectedTokenId = try self.sampleTokenId(
                sampler: activeSampler, lastLogits: lastLogits);
        }
        let isReasoningToken: Bool = try self.observeCommittedThinkingToken(selectedTokenId);
        let decodeForwardStart: ContinuousClock.Instant? =
            ServingPerformanceAttribution.startedOperation(
                operationName: "qwen35_decode_step",
                attributionEnabled: self.attributionEnabled);
        let decodeOutput: LMOutput = denseModel(
            LMInput.Text(tokens: MLXArray([selectedTokenId], [1, 1])),
            cache: activeCache,
            state: nil);
        let firstDecodeForwardElapsedMillis: UInt64? = self.hasEmittedFirstDecode
            ? nil : Qwen35DenseEngine.millisSince(decodeForwardStart);
        self.hasEmittedFirstDecode = true;
        self.lastLogits = decodeOutput.logits;
        ServingPerformanceAttribution.endedOperation(
            operationName: "qwen35_decode_step",
            operationStart: decodeForwardStart,
            attributionEnabled: self.attributionEnabled);
        return .tokenId(
            generatedTokenId: UInt32(clamping: selectedTokenId),
            isReasoningToken: isReasoningToken,
            expertMemoryMode: nil,
            mlxMemorySnapshot: nil,
            firstDecodeForwardElapsedMillis: firstDecodeForwardElapsedMillis,
            generationFinalization: nil);
    }

    /// Selects the next budgeted forced-transition token, if the hard
    /// allowance has been exhausted. Budget-state violations are fatal: a
    /// diverging forced stream would corrupt decoder history.
    private func nextForcedThinkingTransitionTokenId() throws -> UInt32? {
        guard var budgetState = self.activeThinkingBudgetState else {
            return nil;
        }
        do {
            let forcedTokenId = try budgetState.nextForcedTransitionTokenId();
            self.activeThinkingBudgetState = budgetState;
            return forcedTokenId;
        } catch {
            throw InferenceEngineError.fatalExecution(
                reason: "invalid Qwen3.5 thinking-budget state: \(error)");
        }
    }

    /// Observes the token committed to decoder history through the budget
    /// state and keeps the engine's thinking-phase flag in sync.
    private func observeCommittedThinkingToken(_ committedTokenId: Int) throws -> Bool {
        guard var budgetState = self.activeThinkingBudgetState else {
            return false;
        }
        do {
            let isReasoningToken = try budgetState.observeCommittedToken(
                UInt32(clamping: committedTokenId));
            self.isInsideThinking = budgetState.isInsideThinking;
            self.activeThinkingBudgetState = budgetState;
            return isReasoningToken;
        } catch {
            throw InferenceEngineError.fatalExecution(
                reason: "invalid Qwen3.5 thinking-budget state: \(error)");
        }
    }

    private func decodeNextPrefillChunk(activeCache: [KVCache]) throws -> GeneratedToken {
        guard let denseModel = self.denseModel else {
            throw InferenceEngineError.invalidRequest(reason: "no dense model is loaded");
        }
        let chunkStart: ContinuousClock.Instant? = ServingPerformanceAttribution
            .startedOperation(
                operationName: "qwen35_prefill_chunk",
                attributionEnabled: self.attributionEnabled);
        if self.prefillStartClock == nil {
            self.prefillStartClock = ContinuousClock.now;
        }
        let chunkEnd: Int = min(
            self.prefillNextTokenOffset + self.prefillChunkTokenCount,
            self.promptTokenIds.count);
        let chunkTokenIds: Array<UInt32> = Array(
            self.promptTokenIds[self.prefillNextTokenOffset ..< chunkEnd]);
        let chunkOutput: LMOutput = denseModel(
            LMInput.Text(tokens: MLXArray(chunkTokenIds, [1, chunkTokenIds.count])),
            cache: activeCache,
            state: nil);
        self.prefillNextTokenOffset = chunkEnd;
        if self.prefillNextTokenOffset >= self.promptTokenIds.count {
            self.lastLogits = chunkOutput.logits;
        }
        let chunkElapsedMillis: UInt64 = Qwen35DenseEngine.millisSince(chunkStart);
        self.prefillElapsedMillis = self.prefillElapsedMillis.addingReportingOverflow(
            chunkElapsedMillis).partialValue;
        ServingPerformanceAttribution.endedOperation(
            operationName: "qwen35_prefill_chunk",
            operationStart: chunkStart,
            attributionEnabled: self.attributionEnabled);
        return .prefillProgress(
            processedTokenCount: UInt32(chunkTokenIds.count),
            elapsedMillis: chunkElapsedMillis,
            forwardPrefillChunkElapsedMillis: chunkElapsedMillis,
            completedPrefillChunkTokens: UInt32(chunkTokenIds.count),
            mlxMemorySnapshot: nil,
            expertResidencyTelemetry: nil,
            expertMemoryMode: nil,
            promptWorkReuse: WorkerPromptWorkReuse(
                targetEligibleTokenCount: 0, targetRestoredTokenCount: 0));
    }

    public func injectInputTokens(requestId: RequestId, inputTokenIds: Array<UInt32>) throws {
        guard self.cancelledRequestIds.contains(requestId) == false else {
            throw InferenceEngineError.invalidRequest(reason: "the generation was cancelled");
        }
        self.promptTokenIds.append(contentsOf: inputTokenIds);
    }

    public func cancelGeneration(requestId: RequestId) throws -> GenerationFinalization {
        self.cancelledRequestIds.insert(requestId);
        let finalizedSnapshot: WorkerMlxMemorySnapshot? = self.collectMlxMemorySnapshot();
        self.releaseActiveRequest();
        return GenerationFinalization(
            expertMemoryMode: nil,
            mlxMemorySnapshot: finalizedSnapshot,
            expertResidencyTelemetry: nil);
    }

    public func collectMlxMemorySnapshot() -> WorkerMlxMemorySnapshot? {
        let memoryObservation: Memory.Snapshot = Memory.snapshot();
        let modelCorePayloadBytes: UInt64 = self.modelPayloadBytes > 0
            ? self.modelPayloadBytes
            : UInt64(max(0, memoryObservation.activeMemory));
        return WorkerMlxMemorySnapshot(
            source: .idlePoll,
            activeMemoryBytes: UInt64(max(0, memoryObservation.activeMemory)),
            allocatorCacheMemoryBytes: UInt64(max(0, memoryObservation.cacheMemory)),
            peakMemoryBytes: UInt64(max(0, memoryObservation.peakMemory)),
            expertPayloadBytes: 0,
            modelCorePayloadBytes: modelCorePayloadBytes,
            contextStatePayloadBytes: 0,
            memoryCeilingUtilization: nil);
    }

    public func applyMlxMemoryLimit(_ requestedMlxMemoryCeilingBytes: UInt64) throws {
        Memory.cacheLimit = Int(clamping: requestedMlxMemoryCeilingBytes);
    }

    /** Samples one token from the last logit row, applying the guided
    constraint's grammar mask on visible tokens only — mirroring the Rust
    generated-token emission rule where thinking tokens feed the budget state
    and visible tokens feed the constraint. */
    private func sampleTokenId(
        sampler: any LogitSampler,
        lastLogits: MLXArray
    ) throws -> Int {
        var logitRow: MLXArray = lastLogits[0, -1];
        if let guidedConstraint = self.activeGuidedConstraint,
           self.isInsideThinking == false,
           guidedConstraint.isTerminated() == false {
            do {
                logitRow = try guidedConstraint.maskLogits(logitRow);
            } catch {
                throw InferenceEngineError.invalidRequest(
                    reason: "the structured-generation constraint failed to mask the decode step: \(error)");
            }
        }
        let sampledTokenId: Int = sampler.sample(logits: logitRow).item(Int.self);
        if let guidedConstraint = self.activeGuidedConstraint {
            if self.isInsideThinking {
                if self.currentReasoningEndTokenIds().contains(UInt32(clamping: sampledTokenId)) {
                    self.isInsideThinking = false;
                }
            } else if guidedConstraint.isTerminated() == false {
                do {
                    try guidedConstraint.commitToken(Int32(clamping: sampledTokenId));
                } catch {
                    throw InferenceEngineError.invalidRequest(
                        reason: "the sampled token violated the structured-generation constraint: \(error)");
                }
            }
        }
        return sampledTokenId;
    }

    private func currentReasoningEndTokenIds() -> Set<UInt32> {
        return self.activeReasoningEndTokenIds;
    }

    private func releaseActiveRequest() -> Void {
        self.activeCache = nil;
        self.activeSampler = nil;
        self.lastLogits = nil;
        self.promptTokenIds = [];
        self.prefillNextTokenOffset = 0;
        self.prefillStartClock = nil;
        self.prefillElapsedMillis = 0;
        self.hasEmittedPreparation = false;
        self.hasEmittedFirstDecode = false;
        self.activeGuidedConstraint = nil;
        self.activeThinkingBudgetState = nil;
        self.isInsideThinking = false;
    }

    private static func millisSince(
        _ startedAt: ContinuousClock.Instant?
    ) -> UInt64 {
        guard let startedAt = startedAt else {
            return 0;
        }
        let elapsed: Duration = ContinuousClock.now.duration(to: startedAt);
        return UInt64(max(0, elapsed.components.seconds * 1000
            + elapsed.components.attoseconds / 1_000_000_000_000_000));
    }
}
