import Foundation;

import IpcProtocol;
import MLX;
import MLXLMCommon;
import MLXLLM;
import MLXNN;

/// The resident-execution Qwen3.5 Mixture-of-Experts inference engine over
/// the upstream mlx-swift-lm model.
///
/// Mirrors the serving behavior of the Rust qwen3_5_moe family runtime on
/// the engine adapter seam, in the resident execution mode: every routed
/// expert of every decoder layer sits in wired memory, chunked prompt
/// processing reports progress boundaries with the resident telemetry, and
/// one seeded sampler serves one request at a time. Weights load through
/// the in-memory config path so hermetic journeys run a real forward pass;
/// the artifact-streaming load lands with the MoE artifact surface. The
/// dense sibling keeps its own engine type — the Rust split between the
/// qwen3_5 and qwen3_5_moe module trees carries over. Model-load,
/// prefill-chunk, and decode-step attribution spans switch through
/// configuration.
public final class Qwen35MoeEngine: InferenceEngine {

    private let attributionEnabled: Bool;
    private let prefillChunkTokenCount: Int;
    private var moeModel: (any LanguageModel)?;
    private var moeRepositoryConfiguration: Qwen3_5Config?;
    private var expertResidency: (any Qwen35MoeExpertResidency)?;
    private var pagedForwardContext: Qwen35MoePagedForwardContext?;
    private var pagedExpertDecorator: Qwen35MoePagedExpertDecorator?;
    private var totalLayerCount: UInt32 = 0;
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

    /// Constructs the MoE model from artifact config bytes with the
    /// in-memory path; journeys that need a real forward pass without a
    /// packaged artifact use this. A dense checkpoint fails closed: the MoE
    /// engine never silently serves the dense sibling architecture.
    public func loadInMemoryModel(configBytes: Data) throws {
        let upstreamConfiguration: Qwen35Configuration;
        do {
            upstreamConfiguration = try JSONDecoder().decode(
                Qwen35Configuration.self, from: configBytes);
        } catch {
            throw InferenceEngineError.modelLoad(
                reason: "the MoE model configuration could not be decoded");
        }
        let repositoryConfiguration: Qwen3_5Config;
        do {
            repositoryConfiguration = try Qwen3_5Config.fromJsonBytes(
                configBytes: Array(configBytes));
        } catch {
            throw InferenceEngineError.modelLoad(
                reason: "the MoE model configuration could not be validated");
        }
        guard repositoryConfiguration.feedForwardArchitecture() == .mixtureOfExperts else {
            throw InferenceEngineError.modelLoad(
                reason: "the Qwen3.5 checkpoint is not a Mixture of Experts configuration");
        }
        let modelLoadStart: ContinuousClock.Instant? =
            ServingPerformanceAttribution.startedOperation(
                operationName: "qwen35_moe_model_load", attributionEnabled: self.attributionEnabled);
        let loadedModel: Qwen35MoEModel = Qwen35MoEModel(upstreamConfiguration);
        try loadedModel.prepare();
        ServingPerformanceAttribution.endedOperation(
            operationName: "qwen35_moe_model_load",
            operationStart: modelLoadStart,
            attributionEnabled: self.attributionEnabled);
        self.moeModel = loadedModel;
        self.totalLayerCount = repositoryConfiguration.layerCount();
        self.moeRepositoryConfiguration = repositoryConfiguration;
        self.expertResidency = try Qwen35MoeResidentExpertResidency(
            repositoryConfiguration: repositoryConfiguration);
    }

    /**
     * Installs paged expert execution over the loaded model: every MoE
     * layer's upstream SwitchGLU is swapped for the paged primitive that
     * materializes routed-but-missing experts through the given page
     * source, and the plan's retained experts are installed at startup as
     * the layer's resident payload. Must run after `loadInMemoryModel` and
     * before the first generation; a resident install stays untouched when
     * paging is never requested.
     */
    public func installPagedExpertExecution(
        retainedExpertIdsPerLayer: Array<Array<Int>>,
        expertPageMaterializer: any Qwen35MoeExpertPageMaterializing
    ) throws {
        guard let moeModel = self.moeModel, let repositoryConfiguration = self.moeRepositoryConfiguration
        else {
            throw InferenceEngineError.modelLoad(reason: "no MoE model is loaded");
        }
        let pagedResidency: Qwen35MoePagedExpertResidency = try Qwen35MoePagedExpertResidency(
            repositoryConfiguration: repositoryConfiguration,
            retainedExpertIdsPerLayer: retainedExpertIdsPerLayer);
        let forwardContext: Qwen35MoePagedForwardContext = Qwen35MoePagedForwardContext(
            currentInputTokenId: 0);
        let pagingDecorator: Qwen35MoePagedExpertDecorator = Qwen35MoePagedExpertDecorator(
            expertPageMaterializer: expertPageMaterializer,
            routeObservationRing: RouteObservationRing(
                capacity: RouteObservationRing.defaultObservationCapacity),
            attributionEnabled: self.attributionEnabled);

        let switchMluPathsByLayerIndex: Array<String> = try
            Qwen35MoeEngine.switchGluInstallations(
                model: moeModel, expectedLayerCount: Int(pagedResidency.totalLayerCount));
        // Paged decode mutates the switch_mlp modules on every step
        // (materialization), and a compiled trace freezes the module graph it
        // was built from — upstream invalidates traces on module replacement,
        // and the re-trace would capture the decorator and abort on its
        // in-transform eval. Disabling MLX compile process-wide makes every
        // CompiledTrace evaluate eagerly (the gate sits at trace-build time
        // and CompiledTrace compiles lazily on first call), which is the only
        // execution mode that honors per-step module mutation. The SSM cache
        // marker in startGeneration handles the model-level decode schedule;
        // this handles the per-block MoE trace. Resident execution never
        // reaches here.
        MLX.compile(enable: false);
        var pagedSwitchGlusByLayerIndexByPath: Array<(String, Module)> = [];
        for (layerIndex, retainedExpertIds) in retainedExpertIdsPerLayer.enumerated() {
            let switchMluPath: String = switchMluPathsByLayerIndex[layerIndex];
            let pagedSwitchGlu: Qwen35MoePagedSwitchGLU = Qwen35MoePagedSwitchGLU(
                inputDims: Int(repositoryConfiguration.hiddenSize()),
                hiddenDims: Int(repositoryConfiguration.expertIntermediateSize()),
                numExperts: Int(repositoryConfiguration.expertCount()),
                decoderLayerIndex: layerIndex,
                pagingDecorator: pagingDecorator,
                forwardContext: forwardContext);
            try pagingDecorator.installRetainedExperts(
                layerIndex: layerIndex,
                expertIds: retainedExpertIds,
                switchGlu: pagedSwitchGlu);
            pagedSwitchGlusByLayerIndexByPath.append((switchMluPath, pagedSwitchGlu));
        }
        // One update call over every layer at once: an unflattened single-layer
        // path leaves the sibling decoder-layer slots as `.none`, which the
        // upstream array traversal rejects as an unexpected structure.
        try moeModel.update(
            modules: ModuleChildren.unflattened(pagedSwitchGlusByLayerIndexByPath),
            verify: [.all]);

        self.pagedForwardContext = forwardContext;
        self.pagedExpertDecorator = pagingDecorator;
        self.expertResidency = pagedResidency;
    }

    /// Count of routed-expert observations the paged primitive has recorded
    /// across every layer forward so far — the seam hermetic journeys use to
    /// prove per-step dispatch stays live (it grows once per eager forward
    /// and freezes under a compiled decode trace).
    public func pagedRouteObservationCount() -> Int {
        return self.pagedExpertDecorator?.routeObservationRing.observationCount ?? 0;
    }

    /// Total expert-page reads the paged primitive has served so far.
    public func pagedExpertPageReadCount() -> UInt64 {
        return self.pagedExpertDecorator?.totalExpertPageReadCount ?? 0;
    }

    /// Marks the SSM caches so every decode step takes the upstream general
    /// (eager) body instead of the compiled decode segments. The compiled
    /// segments run the paged primitive's materialize-and-update pass only
    /// at trace time, so later steps would silently re-read the first
    /// step's expert rows; upstream's own decode schedule treats a
    /// non-empty SSM mask as the signal that the general path must run,
    /// and a left-padded `MambaCache` produces exactly that mask. The
    /// all-true mask is numerically inert in the gated-delta-net forward
    /// (an all-true `where` selects the original rows), and paged journeys
    /// compare against resident twins built through the same cache
    /// configuration, so paging stays the only difference under test.
    private static func pagedExecutionCaches(_ caches: [KVCache]) -> [KVCache] {
        return caches.map { cache in
            guard cache is MambaCache else {
                return cache;
            }
            return MambaCache(leftPadding: [0]);
        };
    }

    /// Exposes one decoder layer's expert primitive parameter arrays by
    /// parameter basename — the introspection seam hermetic paging journeys
    /// use to seed a page source from a fully resident engine. A pre-install
    /// read observes the loaded resident weights.
    internal func switchGluParameterArrays(layerIndex: Int) throws -> Dictionary<String, MLXArray> {
        guard let moeModel = self.moeModel else {
            throw InferenceEngineError.modelLoad(reason: "no MoE model is loaded");
        }
        let module: Module = moeModel;
        for (modulePath, childModule) in module.namedModules() {
            guard modulePath.hasSuffix(".mlp.switch_mlp") else {
                continue;
            }
            let pathComponents: Array<String> = modulePath.split(separator: ".").map(String.init);
            // The decoder path ends `layers.<index>.mlp.switch_mlp`, so the
            // layer index sits three components from the end.
            guard pathComponents.count >= 3,
                pathComponents[pathComponents.count - 2] == "mlp",
                let componentLayerIndex: Int = Int(pathComponents[pathComponents.count - 3]),
                componentLayerIndex == layerIndex
            else {
                continue;
            }
            guard let switchGlu: SwitchGLU = childModule as? SwitchGLU else {
                throw InferenceEngineError.modelLoad(
                    reason: "the MoE layer primitive at \(modulePath) is not an expert SwitchGLU");
            }
            return Dictionary(uniqueKeysWithValues: switchGlu.parameters().flattened());
        }
        throw InferenceEngineError.modelLoad(
            reason: "the model exposes no MoE layer primitive for layer \(layerIndex)");
    }

    /// Locates every MoE layer's expert primitive by its structural path,
    /// in decoder order; a checkpoint whose MoE layer count disagrees with
    /// the residency plan fails closed instead of paging a partial install.
    /// The located upstream primitives are intentionally discarded — each
    /// is replaced by a fresh paged primitive whose weights come solely
    /// from the page source.
    private static func switchGluInstallations(
        model: any LanguageModel,
        expectedLayerCount: Int
    ) throws -> Array<String> {
        let module: Module = model;
        var switchMluPaths: Array<String> = [];
        for (modulePath, childModule) in module.namedModules() {
            guard modulePath.hasSuffix(".mlp.switch_mlp") else {
                continue;
            }
            guard childModule is SwitchGLU else {
                throw InferenceEngineError.modelLoad(
                    reason: "the MoE layer primitive at \(modulePath) is not an expert SwitchGLU");
            }
            switchMluPaths.append(modulePath);
        }
        guard switchMluPaths.count == expectedLayerCount else {
            throw InferenceEngineError.modelLoad(
                reason: "the model exposes \(switchMluPaths.count) MoE layers but the residency plan names \(expectedLayerCount)");
        }
        return switchMluPaths;
    }

    public func load() throws -> EngineLoadResult {
        guard let expertResidency = self.expertResidency, self.moeModel != nil else {
            throw InferenceEngineError.modelLoad(
                reason: "no MoE model is loaded");
        }
        return EngineLoadResult(
            minimumMlxMemoryCeilingBytes: 1,
            expertMemoryMode: expertResidency.expertMemoryMode);
    }

    public func startGeneration(
        _ inferenceRequest: any PreparedInferenceRequest
    ) throws -> EngineGenerationStart {
        guard let moeModel = self.moeModel else {
            throw InferenceEngineError.invalidRequest(reason: "no MoE model is loaded");
        }
        guard let preparedRequest = inferenceRequest as? Qwen35PreparedInferenceRequest else {
            throw InferenceEngineError.invalidRequest(
                reason: "the request does not belong to the MoE Qwen3.5 engine");
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
        self.activeCache = try moeModel.newCache(parameters: nil);
        if self.pagedForwardContext != nil {
            self.activeCache = Qwen35MoeEngine.pagedExecutionCaches(self.activeCache!);
        }
        return EngineGenerationStart(
            cachedTokenCount: 0,
            restoredPromptPrefixTokenCount: 0,
            expertMemoryMode: self.expertResidency?.expertMemoryMode,
            promptProcessingPhase: .target);
    }

    public func decodeNextToken(requestId: RequestId) throws -> GeneratedToken {
        guard let moeModel = self.moeModel, let activeCache = self.activeCache else {
            throw InferenceEngineError.invalidRequest(
                reason: "the engine holds no active MoE request");
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
                residentExpertCount: self.expertResidency?.telemetry.residentExpertCount ?? 0,
                residentExpertPayloadBytes: self.expertResidency?
                    .telemetry.residentExpertPayloadBytes ?? 0,
                mlxMemorySnapshot: nil);
        }
        guard let activeSampler = self.activeSampler, let lastLogits = self.lastLogits else {
            throw InferenceEngineError.fatalExecution(
                reason: "the MoE engine lost its sampler or logits before decode");
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
                operationName: "qwen35_moe_decode_step",
                attributionEnabled: self.attributionEnabled);
        self.pagedForwardContext?.currentInputTokenId = UInt32(clamping: selectedTokenId);
        let decodeOutput: LMOutput = moeModel(
            LMInput.Text(tokens: MLXArray([selectedTokenId], [1, 1])),
            cache: activeCache,
            state: nil);
        try self.throwIfPagedForwardFaulted();
        let firstDecodeForwardElapsedMillis: UInt64? = self.hasEmittedFirstDecode
            ? nil : Qwen35MoeEngine.millisSince(decodeForwardStart);
        self.hasEmittedFirstDecode = true;
        self.lastLogits = decodeOutput.logits;
        ServingPerformanceAttribution.endedOperation(
            operationName: "qwen35_moe_decode_step",
            operationStart: decodeForwardStart,
            attributionEnabled: self.attributionEnabled);
        return .tokenId(
            generatedTokenId: UInt32(clamping: selectedTokenId),
            isReasoningToken: isReasoningToken,
            expertMemoryMode: self.expertResidency?.expertMemoryMode,
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
        guard let moeModel = self.moeModel else {
            throw InferenceEngineError.invalidRequest(reason: "no MoE model is loaded");
        }
        let chunkStart: ContinuousClock.Instant? = ServingPerformanceAttribution
            .startedOperation(
                operationName: "qwen35_moe_prefill_chunk",
                attributionEnabled: self.attributionEnabled);
        if self.prefillStartClock == nil {
            self.prefillStartClock = ContinuousClock.now;
        }
        let chunkEnd: Int = min(
            self.prefillNextTokenOffset + self.prefillChunkTokenCount,
            self.promptTokenIds.count);
        let chunkTokenIds: Array<UInt32> = Array(
            self.promptTokenIds[self.prefillNextTokenOffset ..< chunkEnd]);
        self.pagedForwardContext?.currentInputTokenId = chunkTokenIds.first ?? 0;
        let chunkOutput: LMOutput = moeModel(
            LMInput.Text(tokens: MLXArray(chunkTokenIds, [1, chunkTokenIds.count])),
            cache: activeCache,
            state: nil);
        try self.throwIfPagedForwardFaulted();
        self.prefillNextTokenOffset = chunkEnd;
        if self.prefillNextTokenOffset >= self.promptTokenIds.count {
            self.lastLogits = chunkOutput.logits;
        }
        let chunkElapsedMillis: UInt64 = Qwen35MoeEngine.millisSince(chunkStart);
        self.prefillElapsedMillis = self.prefillElapsedMillis.addingReportingOverflow(
            chunkElapsedMillis).partialValue;
        ServingPerformanceAttribution.endedOperation(
            operationName: "qwen35_moe_prefill_chunk",
            operationStart: chunkStart,
            attributionEnabled: self.attributionEnabled);
        return .prefillProgress(
            processedTokenCount: UInt32(chunkTokenIds.count),
            elapsedMillis: chunkElapsedMillis,
            forwardPrefillChunkElapsedMillis: chunkElapsedMillis,
            completedPrefillChunkTokens: UInt32(chunkTokenIds.count),
            mlxMemorySnapshot: nil,
            expertResidencyTelemetry: self.expertResidency?.telemetry,
            expertMemoryMode: self.expertResidency?.expertMemoryMode,
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
            expertMemoryMode: self.expertResidency?.expertMemoryMode,
            mlxMemorySnapshot: finalizedSnapshot,
            expertResidencyTelemetry: self.expertResidency?.telemetry);
    }

    public func collectMlxMemorySnapshot() -> WorkerMlxMemorySnapshot? {
        let memoryObservation: Memory.Snapshot = Memory.snapshot();
        let observedActiveBytes: UInt64 = UInt64(max(0, memoryObservation.activeMemory));
        return WorkerMlxMemorySnapshot(
            source: .idlePoll,
            activeMemoryBytes: observedActiveBytes,
            allocatorCacheMemoryBytes: UInt64(max(0, memoryObservation.cacheMemory)),
            peakMemoryBytes: UInt64(max(0, memoryObservation.peakMemory)),
            expertPayloadBytes: self.expertResidency?.telemetry.residentExpertPayloadBytes ?? 0,
            modelCorePayloadBytes: observedActiveBytes,
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

    /// Aborts the generation fail-closed when any paged layer forward
    /// recorded a fault: the returned expert outputs are placeholders, so
    /// nothing sampled from them may reach the caller.
    private func throwIfPagedForwardFaulted() throws -> Void {
        guard let pagedForwardContext = self.pagedForwardContext,
            let recordedFault = pagedForwardContext.consumeRecordedForwardFault()
        else {
            return;
        }
        throw InferenceEngineError.fatalExecution(
            reason: "the paged expert forward failed: \(recordedFault)");
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
