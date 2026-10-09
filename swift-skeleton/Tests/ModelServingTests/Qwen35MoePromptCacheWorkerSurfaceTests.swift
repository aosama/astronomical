import Foundation;

import Testing;

import IpcProtocol;
import ModelServing;
import ModelServingTestSupport;
import JourneyCategories;

@testable import ModelServing;

/// The block length every worker-surface journey caches at; it matches the
/// pinned fixture engine's prefill chunk so chunks and blocks coincide.
private let WORKER_SURFACE_BLOCK_TOKEN_COUNT: Int = 8;

/// The overcharged per-token context charge the rejection journey reports.
private let OVERCHARGED_CONTEXT_BYTES_PER_TOKEN: Int = 1_000_000;


/**
 * Hermetic persistent prompt-cache worker-surface journeys: the cache
 * policy derived from the worker startup configuration attaches at model
 * swap through the same spawn-policy seam the production factory uses, the
 * second identical request reports its restored prefix on the wire, the
 * idle memory sample carries the prompt-cache stats event, and the clear
 * verb removes the persisted blocks with real counts. The suite is
 * serialized with the repository's one-model-at-a-time rule.
 */
extension MlxGpuJourneyContainer {

    @Suite(.tags(.hermeticMlxJourney))
    final class Qwen35MoePromptCacheWorkerSurfaceTests {

    private static let ROMEO_AND_JULIET_PROMPT: String = "What is the play about?";
    private static let BLOCK_TOKEN_COUNT: Int = WORKER_SURFACE_BLOCK_TOKEN_COUNT;

    private var nextRequestIdValue: UInt64 = 500;

    init() {
        signal(SIGPIPE, SIG_IGN);
        MLXMetallibLocator.overrideMetallibPathIfNecessary();
    }

    @Test(.timeLimit(.minutes(1)))
    func should_report_prompt_cache_stats_and_clear_counts_through_the_worker_wire() throws {
        let globalPromptCacheRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("moe-worker-cache-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: globalPromptCacheRoot); }
        let spawnPolicy: Qwen35MoePromptCacheSpawnPolicy = Qwen35MoePromptCacheSpawnPolicy(
            globalPromptCacheRootDirectory: globalPromptCacheRoot,
            globalPromptCacheMaximumSizeBytes: 1_000_000_000,
            effectiveMlxMemoryCeilingBytes: 20_000_000_000);
        // The ceiling must cover the tiny engine's real warm footprint
        // (process-global MLX active memory), or the governor would
        // correctly reject the second generation's context admission.
        let harness: WorkerHarness = try WorkerHarness.start(
            factory: PromptCacheAttachedWorkerRuntimeFactory(spawnPolicy: spawnPolicy),
            machineMlxMemoryCeilingBytes: 64_000_000,
            globalPromptCacheRootDirectory: globalPromptCacheRoot.path);
        defer { harness.finish(); }
        _ = try harness.expectBootstrappedLifecycle();

        // The swap lifecycle drains through the factory that attaches the
        // cache from the spawn policy.
        try harness.swapModelIn(
            modelDirectory: "/fictional/models/qwen3.5-moe",
            modelConfiguration: Self.autoregressiveModelConfiguration());

        // First request: cold prefill publishes the prompt.
        self.nextRequestIdValue += 1;
        let firstCompletionCachedTokens: UInt32 = try self.runGeneration(
            harness: harness, requestIdValue: self.nextRequestIdValue);
        #expect(firstCompletionCachedTokens == 0);

        // Second identical request: the restored prefix reports on the wire.
        self.nextRequestIdValue += 1;
        let secondCompletionCachedTokens: UInt32 = try self.runGeneration(
            harness: harness, requestIdValue: self.nextRequestIdValue);
        #expect(secondCompletionCachedTokens == UInt32(Self.BLOCK_TOKEN_COUNT * 2));

        // The idle memory sample carries the prompt-cache stats event.
        try harness.sendCommand(.sampleMlxMemory);
        let sampleEvent: WorkerEvent = try harness.expectEvent();
        guard case .mlxMemorySample = sampleEvent else {
            Issue.record("expected the memory sample, got \(sampleEvent)");
            return;
        }
        let statsEvent: WorkerEvent = try harness.expectEvent();
        guard case let .persistentPromptCacheStats(promptCacheStats) = statsEvent else {
            Issue.record("expected the prompt-cache stats, got \(statsEvent)");
            return;
        }
        #expect(promptCacheStats.persistentPromptCacheHits == 1);
        #expect(promptCacheStats.persistentPromptCacheMisses == 1);
        #expect(promptCacheStats.persistentPromptCacheTokensSaved
            == UInt64(Self.BLOCK_TOKEN_COUNT * 2));
        #expect(promptCacheStats.persistentPromptCacheBlockTokenCount
            == UInt64(Self.BLOCK_TOKEN_COUNT));
        // Two complete blocks plus the seven-token tail.
        #expect(promptCacheStats.persistentPromptCacheSequenceStateBlockCount == 3);
        #expect(promptCacheStats.persistentPromptCacheBoundaryStateSnapshotCount == 3);
        #expect(promptCacheStats.persistentPromptCacheMaximumSizeBytes == 1_000_000_000);

        // The clear verb removes the persisted blocks with real counts, and
        // the next stats sample reports the emptied store.
        try harness.sendCommand(.clearPromptCache(modelId: nil));
        let clearedEvent: WorkerEvent = try harness.expectEvent();
        guard case let .promptCacheCleared(clearedModelId, blocksRemoved, _) = clearedEvent else {
            Issue.record("expected the clear acknowledgement, got \(clearedEvent)");
            return;
        }
        #expect(clearedModelId == nil);
        #expect(blocksRemoved == 3);

        try harness.sendCommand(.sampleMlxMemory);
        let secondSampleEvent: WorkerEvent = try harness.expectEvent();
        guard case .mlxMemorySample = secondSampleEvent else {
            Issue.record("expected the second memory sample, got \(secondSampleEvent)");
            return;
        }
        let secondStatsEvent: WorkerEvent = try harness.expectEvent();
        guard case let .persistentPromptCacheStats(emptiedStats) = secondStatsEvent else {
            Issue.record("expected the emptied stats, got \(secondStatsEvent)");
            return;
        }
        #expect(emptiedStats.persistentPromptCacheSequenceStateBlockCount == 0);
        #expect(emptiedStats.persistentPromptCacheBoundaryStateSnapshotCount == 0);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_reject_a_request_whose_context_exceeds_the_wired_budget() throws {
        let globalPromptCacheRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("moe-governor-cache-\(UUID().uuidString)",
                isDirectory: true);
        defer { try? FileManager.default.removeItem(at: globalPromptCacheRoot); }
        let spawnPolicy: Qwen35MoePromptCacheSpawnPolicy = Qwen35MoePromptCacheSpawnPolicy(
            globalPromptCacheRootDirectory: globalPromptCacheRoot,
            globalPromptCacheMaximumSizeBytes: 1_000_000_000,
            effectiveMlxMemoryCeilingBytes: 20_000_000_000);
        // The overcharged engine reports one megabyte of persisted state
        // per context token, so any request's context need exceeds the
        // harness's one-megabyte machine ceiling and the governor must
        // reject it before the engine starts.
        let harness: WorkerHarness = try WorkerHarness.start(
            factory: ContextOverchargedWorkerRuntimeFactory(spawnPolicy: spawnPolicy),
            machineMlxMemoryCeilingBytes: 1_000_000,
            globalPromptCacheRootDirectory: globalPromptCacheRoot.path);
        defer { harness.finish(); }
        _ = try harness.expectBootstrappedLifecycle();
        try harness.swapModelIn(
            modelDirectory: "/fictional/models/qwen3.5-moe",
            modelConfiguration: Self.autoregressiveModelConfiguration());

        self.nextRequestIdValue += 1;
        try harness.sendCommand(.generate(Self.workerChatGenerationCommand(
            requestId: self.nextRequestIdValue)));
        var rejectionRecorded: Bool = false;
        for _ in 0..<10 {
            let workerEvent: WorkerEvent = try harness.expectEvent();
            if case let .failed(_, failureReason) = workerEvent {
                rejectionRecorded = true;
                guard case let .invalidRequest(rejectionReason) = failureReason else {
                    Issue.record("expected an invalid-request rejection, got \(failureReason)");
                    break;
                }
                #expect(rejectionReason.hasPrefix(
                    "the request context exceeds the wired memory budget"),
                    "the governor's deficit reason must reject the request: \(rejectionReason)");
                break;
            }
            if case .completed = workerEvent {
                break;
            }
        }
        #expect(rejectionRecorded,
            "the governor must reject the overcharged context before the engine starts");
    }

    @Test(.timeLimit(.minutes(1)))
    func should_derive_the_attachment_from_the_spawn_policy() throws {
        let globalPromptCacheRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("moe-policy-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: globalPromptCacheRoot); }
        let spawnPolicy: Qwen35MoePromptCacheSpawnPolicy = Qwen35MoePromptCacheSpawnPolicy(
            globalPromptCacheRootDirectory: globalPromptCacheRoot,
            globalPromptCacheMaximumSizeBytes: 2_000_000_000,
            effectiveMlxMemoryCeilingBytes: 20_000_000_000);
        let attachment: Qwen35MoePromptCacheAttachment = spawnPolicy.makeAttachment(
            modelId: "model-a", modelRevision: "rev-1",
            configuredBlockTokenCount: Self.BLOCK_TOKEN_COUNT);
        #expect(attachment.globalPromptCacheRootDirectory == globalPromptCacheRoot);
        #expect(attachment.activeModelPromptCacheDirectory
            == globalPromptCacheRoot.appendingPathComponent("model-a", isDirectory: true)
                .appendingPathComponent("rev-1", isDirectory: true));
        #expect(attachment.configuredBlockTokenCount == Self.BLOCK_TOKEN_COUNT);

        // The derived attachment opens against the policy's namespace.
        let engine: Qwen35MoeEngine = try Qwen35MoeInMemoryEngineFixture.makePinnedEngine();
        try engine.attachPersistentPromptCache(attachment);
        let stats: WorkerPersistentPromptCacheStats? = engine
            .collectPersistentPromptCacheStats();
        #expect(stats?.persistentPromptCacheBlockTokenCount == UInt64(Self.BLOCK_TOKEN_COUNT));
        #expect(stats?.persistentPromptCacheMaximumSizeBytes == 2_000_000_000);
        engine.detachPersistentPromptCacheForJourneys();
    }

    // MARK: - Fixtures

    /// Runs one generation through the worker and returns the completed
    /// request's cached-token count.
    private func runGeneration(
        harness: WorkerHarness, requestIdValue: UInt64
    ) throws -> UInt32 {
        self.nextRequestIdValue = requestIdValue;
        try harness.sendCommand(.generate(Self.workerChatGenerationCommand(
            requestId: requestIdValue)));
        var cachedTokenCount: UInt32? = nil;
        for _ in 0..<40 {
            let workerEvent: WorkerEvent = try harness.expectEvent();
            if case let .completed(_, _, _, _, completedCachedTokenCount, _, _) = workerEvent {
                cachedTokenCount = completedCachedTokenCount;
                break;
            }
            if case let .failed(_, failureReason) = workerEvent {
                Issue.record("the worker prompt-cache journey failed with \(failureReason)");
                return 0;
            }
        }
        guard let completedCachedTokenCount: UInt32 = cachedTokenCount else {
            Issue.record("the worker prompt-cache journey never completed");
            return 0;
        }
        return completedCachedTokenCount;
    }

    private static func workerChatGenerationCommand(
        requestId: UInt64
    ) -> ChatGenerationCommand {
        return ChatGenerationCommand(
            requestId: RequestId(rawRequestId: requestId),
            model: "qwen3.5-moe",
            messages: [.user(content: ROMEO_AND_JULIET_PROMPT, images: [])],
            tools: [],
            toolChoice: .auto,
            settings: ChatGenerationSettings(
                maxOutputTokens: 2,
                temperatureThousandths: nil,
                topPThousandths: nil,
                seed: 42,
                thinkingBudget: nil),
            structuredGeneration: nil);
    }

    private static func autoregressiveModelConfiguration() -> WorkerModelConfiguration {
        return WorkerModelConfiguration.autoregressive(WorkerAutoregressiveModelConfiguration(
            modelId: "qwen3.5-moe",
            maximumContextTokens: 4096,
            maximumOutputTokens: 1024,
            chunking: WorkerChunkingConfiguration(
                fixedPromptProcessingChunkSizeTokens: UInt32(BLOCK_TOKEN_COUNT),
                fixedSsdStreamingPromptProcessingChunkSizeTokens: UInt32(BLOCK_TOKEN_COUNT),
                fullAttentionKeyValueGrowthTokens: UInt32(BLOCK_TOKEN_COUNT),
                prefillGraphSubmissionLayerInterval: 1,
                experimentalSsdPagingPrefillGraphSubmissionLayerInterval: 1,
                experimentalSsdPagingGenerationGraphSubmissionLayerInterval: 1,
                promptCacheBlockTokens: UInt32(BLOCK_TOKEN_COUNT),
                promptCacheCommonPrefixStrideBlocks: 4,
                experimentalDecodeStageAttributionEnabled: false,
                experimentalQuantizedKvCacheEnabled: false,
                experimentalFusedMoeDecodeEnabled: false)));
    }
}

/// The pinned MoE engine with an overcharged context-workspace report, so
/// the governor's admission path is exercised without a real deficit model.
final class ContextOverchargedEngine: InferenceEngine {

    private let baseEngine: Qwen35MoeEngine;

    init(baseEngine: Qwen35MoeEngine) {
        self.baseEngine = baseEngine;
    }

    func load() throws -> EngineLoadResult {
        return try self.baseEngine.load();
    }

    func startGeneration(
        _ inferenceRequest: any PreparedInferenceRequest
    ) throws -> EngineGenerationStart {
        return try self.baseEngine.startGeneration(inferenceRequest);
    }

    func decodeNextToken(requestId: RequestId) throws -> GeneratedToken {
        return try self.baseEngine.decodeNextToken(requestId: requestId);
    }

    func injectInputTokens(requestId: RequestId, inputTokenIds: Array<UInt32>) throws {
        try self.baseEngine.injectInputTokens(requestId: requestId,
            inputTokenIds: inputTokenIds);
    }

    func cancelGeneration(requestId: RequestId) throws -> GenerationFinalization {
        return try self.baseEngine.cancelGeneration(requestId: requestId);
    }

    func collectMlxMemorySnapshot() -> WorkerMlxMemorySnapshot? {
        return self.baseEngine.collectMlxMemorySnapshot();
    }

    func applyMlxMemoryLimit(_ requestedMlxMemoryCeilingBytes: UInt64) throws {
        try self.baseEngine.applyMlxMemoryLimit(requestedMlxMemoryCeilingBytes);
    }

    func contextWorkspaceBytesPerToken() -> Int? {
        return OVERCHARGED_CONTEXT_BYTES_PER_TOKEN;
    }
}

/// Factory mirroring the production spawn: the pinned in-memory MoE runtime
/// gains its prompt cache through the spawn-policy seam at swap time.
struct PromptCacheAttachedWorkerRuntimeFactory: ChatModelRuntimeFactory {

    let spawnPolicy: Qwen35MoePromptCacheSpawnPolicy;

    func createChatRuntime(
        modelDirectory: String,
        modelConfiguration: WorkerModelConfiguration
    ) throws -> LoadedChatRuntime {
        return try self.createChatRuntimeCandidate(
            modelDirectory: modelDirectory,
            modelConfiguration: modelConfiguration).load();
    }

    func createChatRuntimeCandidate(
        modelDirectory: String,
        modelConfiguration: WorkerModelConfiguration
    ) throws -> ChatRuntimeCandidate {
        let baseCandidate: ChatRuntimeCandidate = try Qwen35MoeWorkerRuntimeFactory()
            .createChatRuntimeCandidate(
                modelDirectory: modelDirectory, modelConfiguration: modelConfiguration);
        return ChatRuntimeCandidate {
            let baseRuntime: LoadedChatRuntime = try baseCandidate.load();
            guard let moeEngine: Qwen35MoeEngine = baseRuntime.engine as? Qwen35MoeEngine else {
                return baseRuntime;
            }
            try moeEngine.attachPersistentPromptCache(self.spawnPolicy.makeAttachment(
                modelId: "qwen3.5-moe",
                modelRevision: "journey",
                configuredBlockTokenCount: WORKER_SURFACE_BLOCK_TOKEN_COUNT));
            return baseRuntime;
        };
    }
}

/// Factory whose engine overcharges the context workspace.
struct ContextOverchargedWorkerRuntimeFactory: ChatModelRuntimeFactory {

    let spawnPolicy: Qwen35MoePromptCacheSpawnPolicy;

    func createChatRuntime(
        modelDirectory: String,
        modelConfiguration: WorkerModelConfiguration
    ) throws -> LoadedChatRuntime {
        return try self.createChatRuntimeCandidate(
            modelDirectory: modelDirectory,
            modelConfiguration: modelConfiguration).load();
    }

    func createChatRuntimeCandidate(
        modelDirectory: String,
        modelConfiguration: WorkerModelConfiguration
    ) throws -> ChatRuntimeCandidate {
        let baseCandidate: ChatRuntimeCandidate = try PromptCacheAttachedWorkerRuntimeFactory(
            spawnPolicy: self.spawnPolicy)
            .createChatRuntimeCandidate(
                modelDirectory: modelDirectory, modelConfiguration: modelConfiguration);
        return ChatRuntimeCandidate {
            let baseRuntime: LoadedChatRuntime = try baseCandidate.load();
            guard let moeEngine: Qwen35MoeEngine = baseRuntime.engine as? Qwen35MoeEngine else {
                return baseRuntime;
            }
            return LoadedChatRuntime(
                processor: baseRuntime.processor,
                engine: ContextOverchargedEngine(baseEngine: moeEngine));
        };
    }
}

}
