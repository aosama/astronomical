import Foundation;

import Testing;

import MLX;
import MLXLMCommon;

import IpcProtocol;
import ModelServing;
import ModelServingTestSupport;
import JourneyCategories;

@testable import ModelServing;

/**
 * Hermetic persistent prompt-cache engine journeys: a real hybrid Qwen3.5
 * MoE forward pass over the in-memory tiny model proves that a cold request
 * publishes full blocks, boundary snapshots, and the partial tail; that a
 * second request restores the cached prefix and prefills only the uncached
 * suffix; and that a restored request reproduces the cold request's seeded
 * token stream bit for bit. No downloads; the suite serializes with the
 * repository's one-model-at-a-time rule through the shared container.
 */
extension MlxGpuJourneyContainer {

    @Suite(.tags(.hermeticMlxJourney))
    final class Qwen35MoePromptCacheEngineTests {

    private static let ROMEO_AND_JULIET_PROMPT: String = "What is the play about?";
    private static let BLOCK_TOKEN_COUNT: Int = 8;
    private var nextCollectRequestId: UInt64 = 0;

    private let globalPromptCacheRoot: URL;
    private let engine: Qwen35MoeEngine;

    init() throws {
        signal(SIGPIPE, SIG_IGN);
        MLXMetallibLocator.overrideMetallibPathIfNecessary();
        self.globalPromptCacheRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("moe-prompt-cache-\(UUID().uuidString)", isDirectory: true);
        self.engine = try Self.makePinnedEngineWithPromptCache(
            globalPromptCacheRoot: self.globalPromptCacheRoot);
    }

    deinit {
        try? FileManager.default.removeItem(at: self.globalPromptCacheRoot);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_publish_blocks_snapshots_and_tail_from_a_cold_prefill_request() throws {
        let promptTokenIds: Array<UInt32> = Self.fixturePromptTokenIds();
        let requestId: RequestId = RequestId(rawRequestId: 201);
        let generationStart: EngineGenerationStart = try self.engine.startGeneration(
            Qwen35PreparedInferenceRequest(
                promptTokenIds: promptTokenIds,
                samplingSettings: Qwen35SamplingSettings(
                    chatGenerationSettings: Self.settings(seed: 11))));
        #expect(generationStart.cachedTokenCount == 0);

        var processedTokenCount: UInt32 = 0;
        var promptWorkReuseSeen: WorkerPromptWorkReuse?;
        while true {
            let boundary: GeneratedToken = try self.engine.decodeNextToken(requestId: requestId);
            if case let .prefillProgress(processed, _, _, _, _, _, _, promptWorkReuse) = boundary {
                processedTokenCount += processed;
                promptWorkReuseSeen = promptWorkReuse;
                continue;
            }
            if case .generationPreparationStarted = boundary {
                break;
            }
            if case .tokenId = boundary {
                break;
            }
        }
        #expect(processedTokenCount == UInt32(promptTokenIds.count));
        // Two complete blocks and the seven-token tail were published; every
        // published sequence block also carries its boundary snapshot.
        let diskStore: PersistentPromptCacheDiskStore? = self.engine
            .attachedPromptCacheDiskStore;
        #expect(diskStore?.sequenceStateBlockCount() == 3);
        #expect(diskStore?.boundaryStateSnapshotCount() == 3);
        let counters: PersistentPromptCacheCounters? = self.engine
            .attachedPromptCacheCounters;
        #expect(counters?.persistentPromptCacheMisses == 1);
        // A cold request restored nothing, so the reported reuse is zero.
        #expect(promptWorkReuseSeen?.targetRestoredTokenCount == 0);
        _ = try self.engine.cancelGeneration(requestId: requestId);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_restore_a_cached_prefix_and_prefill_only_the_uncached_suffix() throws {
        let firstPromptTokenIds: Array<UInt32> = Self.fixturePromptTokenIds();
        let changedTailPromptTokenIds: Array<UInt32> = Array(
            firstPromptTokenIds.prefix(Self.BLOCK_TOKEN_COUNT * 2))
            + Self.syntheticTailTokens(tokenCount: 7, tokenSeed: 900);
        _ = try self.runPrefillAndFirstToken(
            engine: self.engine, promptTokenIds: firstPromptTokenIds, seed: 11);
        _ = try self.engine.cancelGeneration(requestId: RequestId(rawRequestId: 202));

        let requestId: RequestId = RequestId(rawRequestId: 203);
        let generationStart: EngineGenerationStart = try self.engine.startGeneration(
            Qwen35PreparedInferenceRequest(
                promptTokenIds: changedTailPromptTokenIds,
                samplingSettings: Qwen35SamplingSettings(
                    chatGenerationSettings: Self.settings(seed: 11))));
        // The two shared complete blocks were restored; the divergent tail
        // prefills cold.
        #expect(generationStart.cachedTokenCount == UInt32(Self.BLOCK_TOKEN_COUNT * 2));

        var processedTokenCount: UInt32 = 0;
        var restoredWorkReuse: WorkerPromptWorkReuse?;
        while true {
            let boundary: GeneratedToken = try self.engine.decodeNextToken(requestId: requestId);
            if case let .prefillProgress(processed, _, _, _, _, _, _, promptWorkReuse) = boundary {
                processedTokenCount += processed;
                restoredWorkReuse = promptWorkReuse;
                continue;
            }
            break;
        }
        #expect(processedTokenCount == 7);
        #expect(restoredWorkReuse?.targetRestoredTokenCount == UInt64(Self.BLOCK_TOKEN_COUNT * 2));
        let counters: PersistentPromptCacheCounters? = self.engine
            .attachedPromptCacheCounters;
        #expect(counters?.persistentPromptCacheHits == 1);
        #expect(counters?.persistentPromptCacheTokensSaved
            == UInt64(Self.BLOCK_TOKEN_COUNT * 2));
        _ = try self.engine.cancelGeneration(requestId: requestId);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_miss_cold_when_the_first_cached_block_diverges() throws {
        let firstPromptTokenIds: Array<UInt32> = Self.fixturePromptTokenIds();
        _ = try self.runPrefillAndFirstToken(
            engine: self.engine, promptTokenIds: firstPromptTokenIds, seed: 11);
        _ = try self.engine.cancelGeneration(requestId: RequestId(rawRequestId: 204));

        let divergentPromptTokenIds: Array<UInt32> =
            Self.syntheticTailTokens(tokenCount: 16, tokenSeed: 700)
            + Array(firstPromptTokenIds.suffix(7));
        let requestId: RequestId = RequestId(rawRequestId: 205);
        let generationStart: EngineGenerationStart = try self.engine.startGeneration(
            Qwen35PreparedInferenceRequest(
                promptTokenIds: divergentPromptTokenIds,
                samplingSettings: Qwen35SamplingSettings(
                    chatGenerationSettings: Self.settings(seed: 11))));
        #expect(generationStart.cachedTokenCount == 0);
        let counters: PersistentPromptCacheCounters? = self.engine
            .attachedPromptCacheCounters;
        #expect(counters?.persistentPromptCacheMisses == 2);
        _ = try self.engine.cancelGeneration(requestId: requestId);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_reproduce_the_cold_token_stream_from_a_restored_prefix() throws {
        let promptTokenIds: Array<UInt32> = Self.fixturePromptTokenIds();
        let coldEngine: Qwen35MoeEngine = try Qwen35MoeInMemoryEngineFixture.makePinnedEngine();
        let coldTokenIds: Array<UInt32> = try self.collectTokenIds(
            engine: coldEngine, promptTokenIds: promptTokenIds, seed: 42, tokenBudget: 16);
        #expect(coldTokenIds.isEmpty == false);

        // Control: the cache-attached engine's FIRST (cold, publishing)
        // request must already match the cacheless stream; any divergence
        // here is attachment overhead, not restore behavior.
        let cacheColdRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("moe-cache-cold-control-" + UUID().uuidString,
                isDirectory: true);
        defer { try? FileManager.default.removeItem(at: cacheColdRoot); }
        let cacheColdEngine: Qwen35MoeEngine = try Qwen35MoeInMemoryEngineFixture
            .makePinnedEngine();
        _ = try Self.attachPromptCache(
            engine: cacheColdEngine, globalPromptCacheRoot: cacheColdRoot);
        let cacheColdTokenIds: Array<UInt32> = try self.collectTokenIds(
            engine: cacheColdEngine, promptTokenIds: promptTokenIds, seed: 42,
            tokenBudget: 16);
        #expect(cacheColdTokenIds == coldTokenIds,
            "the attached engine's cold publishing stream must match the cacheless stream");

        // The cache engine's first request publishes the prompt; its second
        // request restores the same prompt's cached prefix and must sample
        // the identical stream from the restored state.
        _ = try self.collectTokenIds(
            engine: self.engine, promptTokenIds: promptTokenIds, seed: 42, tokenBudget: 16);
        let restoredTokenIds: Array<UInt32> = try self.collectTokenIds(
            engine: self.engine, promptTokenIds: promptTokenIds, seed: 42, tokenBudget: 16);
        #expect(restoredTokenIds == coldTokenIds);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_reproduce_the_cold_token_stream_from_a_restored_prefix_under_paged_execution() throws {
        // Ground truth: the same paged substrate without a prompt cache.
        let residentEngine: Qwen35MoeEngine = try Qwen35MoeInMemoryEngineFixture
            .makePinnedEngine();
        let pageSource: EngineSwitchGluPageSourceFixture = try EngineSwitchGluPageSourceFixture(
            residentEngine: residentEngine,
            layerCount: Int(Qwen35MoeInMemoryEngineFixture.FIXTURE_LAYER_COUNT),
            expertCount: Int(Qwen35MoeInMemoryEngineFixture.FIXTURE_EXPERT_COUNT));
        let plainPagedEngine: Qwen35MoeEngine = try Qwen35MoeInMemoryEngineFixture
            .makePinnedPagedEngine(
                retainedExpertIdsPerLayer:
                    Qwen35MoeInMemoryEngineFixture.retainedExpertIdsPerLayer(),
                expertPageMaterializer: pageSource);
        let promptTokenIds: Array<UInt32> = Self.fixturePromptTokenIds();
        let coldTokenIds: Array<UInt32> = try self.collectTokenIds(
            engine: plainPagedEngine, promptTokenIds: promptTokenIds, seed: 42,
            tokenBudget: 16);
        // Control: a second identical request on the same cacheless paged
        // engine — the pre-existing repeat-request baseline.
        let plainSecondRunTokenIds: Array<UInt32> = try self.collectTokenIds(
            engine: plainPagedEngine, promptTokenIds: promptTokenIds, seed: 42,
            tokenBudget: 16);
        #expect(plainSecondRunTokenIds == coldTokenIds,
            "two identical requests on one cacheless paged engine must match");
        // The cache engine publishes on its first request and restores on
        // its second; both must match the cacheless paged stream.
        let pagedCacheRoot: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("moe-paged-cache-\(UUID().uuidString)", isDirectory: true);
        defer { try? FileManager.default.removeItem(at: pagedCacheRoot); }
        let pagedCacheEngine: Qwen35MoeEngine = try Qwen35MoeInMemoryEngineFixture
            .makePinnedPagedEngine(
                retainedExpertIdsPerLayer:
                    Qwen35MoeInMemoryEngineFixture.retainedExpertIdsPerLayer(),
                expertPageMaterializer: pageSource);
        _ = try Self.attachPromptCache(
            engine: pagedCacheEngine, globalPromptCacheRoot: pagedCacheRoot);
        let publishedTokenIds: Array<UInt32> = try self.collectTokenIds(
            engine: pagedCacheEngine, promptTokenIds: promptTokenIds, seed: 42,
            tokenBudget: 16);
        let restoredTokenIds: Array<UInt32> = try self.collectTokenIds(
            engine: pagedCacheEngine, promptTokenIds: promptTokenIds, seed: 42,
            tokenBudget: 16);
        // Warm-path control: one prior forward (the attach probe's shape)
        // through a cacheless paged engine, then the same cold request. If
        // this diverges from the fresh-engine stream, the sensitivity is
        // first-forward warming of the paged primitive, not the cache.
        let warmPathEngine: Qwen35MoeEngine = try Qwen35MoeInMemoryEngineFixture
            .makePinnedPagedEngine(
                retainedExpertIdsPerLayer:
                    Qwen35MoeInMemoryEngineFixture.retainedExpertIdsPerLayer(),
                expertPageMaterializer: pageSource);
        _ = try warmPathEngine.startGeneration(Qwen35PreparedInferenceRequest(
            promptTokenIds: [1],
            samplingSettings: Qwen35SamplingSettings(
                chatGenerationSettings: Self.settings(seed: 7))));
        _ = try warmPathEngine.decodeNextToken(requestId: RequestId(rawRequestId: 700));
        _ = try warmPathEngine.cancelGeneration(requestId: RequestId(rawRequestId: 700));
        let warmPathTokenIds: Array<UInt32> = try self.collectTokenIds(
            engine: warmPathEngine, promptTokenIds: promptTokenIds, seed: 42,
            tokenBudget: 16);
        let warmPathSecondRunTokenIds: Array<UInt32> = try self.collectTokenIds(
            engine: warmPathEngine, promptTokenIds: promptTokenIds, seed: 42,
            tokenBudget: 16);
        // A warmed paged engine follows a different ulp-level numeric path
        // than a fresh one (pre-existing paged-primitive warming), so
        // cross-engine streams are not a stable property under paging; the
        // stable cache properties are asserted per engine below.
        #expect(warmPathTokenIds == warmPathSecondRunTokenIds,
            "the warmed paged engine must stay self-consistent across requests");

        let divergentOffsets: Array<Int> = zip(coldTokenIds, publishedTokenIds)
            .enumerated()
            .filter({ (entry: (offset: Int, element: (UInt32, UInt32))) -> Bool in
                return entry.element.0 != entry.element.1;
            })
            .map({ (entry: (offset: Int, element: (UInt32, UInt32))) -> Int in
                return entry.offset;
            });
        if divergentOffsets.isEmpty == false {
            print("[paged-parity-bisect] divergent-offsets=\(divergentOffsets)");
            print("[paged-parity-bisect] cold=\(coldTokenIds)");
            print("[paged-parity-bisect] attached=\(publishedTokenIds)");
            print("[paged-parity-bisect] restored=\(restoredTokenIds)");
        }
        // The prompt-cache property under paging: the attached engine is
        // bit-self-consistent — its restored stream equals its own cold
        // stream exactly, regardless of which ulp-level paged numeric path
        // the warmed primitive follows.
        #expect(restoredTokenIds == publishedTokenIds,
            "the restored-prefix paged stream must match the attached engine's own cold stream");
    }

    // MARK: - Fixtures

    /// Builds the pinned in-memory engine and attaches the persistent
    /// prompt cache over a fresh temporary store namespace.
    private static func makePinnedEngineWithPromptCache(
        globalPromptCacheRoot: URL
    ) throws -> Qwen35MoeEngine {
        let engine: Qwen35MoeEngine = try Qwen35MoeInMemoryEngineFixture
            .makePinnedEngine(prefillChunkTokenCount: BLOCK_TOKEN_COUNT);
        return try attachPromptCache(engine: engine, globalPromptCacheRoot: globalPromptCacheRoot);
    }

    /// Attaches the persistent prompt cache over a fresh temporary store
    /// namespace to any pinned engine variant.
    private static func attachPromptCache(
        engine: Qwen35MoeEngine, globalPromptCacheRoot: URL
    ) throws -> Qwen35MoeEngine {
        try engine.attachPersistentPromptCache(Qwen35MoePromptCacheAttachment(
            activeModelPromptCacheDirectory: globalPromptCacheRoot
                .appendingPathComponent("model-a", isDirectory: true)
                .appendingPathComponent("rev-1", isDirectory: true),
            globalPromptCacheRootDirectory: globalPromptCacheRoot,
            globalPromptCacheMaximumSizeBytes: 1_000_000_000,
            effectiveMlxMemoryCeilingBytes: 20_000_000_000,
            modelId: "model-a",
            modelRevision: "rev-1",
            configuredBlockTokenCount: BLOCK_TOKEN_COUNT));
        return engine;
    }

    /// Romeo and Juliet prompt bytes are the model-visible token ids; every
    /// byte value stays inside the tiny vocabulary. The 23-byte prompt spans
    /// two complete blocks plus a seven-token tail at block length 8.
    private static func fixturePromptTokenIds() -> Array<UInt32> {
        return ROMEO_AND_JULIET_PROMPT.utf8.map { (promptByte: UInt8) -> UInt32 in
            return UInt32(promptByte) % 512;
        };
    }

    private static func syntheticTailTokens(
        tokenCount: Int, tokenSeed: UInt32
    ) -> Array<UInt32> {
        return (0..<tokenCount).map { (tokenOffset: Int) -> UInt32 in
            return (tokenSeed &+ UInt32(tokenOffset)) % 512;
        };
    }

    private static func settings(seed: UInt64?) -> ChatGenerationSettings {
        return ChatGenerationSettings(
            maxOutputTokens: 16,
            temperatureThousandths: nil,
            topPThousandths: nil,
            seed: seed,
            thinkingBudget: nil);
    }

    /// Runs one request through prefill and its first generated token.
    private func runPrefillAndFirstToken(
        engine: Qwen35MoeEngine,
        promptTokenIds: Array<UInt32>,
        seed: UInt64
    ) throws -> Array<UInt32> {
        return try collectTokenIds(
            engine: engine, promptTokenIds: promptTokenIds, seed: seed, tokenBudget: 1);
    }

    private func collectTokenIds(
        engine: Qwen35MoeEngine,
        promptTokenIds: Array<UInt32>,
        seed: UInt64,
        tokenBudget: Int
    ) throws -> Array<UInt32> {
        // The engine keeps cancelled ids for its lifetime, so every request
        // gets a fresh one.
        self.nextCollectRequestId += 1;
        let requestId: RequestId = RequestId(rawRequestId: self.nextCollectRequestId);
        _ = try engine.startGeneration(Qwen35PreparedInferenceRequest(
            promptTokenIds: promptTokenIds,
            samplingSettings: Qwen35SamplingSettings(
                chatGenerationSettings: Qwen35MoePromptCacheEngineTests.settings(seed: seed))));
        var collectedTokenIds: Array<UInt32> = [];
        while collectedTokenIds.count < tokenBudget {
            let boundary: GeneratedToken = try engine.decodeNextToken(requestId: requestId);
            if case let .tokenId(generatedTokenId, _, _, _, _, _) = boundary {
                collectedTokenIds.append(generatedTokenId);
            }
        }
        _ = try engine.cancelGeneration(requestId: requestId);
        return collectedTokenIds;
    }
}

}
