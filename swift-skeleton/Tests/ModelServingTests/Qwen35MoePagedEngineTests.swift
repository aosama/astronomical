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
 * Hermetic paged-expert engine journeys: a paged install reports the hybrid
 * ownership mode with the retained-set telemetry, a paged generation streams
 * bit-identical tokens to its zero-paging twin (the same paged substrate
 * with every expert installed at setup — the only twin whose execution path
 * the paging seams do not alter), a repeated identical generation reads no
 * additional expert pages (the engine-scale issue #629 property), and an
 * invalid paging plan fails closed before any install. The suite is
 * serialized so the MLX journeys never overlap (the repository's
 * one-model-at-a-time rule).
 */
extension MlxGpuJourneyContainer {

    @Suite(.tags(.hermeticMlxJourney))
    final class Qwen35MoePagedEngineTests {

    private static let ROMEO_AND_JULIET_PROMPT: String = "What is the play about?";
    private static let OUTPUT_TOKEN_BUDGET: Int = 5;

    init() {
        signal(SIGPIPE, SIG_IGN);
        MLXMetallibLocator.overrideMetallibPathIfNecessary();
    }

    @Test(.timeLimit(.minutes(1)))
    func should_report_the_hybrid_mode_and_retained_telemetry_from_a_paged_install() throws {
        let pagingFixture: PagedEngineJourneyFixture = try PagedEngineJourneyFixture();
        let loadResult: EngineLoadResult = try pagingFixture.pagedEngine.load();
        #expect(loadResult.expertMemoryMode == .hybrid);

        let expectedTelemetry: ExpertResidencyTelemetry = ExpertResidencyTelemetry(
            totalLayerCount: Qwen35MoeInMemoryEngineFixture.FIXTURE_LAYER_COUNT,
            residentExpertCount: Qwen35MoeInMemoryEngineFixture.retainedExpertCount(),
            residentExpertPayloadBytes: Qwen35MoeInMemoryEngineFixture.retainedExpertPayloadBytes());
        let requestId: RequestId = RequestId(rawRequestId: 71);
        let generationStart: EngineGenerationStart = try pagingFixture.pagedEngine.startGeneration(
            Qwen35PreparedInferenceRequest(
                promptTokenIds: Self.fixturePromptTokenIds(),
                samplingSettings: Qwen35SamplingSettings(
                    chatGenerationSettings: Self.settings(seed: 42))));
        #expect(generationStart.expertMemoryMode == .hybrid);

        var sawHybridPrefillTelemetry: Bool = false;
        var boundaryBudget: Int = 8;
        while sawHybridPrefillTelemetry == false && boundaryBudget > 0 {
            boundaryBudget -= 1;
            let boundary: GeneratedToken = try pagingFixture.pagedEngine.decodeNextToken(
                requestId: requestId);
            guard case let .prefillProgress(
                _, _, _, _, _, expertResidencyTelemetry, expertMemoryMode, _) = boundary
            else {
                continue;
            }
            #expect(expertResidencyTelemetry == expectedTelemetry);
            #expect(expertMemoryMode == .hybrid);
            sawHybridPrefillTelemetry = true;
        }
        #expect(sawHybridPrefillTelemetry);
        _ = try pagingFixture.pagedEngine.cancelGeneration(requestId: requestId);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_stream_a_paged_generation_bit_identical_to_its_zero_paging_twin() throws {
        let pagingFixture: PagedEngineJourneyFixture = try PagedEngineJourneyFixture();
        let pagedJourney: CollectedEngineJourney = try Self.collectTokenIds(
            engine: pagingFixture.pagedEngine, journeyRequestId: 72, seed: 42);
        let zeroPagingJourney: CollectedEngineJourney = try Self.collectTokenIds(
            engine: pagingFixture.fullRetentionPagedEngine, journeyRequestId: 73, seed: 42);

        #expect(pagedJourney.tokenIds.isEmpty == false);
        #expect(pagedJourney.tokenIds == zeroPagingJourney.tokenIds,
            "a paged generation must be bit-identical to the same substrate with zero runtime paging");
        #expect(pagedJourney.boundaryExpertModes.allSatisfy({ $0 == .hybrid }),
            "every paged boundary must report the hybrid ownership mode");
    }

    @Test(.timeLimit(.minutes(1)))
    func should_read_no_expert_pages_on_a_second_identical_paged_generation() throws {
        let pagingFixture: PagedEngineJourneyFixture = try PagedEngineJourneyFixture();
        let firstJourney: CollectedEngineJourney = try Self.collectTokenIds(
            engine: pagingFixture.pagedEngine, journeyRequestId: 74, seed: 42);
        let firstGenerationPageReadTotal: Int = pagingFixture.pageSource.servedExpertPageTotal;

        let secondJourney: CollectedEngineJourney = try Self.collectTokenIds(
            engine: pagingFixture.pagedEngine, journeyRequestId: 75, seed: 42);
        let secondGenerationPageReadTotal: Int = pagingFixture.pageSource.servedExpertPageTotal;

        #expect(firstGenerationPageReadTotal > 0,
            "the first generation must read the routed-but-missing expert pages");
        #expect(secondGenerationPageReadTotal == firstGenerationPageReadTotal,
            "issue #629: a repeated identical generation must read no additional expert pages");
        #expect(secondJourney.tokenIds == firstJourney.tokenIds,
            "the repeated generation must replay the exact token stream");
    }

    @Test(.timeLimit(.minutes(1)))
    func should_fail_closed_on_an_invalid_paging_plan() throws {
        #expect(throws: InferenceEngineError.modelLoad(
            reason: "the paging plan must name retained experts for every decoder layer")) {
            let incompletePlanEngine: Qwen35MoeEngine = try Qwen35MoeInMemoryEngineFixture
                .makePinnedPagedEngine(
                    retainedExpertIdsPerLayer: [[0, 1]],
                    expertPageMaterializer: FailingExpertPageSourceFixture());
            _ = incompletePlanEngine;
        }

        #expect(throws: InferenceEngineError.modelLoad(
            reason: "the paging plan retains expert 9 outside layer 0's expert range")) {
            let outOfRangePlanEngine: Qwen35MoeEngine = try Qwen35MoeInMemoryEngineFixture
                .makePinnedPagedEngine(
                    retainedExpertIdsPerLayer: [[0, 9], [0, 1]],
                    expertPageMaterializer: FailingExpertPageSourceFixture());
            _ = outOfRangePlanEngine;
        }
    }

    // MARK: - Fixtures

    /// The resident twin seeding the page source, the page source itself,
    /// the paged engine whose misses page through the seam, and the zero-
    /// paging twin that installs every expert at setup — the parties the
    /// paged journeys need.
    private final class PagedEngineJourneyFixture {
        let residentEngine: Qwen35MoeEngine;
        let pageSource: EngineSwitchGluPageSourceFixture;
        let pagedEngine: Qwen35MoeEngine;
        let fullRetentionPagedEngine: Qwen35MoeEngine;

        init() throws {
            self.residentEngine = try Qwen35MoeInMemoryEngineFixture.makePinnedEngine();
            self.pageSource = try EngineSwitchGluPageSourceFixture(
                residentEngine: self.residentEngine,
                layerCount: Int(Qwen35MoeInMemoryEngineFixture.FIXTURE_LAYER_COUNT),
                expertCount: Int(Qwen35MoeInMemoryEngineFixture.FIXTURE_EXPERT_COUNT));
            self.pagedEngine = try Qwen35MoeInMemoryEngineFixture.makePinnedPagedEngine(
                retainedExpertIdsPerLayer: Qwen35MoeInMemoryEngineFixture.retainedExpertIdsPerLayer(),
                expertPageMaterializer: self.pageSource);
            self.fullRetentionPagedEngine = try Qwen35MoeInMemoryEngineFixture.makePinnedPagedEngine(
                retainedExpertIdsPerLayer: Qwen35MoeInMemoryEngineFixture.fullyRetainedExpertIdsPerLayer(),
                expertPageMaterializer: self.pageSource);
        }
    }

    /// A page source that fails every consultation; only the plan-validation
    /// journeys reach it, so no consultation ever happens.
    private final class FailingExpertPageSourceFixture: Qwen35MoeExpertPageMaterializing {
        func materializeExpertWeights(
            layerIndex: Int,
            expertIds: Array<Int>
        ) throws -> Qwen35MoeMaterializedExpertWeights {
            throw InferenceEngineError.fatalExecution(
                reason: "the failing page source must never be consulted");
        }
    }

    /// Romeo and Juliet prompt bytes are the model-visible token ids; every
    /// byte value stays inside the tiny vocabulary.
    private static func fixturePromptTokenIds() -> Array<UInt32> {
        return ROMEO_AND_JULIET_PROMPT.utf8.map { (promptByte: UInt8) -> UInt32 in
            return UInt32(promptByte) % 512;
        };
    }

    private static func settings(
        seed: UInt64?
    ) -> ChatGenerationSettings {
        return ChatGenerationSettings(
            maxOutputTokens: 16,
            temperatureThousandths: nil,
            topPThousandths: nil,
            seed: seed,
            thinkingBudget: nil);
    }

    /// The token ids and expert ownership modes one seeded engine journey
    /// produced.
    private struct CollectedEngineJourney {
        let tokenIds: Array<UInt32>;
        let boundaryExpertModes: Array<ExpertMemoryMode>;
    }

    /// Pumps one seeded generation to the token budget and returns the
    /// generated token ids with the ownership modes the token boundaries
    /// reported. Every journey takes its own request id — the engine marks
    /// an id cancelled forever once its generation ends, and production
    /// workers likewise assign a fresh id per generation.
    private static func collectTokenIds(
        engine: Qwen35MoeEngine,
        journeyRequestId: UInt64,
        seed: UInt64
    ) throws -> CollectedEngineJourney {
        let requestId: RequestId = RequestId(rawRequestId: journeyRequestId);
        _ = try engine.startGeneration(Qwen35PreparedInferenceRequest(
            promptTokenIds: fixturePromptTokenIds(),
            samplingSettings: Qwen35SamplingSettings(
                chatGenerationSettings: settings(seed: seed))));
        var collectedTokenIds: Array<UInt32> = [];
        var boundaryExpertModes: Array<ExpertMemoryMode> = [];
        while collectedTokenIds.count < OUTPUT_TOKEN_BUDGET {
            let boundary: GeneratedToken = try engine.decodeNextToken(requestId: requestId);
            switch boundary {
            case let .tokenId(generatedTokenId, _, boundaryExpertMode, _, _, _):
                collectedTokenIds.append(generatedTokenId);
                if let boundaryExpertMode = boundaryExpertMode {
                    boundaryExpertModes.append(boundaryExpertMode);
                }
            default:
                continue;
            }
        }
        _ = try engine.cancelGeneration(requestId: requestId);
        return CollectedEngineJourney(
            tokenIds: collectedTokenIds, boundaryExpertModes: boundaryExpertModes);
    }
}

}
