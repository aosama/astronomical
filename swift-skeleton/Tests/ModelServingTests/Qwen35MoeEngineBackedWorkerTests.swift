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
 * Hermetic engine-backed worker journey over the resident Qwen3.5-MoE
 * engine: a real MoE forward pass driven by the worker loop, with the
 * resident-expert surfaces — the swap mode, the prefill residency snapshot,
 * the preparation barrier counts, and the finalization telemetry — carried
 * all the way to the wire events the supervisor consumes. The suite is
 * serialized so the MLX journeys never overlap (the repository's
 * one-model-at-a-time rule).
 */
extension MlxGpuJourneyContainer {

    @Suite(.tags(.hermeticMlxJourney))
    final class Qwen35MoeEngineBackedWorkerTests {

    static let romeoAndJulietPrompt: String = "What is the play about?";
    static let OUTPUT_TOKEN_BUDGET: UInt16 = 3;

    init() {
        signal(SIGPIPE, SIG_IGN);
        MLXMetallibLocator.overrideMetallibPathIfNecessary();
    }

    @Test(.timeLimit(.minutes(1)))
    func should_pump_resident_moe_expert_telemetry_through_the_worker_wire_events() throws {
        let harness: WorkerHarness = try WorkerHarness.start(
            factory: Qwen35MoeWorkerRuntimeFactory());
        defer { harness.finish(); }
        _ = try harness.expectBootstrappedLifecycle();

        try harness.sendCommand(.swapModel(
            modelDirectory: "/fictional/models/qwen3.5-moe",
            modelConfiguration: Self.autoregressiveModelConfiguration()));

        let swapEvent: WorkerEvent = try harness.expectEvent();
        guard case let .modelSwapped(modelId, _, expertMemoryMode, minimumCeiling) = swapEvent else {
            Issue.record("expected a model swap, got \(swapEvent)");
            return;
        }
        #expect(modelId == Qwen35MoeWorkerChatProcessor.MODEL_IDENTIFIER);
        #expect(expertMemoryMode == .resident);
        #expect(minimumCeiling == 1);

        let policyEvent: WorkerEvent = try harness.expectEvent();
        guard case .runtimeFeatureConfigurationApplied = policyEvent else {
            Issue.record("expected the runtime policy, got \(policyEvent)");
            return;
        }

        let loadedSample: WorkerEvent = try harness.expectEvent();
        guard case let .mlxMemorySample(loadedSnapshot, _) = loadedSample else {
            Issue.record("expected the loaded memory sample, got \(loadedSample)");
            return;
        }
        #expect(loadedSnapshot?.source == .modelLoaded);
        #expect(loadedSnapshot?.expertPayloadBytes
            == Qwen35MoeInMemoryEngineFixture.residentExpertPayloadBytes());

        let expectedResidency: WorkerExpertResidencySnapshot = WorkerExpertResidencySnapshot(
            totalLayerCount: Qwen35MoeInMemoryEngineFixture.FIXTURE_LAYER_COUNT,
            residentExpertCount: Qwen35MoeInMemoryEngineFixture.residentExpertCount(),
            residentExpertPayloadBytes: Qwen35MoeInMemoryEngineFixture.residentExpertPayloadBytes());

        try harness.sendCommand(.generate(Self.moeChatGenerationCommand(
            requestId: 111, maximumOutputTokens: Self.OUTPUT_TOKEN_BUDGET)));

        let promptTokenCount: UInt32 = UInt32(Self.romeoAndJulietPrompt.utf8.count);
        var sawResidentPrefill: Bool = false;
        var sawResidentPreparation: Bool = false;
        var sawFirstDecode: Bool = false;
        var sawResidentFinalization: Bool = false;
        var generatedTokenCount: UInt16 = 0;
        var outputWords: Array<String> = [];
        var completion: WorkerEvent?;
        for _ in 0..<40 {
            let workerEvent: WorkerEvent = try harness.expectEvent();
            switch workerEvent {
            case let .prefillProgress(_, _, processedTokens, totalTokens, _, _, _, _, expertResidency):
                #expect(totalTokens == promptTokenCount);
                if processedTokens == 0 {
                    // The zero-token start announcement is emitted before any expert page access,
                    // so its residency snapshot is legitimately absent.
                    #expect(expertResidency == nil);
                } else {
                    #expect(expertResidency == expectedResidency);
                    sawResidentPrefill = true;
                }
            case let .generationPreparationStarted(
                _, totalLayerCount, residentExpertCount, residentExpertPayloadBytes, _):
                #expect(totalLayerCount == Qwen35MoeInMemoryEngineFixture.FIXTURE_LAYER_COUNT);
                #expect(residentExpertCount == Qwen35MoeInMemoryEngineFixture.residentExpertCount());
                #expect(residentExpertPayloadBytes
                    == Qwen35MoeInMemoryEngineFixture.residentExpertPayloadBytes());
                sawResidentPreparation = true;
            case .firstDecodeCompleted:
                sawFirstDecode = true;
            case let .output(_, _, _, outputs, _, _):
                for output: ChatGenerationOutput in outputs {
                    if case let .text(text: outputWord) = output {
                        outputWords.append(outputWord);
                    }
                }
            case let .generationFinalized(_, finalizedExpertMode, _, expertResidency):
                #expect(finalizedExpertMode == .resident);
                #expect(expertResidency == expectedResidency);
                sawResidentFinalization = true;
            case let .completed(
                _, completedPromptTokenCount, completedGeneratedTokenCount,
                reasoningTokenCount, cachedTokenCount, _, _):
                #expect(completedPromptTokenCount == promptTokenCount);
                #expect(reasoningTokenCount == 0);
                #expect(cachedTokenCount == 0);
                generatedTokenCount = completedGeneratedTokenCount;
                completion = workerEvent;
            case let .failed(_, failureReason):
                Issue.record("the MoE worker journey failed with \(failureReason)");
                return;
            default:
                continue;
            }
            if completion != nil {
                break;
            }
        }

        guard let completion = completion else {
            Issue.record("the MoE worker journey never completed");
            return;
        }
        guard case let .completed(
            _, _, completedGeneratedTokenCount, _, _, _, completionReason) = completion else {
            Issue.record("expected the completion event, got \(completion)");
            return;
        }
        #expect(sawResidentPrefill);
        #expect(sawResidentPreparation);
        #expect(sawFirstDecode);
        #expect(sawResidentFinalization);
        #expect(completedGeneratedTokenCount == UInt32(Self.OUTPUT_TOKEN_BUDGET));
        #expect(generatedTokenCount == completedGeneratedTokenCount);
        #expect(completionReason == .maximumOutputTokens);
        #expect(outputWords == Array(
            EngineBackedWorkerTests.romeoAndJulietOutputWords.prefix(
                Int(Self.OUTPUT_TOKEN_BUDGET))));
    }

    @Test(.timeLimit(.minutes(1)))
    func should_pump_paged_moe_expert_telemetry_through_the_worker_wire_events() throws {
        let harness: WorkerHarness = try WorkerHarness.start(
            factory: Qwen35MoePagedWorkerRuntimeFactory());
        defer { harness.finish(); }
        _ = try harness.expectBootstrappedLifecycle();

        try harness.sendCommand(.swapModel(
            modelDirectory: "/fictional/models/qwen3.5-moe",
            modelConfiguration: Self.autoregressiveModelConfiguration()));

        let swapEvent: WorkerEvent = try harness.expectEvent();
        guard case let .modelSwapped(modelId, _, expertMemoryMode, minimumCeiling) = swapEvent else {
            Issue.record("expected a model swap, got \(swapEvent)");
            return;
        }
        #expect(modelId == Qwen35MoeWorkerChatProcessor.MODEL_IDENTIFIER);
        #expect(expertMemoryMode == .hybrid);
        #expect(minimumCeiling == 1);

        let policyEvent: WorkerEvent = try harness.expectEvent();
        guard case .runtimeFeatureConfigurationApplied = policyEvent else {
            Issue.record("expected the runtime policy, got \(policyEvent)");
            return;
        }

        let loadedSample: WorkerEvent = try harness.expectEvent();
        guard case let .mlxMemorySample(loadedSnapshot, _) = loadedSample else {
            Issue.record("expected the loaded memory sample, got \(loadedSample)");
            return;
        }
        #expect(loadedSnapshot?.source == .modelLoaded);
        #expect(loadedSnapshot?.expertPayloadBytes
            == Qwen35MoeInMemoryEngineFixture.retainedExpertPayloadBytes());

        let expectedResidency: WorkerExpertResidencySnapshot = WorkerExpertResidencySnapshot(
            totalLayerCount: Qwen35MoeInMemoryEngineFixture.FIXTURE_LAYER_COUNT,
            residentExpertCount: Qwen35MoeInMemoryEngineFixture.retainedExpertCount(),
            residentExpertPayloadBytes: Qwen35MoeInMemoryEngineFixture.retainedExpertPayloadBytes());

        try harness.sendCommand(.generate(Self.moeChatGenerationCommand(
            requestId: 112, maximumOutputTokens: Self.OUTPUT_TOKEN_BUDGET)));

        let promptTokenCount: UInt32 = UInt32(Self.romeoAndJulietPrompt.utf8.count);
        var sawHybridPrefill: Bool = false;
        var sawHybridPreparation: Bool = false;
        var sawFirstDecode: Bool = false;
        var sawHybridFinalization: Bool = false;
        var generatedTokenCount: UInt16 = 0;
        var outputWords: Array<String> = [];
        var completion: WorkerEvent?;
        for _ in 0..<40 {
            let workerEvent: WorkerEvent = try harness.expectEvent();
            switch workerEvent {
            case let .prefillProgress(_, _, processedTokens, totalTokens, _, _, _, _, expertResidency):
                #expect(totalTokens == promptTokenCount);
                if processedTokens == 0 {
                    // The zero-token start announcement is emitted before any expert page access,
                    // so its residency snapshot is legitimately absent.
                    #expect(expertResidency == nil);
                } else {
                    #expect(expertResidency == expectedResidency);
                    sawHybridPrefill = true;
                }
            case let .generationPreparationStarted(
                _, totalLayerCount, residentExpertCount, residentExpertPayloadBytes, _):
                #expect(totalLayerCount == Qwen35MoeInMemoryEngineFixture.FIXTURE_LAYER_COUNT);
                #expect(residentExpertCount == Qwen35MoeInMemoryEngineFixture.retainedExpertCount());
                #expect(residentExpertPayloadBytes
                    == Qwen35MoeInMemoryEngineFixture.retainedExpertPayloadBytes());
                sawHybridPreparation = true;
            case .firstDecodeCompleted:
                sawFirstDecode = true;
            case let .output(_, _, _, outputs, _, _):
                for output: ChatGenerationOutput in outputs {
                    if case let .text(text: outputWord) = output {
                        outputWords.append(outputWord);
                    }
                }
            case let .generationFinalized(_, finalizedExpertMode, _, expertResidency):
                #expect(finalizedExpertMode == .hybrid);
                #expect(expertResidency == expectedResidency);
                sawHybridFinalization = true;
            case let .completed(
                _, completedPromptTokenCount, completedGeneratedTokenCount,
                reasoningTokenCount, cachedTokenCount, _, _):
                #expect(completedPromptTokenCount == promptTokenCount);
                #expect(reasoningTokenCount == 0);
                #expect(cachedTokenCount == 0);
                generatedTokenCount = completedGeneratedTokenCount;
                completion = workerEvent;
            case let .failed(_, failureReason):
                Issue.record("the paged MoE worker journey failed with \(failureReason)");
                return;
            default:
                continue;
            }
            if completion != nil {
                break;
            }
        }

        guard let completion = completion else {
            Issue.record("the paged MoE worker journey never completed");
            return;
        }
        guard case .completed = completion else {
            Issue.record("expected the completion event, got \(completion)");
            return;
        }
        #expect(sawHybridPrefill);
        #expect(sawHybridPreparation);
        #expect(sawFirstDecode);
        #expect(sawHybridFinalization);
        #expect(generatedTokenCount == UInt32(Self.OUTPUT_TOKEN_BUDGET));
        #expect(outputWords == Array(
            EngineBackedWorkerTests.romeoAndJulietOutputWords.prefix(
                Int(Self.OUTPUT_TOKEN_BUDGET))),
            "the paged journey with the retained-set install must stream the same words as the resident journey");
    }

    // MARK: - Harness

    private static func autoregressiveModelConfiguration() -> WorkerModelConfiguration {
        return WorkerModelConfiguration.autoregressive(WorkerAutoregressiveModelConfiguration(
            modelId: "qwen3.5-moe",
            maximumContextTokens: 4096,
            maximumOutputTokens: 1024,
            chunking: WorkerChunkingConfiguration(
                fixedPromptProcessingChunkSizeTokens: 512,
                fixedSsdStreamingPromptProcessingChunkSizeTokens: 512,
                fullAttentionKeyValueGrowthTokens: 512,
                prefillGraphSubmissionLayerInterval: 1,
                experimentalSsdPagingPrefillGraphSubmissionLayerInterval: 1,
                experimentalSsdPagingGenerationGraphSubmissionLayerInterval: 1,
                promptCacheBlockTokens: nil,
                promptCacheCommonPrefixStrideBlocks: 1,
                experimentalDecodeStageAttributionEnabled: false,
                experimentalQuantizedKvCacheEnabled: false,
                experimentalFusedMoeDecodeEnabled: false)));
    }

    private static func moeChatGenerationCommand(
        requestId: UInt64,
        maximumOutputTokens: UInt16
    ) -> ChatGenerationCommand {
        return ChatGenerationCommand(
            requestId: RequestId(rawRequestId: requestId),
            model: "qwen3.5-moe",
            messages: [.user(content: Self.romeoAndJulietPrompt, images: [])],
            tools: [],
            toolChoice: .auto,
            settings: ChatGenerationSettings(
                maxOutputTokens: maximumOutputTokens,
                temperatureThousandths: nil,
                topPThousandths: nil,
                seed: 42,
                thinkingBudget: nil),
            structuredGeneration: nil);
    }
}

}
