import Foundation;

import Testing;

import IpcProtocol;
import ModelServing;
import ModelServingTestSupport;
import JourneyCategories;

@testable import ModelServing;

/**
 * Hermetic worker journeys over the real dense artifact runtime: a tiny
 * dense Qwen3.5 artifact (real safetensors shard, real E1 validation, real
 * tiny tokenizer) loads through the production artifact runtime builder
 * and serves chat generations across the real worker pipes — the dense
 * end-to-end leg of #983, proven at the worker boundary with the Romeo and
 * Juliet fixture.
 *
 * The journeys assert the worker's wire contract (swap payload, prefill
 * progress, preparation barrier, first decode, ordered outputs,
 * finalization, completion accounting, cancellation, failure isolation)
 * against structural facts derived from the fixture config — never
 * golden-master token streams. The suite is serialized so the MLX
 * journeys never overlap (the repository's one-model-at-a-time rule).
 */
@Suite(.serialized, .tags(.hermeticMlxJourney))
final class ArtifactRuntimeWorkerJourneyTests {

    private static let ROMEO_AND_JULIET_PROMPT: String =
        "Two households, both alike in dignity: which play opens with this line?";

    init() {
        signal(SIGPIPE, SIG_IGN);
        MLXMetallibLocator.overrideMetallibPathIfNecessary();
    }

    /**
     * The dense artifact serves one full generation across the worker
     * seam: the swap advertises the config-derived capabilities, prefill
     * progress covers the real tokenized prompt exactly, the single-layer
     * preparation barrier fires, every token streams as an ordered output
     * batch, and the bounded budget completes the request with exact
     * accounting.
     */
    @Test(.timeLimit(.minutes(2)))
    func should_serve_the_dense_artifact_runtime_across_the_worker_seam() throws {
        let (modelDirectoryUrl, _): (
            URL, TinyDenseArtifactFixture.SynthesizedLayout
        ) = try TinyDenseArtifactFixture.writeModelDirectory(includeTokenizerFiles: true);
        defer { try? FileManager.default.removeItem(at: modelDirectoryUrl); }

        let harness: WorkerHarness = try WorkerHarness.start(factory: ArtifactRuntimeFactory());
        defer { harness.finish(); }
        _ = try harness.expectBootstrappedLifecycle();

        try harness.sendCommand(.swapModel(
            modelDirectory: modelDirectoryUrl.path,
            modelConfiguration: ArtifactRuntimeWorkerJourneyTests.autoregressiveModelConfiguration()));

        let swapEvent: WorkerEvent = try harness.expectEvent();
        guard case let .modelSwapped(modelId, capabilities, expertMemoryMode, minimumCeiling) = swapEvent else {
            Issue.record("expected a model swap, got \(swapEvent)");
            return;
        }
        #expect(modelId == "qwen3.5");
        #expect(capabilities == WorkerModelCapabilities(
            chat: ChatModelCapabilities(
                supportsReasoning: true,
                supportsToolCalls: true,
                hasVision: false,
                maxInputTokens: 3072,
                maxOutputTokens: 1024,
                contextWindow: 4096),
            imageGeneration: nil,
            embeddings: nil));
        #expect(expertMemoryMode == nil);
        #expect(minimumCeiling == 1);

        let policyEvent: WorkerEvent = try harness.expectEvent();
        guard case .runtimeFeatureConfigurationApplied = policyEvent else {
            Issue.record("expected the runtime policy, got \(policyEvent)");
            return;
        }
        let loadedSampleEvent: WorkerEvent = try harness.expectEvent();
        guard case .mlxMemorySample = loadedSampleEvent else {
            Issue.record("expected the loaded memory sample, got \(loadedSampleEvent)");
            return;
        }

        // The zero thinking budget pins the journey to Qwen3.5's
        // visible-answer mode: the channel otherwise opens inside the
        // thinking markers and a short budget-bound stream stays reasoning.
        try harness.sendCommand(.generate(ArtifactRuntimeWorkerJourneyTests.chatGenerationCommand(
            requestId: 11, maximumOutputTokens: 6, thinkingBudget: 0)));

        var requiredPromptProcessingTokenCount: UInt32 = 0;
        var furthestProcessedTokenCount: UInt32 = 0;
        var preparationTotalLayerCount: UInt32? = nil;
        var firstDecodeSeen: Bool = false;
        var outputFrameCount: Int = 0;
        var lastOutputSequenceNumber: UInt16? = nil;
        var lastOutputGeneratedTokenCount: UInt16 = 0;
        var joinedOutputText: String = "";
        var finalizationSeen: Bool = false;
        var completionEvent: WorkerEvent? = nil;
        for _ in 0..<32 {
            let workerEvent: WorkerEvent = try harness.expectEvent();
            switch workerEvent {
            case let .prefillProgress(_, _, processedTokens, totalTokens, _, _, _, _, _):
                requiredPromptProcessingTokenCount = totalTokens;
                furthestProcessedTokenCount = max(furthestProcessedTokenCount, processedTokens);
            case let .generationPreparationStarted(_, totalLayerCount, _, _, _):
                preparationTotalLayerCount = totalLayerCount;
            case .firstDecodeCompleted:
                firstDecodeSeen = true;
            case let .output(_, sequenceNumber, generatedTokenCount, frameOutputs, _, _):
                if let previousSequenceNumber: UInt16 = lastOutputSequenceNumber {
                    #expect(sequenceNumber > previousSequenceNumber);
                }
                lastOutputSequenceNumber = sequenceNumber;
                lastOutputGeneratedTokenCount = generatedTokenCount;
                outputFrameCount += 1;
                for frameOutput: ChatGenerationOutput in frameOutputs {
                    if case let .text(textPiece) = frameOutput {
                        joinedOutputText += textPiece;
                    }
                }
            case .generationFinalized:
                finalizationSeen = true;
            case .completed:
                completionEvent = workerEvent;
            default:
                continue;
            }
            if completionEvent != nil {
                break;
            }
        }

        guard case let .completed(
            _, completedPromptTokenCount, completedGeneratedTokenCount,
            completedReasoningTokenCount, _, _, completionReason) = completionEvent else {
            Issue.record("expected a completed generation, got \(String(describing: completionEvent))");
            return;
        }
        #expect(requiredPromptProcessingTokenCount > 0);
        #expect(furthestProcessedTokenCount == requiredPromptProcessingTokenCount);
        #expect(completedPromptTokenCount == requiredPromptProcessingTokenCount);
        #expect(preparationTotalLayerCount == 1);
        #expect(firstDecodeSeen);
        #expect(finalizationSeen);
        #expect(outputFrameCount >= 1);
        #expect(lastOutputGeneratedTokenCount == 6);
        #expect(joinedOutputText.isEmpty == false);
        #expect(completedGeneratedTokenCount == 6);
        #expect(completedReasoningTokenCount == 0);
        #expect(completionReason == .maximumOutputTokens);
    }

    /**
     * A cancel command issued mid-generation stops the dense artifact
     * decode, finalizes the request as cancelled, and leaves the worker
     * answering later commands.
     */
    @Test(.timeLimit(.minutes(2)))
    func should_cancel_an_active_artifact_generation_and_stay_responsive() throws {
        let (modelDirectoryUrl, _): (
            URL, TinyDenseArtifactFixture.SynthesizedLayout
        ) = try TinyDenseArtifactFixture.writeModelDirectory(includeTokenizerFiles: true);
        defer { try? FileManager.default.removeItem(at: modelDirectoryUrl); }

        let harness: WorkerHarness = try WorkerHarness.start(factory: ArtifactRuntimeFactory());
        defer { harness.finish(); }
        _ = try harness.expectBootstrappedLifecycle();
        try harness.swapModelIn(
            modelDirectory: modelDirectoryUrl.path,
            modelConfiguration: ArtifactRuntimeWorkerJourneyTests.autoregressiveModelConfiguration());

        try harness.sendCommand(.generate(ArtifactRuntimeWorkerJourneyTests.chatGenerationCommand(
            requestId: 21, maximumOutputTokens: 384)));

        var seenFirstOutput: Bool = false;
        var cancelledCompletion: Bool = false;
        for _ in 0..<1024 {
            let workerEvent: WorkerEvent = try harness.expectEvent();
            switch workerEvent {
            case .output:
                if seenFirstOutput == false {
                    seenFirstOutput = true;
                    try harness.sendCommand(.cancel(requestId: RequestId(rawRequestId: 21)));
                }
            case let .completed(_, _, _, _, _, _, completionReason):
                #expect(completionReason == .cancelled);
                cancelledCompletion = true;
            default:
                continue;
            }
            if cancelledCompletion {
                break;
            }
        }
        #expect(seenFirstOutput);
        #expect(cancelledCompletion);

        try harness.sendCommand(.sampleMlxMemory);
        let pollAnswer: WorkerEvent = try harness.expectEvent();
        guard case .mlxMemorySample = pollAnswer else {
            Issue.record("expected a memory sample after cancellation, got \(pollAnswer)");
            return;
        }
    }

    /**
     * A directory whose shard file disappeared fails the swap closed,
     * naming the absent shard; the idle worker rejects chat generation,
     * and a good artifact swap afterwards restores serving — one failed
     * load never wedges the worker.
     */
    @Test(.timeLimit(.minutes(2)))
    func should_fail_a_swap_missing_its_shard_and_recover_with_a_good_artifact() throws {
        let (brokenDirectoryUrl, _): (
            URL, TinyDenseArtifactFixture.SynthesizedLayout
        ) = try TinyDenseArtifactFixture.writeModelDirectory(includeTokenizerFiles: true);
        defer { try? FileManager.default.removeItem(at: brokenDirectoryUrl); }
        try FileManager.default.removeItem(
            at: brokenDirectoryUrl.appendingPathComponent(TinyDenseArtifactFixture.SHARD_FILE_NAME));

        let harness: WorkerHarness = try WorkerHarness.start(factory: ArtifactRuntimeFactory());
        defer { harness.finish(); }
        _ = try harness.expectBootstrappedLifecycle();

        try harness.sendCommand(.swapModel(
            modelDirectory: brokenDirectoryUrl.path,
            modelConfiguration: ArtifactRuntimeWorkerJourneyTests.autoregressiveModelConfiguration()));
        let swapFailure: WorkerEvent = try harness.expectEvent();
        guard case let .modelSwapFailed(remainsReady, modelLoadFailureReason) = swapFailure else {
            Issue.record("expected a swap failure, got \(swapFailure)");
            return;
        }
        #expect(remainsReady == false);
        #expect(modelLoadFailureReason.contains(TinyDenseArtifactFixture.SHARD_FILE_NAME) == true);

        try harness.sendCommand(.generate(ArtifactRuntimeWorkerJourneyTests.chatGenerationCommand(
            requestId: 31, maximumOutputTokens: 4)));
        let idleRejection: WorkerEvent = try harness.expectEvent();
        #expect(idleRejection == .failed(
            requestId: RequestId(rawRequestId: 31),
            reason: .invalidRequest(reason: "the loaded model does not support chat generation")));

        let (goodDirectoryUrl, _): (
            URL, TinyDenseArtifactFixture.SynthesizedLayout
        ) = try TinyDenseArtifactFixture.writeModelDirectory(includeTokenizerFiles: true);
        defer { try? FileManager.default.removeItem(at: goodDirectoryUrl); }
        try harness.swapModelIn(
            modelDirectory: goodDirectoryUrl.path,
            modelConfiguration: ArtifactRuntimeWorkerJourneyTests.autoregressiveModelConfiguration());

        try harness.sendCommand(.generate(ArtifactRuntimeWorkerJourneyTests.chatGenerationCommand(
            requestId: 32, maximumOutputTokens: 2)));
        var recoveredCompletion: Bool = false;
        for _ in 0..<16 {
            let workerEvent: WorkerEvent = try harness.expectEvent();
            if case let .completed(_, _, generatedTokenCount, _, _, _, _) = workerEvent {
                #expect(generatedTokenCount >= 1);
                recoveredCompletion = true;
                break;
            }
        }
        #expect(recoveredCompletion);
    }

    // MARK: - Fixtures

    private static func autoregressiveModelConfiguration() -> WorkerModelConfiguration {
        return WorkerModelConfiguration.autoregressive(WorkerAutoregressiveModelConfiguration(
            modelId: "qwen3.5",
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

    private static func chatGenerationCommand(
        requestId: UInt64,
        maximumOutputTokens: UInt16,
        thinkingBudget: UInt16? = nil
    ) -> ChatGenerationCommand {
        return ChatGenerationCommand(
            requestId: RequestId(rawRequestId: requestId),
            model: "qwen3.5",
            messages: [.user(content: ROMEO_AND_JULIET_PROMPT, images: [])],
            tools: [],
            toolChoice: .auto,
            settings: ChatGenerationSettings(
                maxOutputTokens: maximumOutputTokens,
                temperatureThousandths: nil,
                topPThousandths: nil,
                seed: 7,
                thinkingBudget: thinkingBudget),
            structuredGeneration: nil);
    }
}

/**
 * Thin dispatch mirroring ModelFamilyFactory's Qwen3.5 dense case: the
 * ModelServingTests target cannot import the InferenceWorker executable,
 * so the journey factory calls the same production artifact runtime
 * builder the factory dispatches to.
 */
struct ArtifactRuntimeFactory: ChatModelRuntimeFactory {

    func createChatRuntime(
        modelDirectory: String,
        modelConfiguration: WorkerModelConfiguration
    ) throws -> LoadedChatRuntime {
        return try Qwen35ChatRuntime.buildArtifactRuntime(
            modelDirectory: modelDirectory,
            modelConfiguration: modelConfiguration);
    }
}
