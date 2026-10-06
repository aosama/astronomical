import Foundation;

import Testing;

import IpcProtocol;
import ModelServing;

@testable import ModelServing;

/**
 * Hermetic journeys for the engine-backed worker: the full swap, generation,
 * rejection, cancellation, and memory contracts over real pipes with a
 * scripted journey runtime. CPU only — no MLX model runs here; the dense
 * engine journeys live in Qwen35DenseEngineTests.
 *
 * Event sequences assert the Rust engine_backed_worker contract the
 * supervisor's event pump already consumes (see #1023): prefill progress
 * frames around prompt processing, the preparation barrier, the first-decode
 * measurement, ordered output batches, and finalization before completion.
 */
@Suite(.serialized)
final class EngineBackedWorkerTests {

    static let romeoAndJulietPrompt: String = "What is the play about?";
    static let romeoAndJulietOutputWords: Array<String> = [
        "Two", "households,", "both", "alike", "in", "dignity",
    ];
    static let endOfSequenceTokenId: UInt32 = 9;

    init() {
        signal(SIGPIPE, SIG_IGN);
    }

    // MARK: - Swap and readiness

    @Test(.timeLimit(.minutes(1)))
    func should_swap_to_a_ready_chat_model_and_report_its_runtime_policy() throws {
        let harness: WorkerHarness = try WorkerHarness.start(
            factory: JourneyChatRuntimeFactory(script: JourneyEngineScript()));
        defer { harness.finish(); }
        _ = try harness.expectBootstrappedLifecycle();

        try harness.sendCommand(.swapModel(
            modelDirectory: "/fictional/models/qwen3.5",
            modelConfiguration: Self.autoregressiveModelConfiguration()));

        let swapEvent: WorkerEvent = try harness.expectEvent();
        guard case let .modelSwapped(modelId, capabilities, expertMemoryMode, minimumCeiling) = swapEvent else {
            Issue.record("expected a model swap, got \(swapEvent)");
            return;
        }
        #expect(modelId == JourneyChatProcessor.modelIdentifier);
        #expect(capabilities == JourneyChatProcessor.chatCapabilities());
        #expect(expertMemoryMode == nil);
        #expect(minimumCeiling == 1);

        let policyEvent: WorkerEvent = try harness.expectEvent();
        guard case let .runtimeFeatureConfigurationApplied(appliedPolicy) = policyEvent else {
            Issue.record("expected the runtime policy, got \(policyEvent)");
            return;
        }
        guard let loadedModel = appliedPolicy.loadedModel else {
            Issue.record("the applied policy lost the loaded model");
            return;
        }
        #expect(loadedModel.modelId() == "qwen3.5");

        let loadedSample: WorkerEvent = try harness.expectEvent();
        guard case let .mlxMemorySample(mlxMemorySnapshot, _) = loadedSample else {
            Issue.record("expected the loaded memory sample, got \(loadedSample)");
            return;
        }
        #expect(mlxMemorySnapshot?.source == .modelLoaded);
        #expect(mlxMemorySnapshot?.activeMemoryBytes == 4_096);

        try harness.sendCommand(.sampleMlxMemory);
        let pollSample: WorkerEvent = try harness.expectEvent();
        guard case let .mlxMemorySample(pollSnapshot, _) = pollSample else {
            Issue.record("expected an idle poll sample, got \(pollSample)");
            return;
        }
        #expect(pollSnapshot?.source == .idlePoll);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_contain_a_failing_swap_and_keep_a_previously_loaded_model_ready() throws {
        let harness: WorkerHarness = try WorkerHarness.start(
            factory: JourneyChatRuntimeFactory(script: JourneyEngineScript()));
        defer { harness.finish(); }
        _ = try harness.expectBootstrappedLifecycle();

        // A failing first swap leaves no model ready.
        try harness.sendCommand(.swapModel(
            modelDirectory: "/fictional/models/unloadable",
            modelConfiguration: Self.autoregressiveModelConfiguration()));
        let firstFailure: WorkerEvent = try harness.expectEvent();
        guard case let .modelSwapFailed(firstRemainsReady, firstReason) = firstFailure else {
            Issue.record("expected a swap failure, got \(firstFailure)");
            return;
        }
        #expect(firstRemainsReady == false);
        #expect(firstReason.contains("journey") == true);

        try harness.sendCommand(.generate(Self.chatGenerationCommand(requestId: 21, maximumOutputTokens: 8)));
        let idleRejection: WorkerEvent = try harness.expectEvent();
        #expect(idleRejection == .failed(
            requestId: RequestId(rawRequestId: 21),
            reason: .invalidRequest(reason: "the loaded model does not support chat generation")));

        // A successful swap loads the journey runtime.
        try harness.sendCommand(.swapModel(
            modelDirectory: "/fictional/models/qwen3.5",
            modelConfiguration: Self.autoregressiveModelConfiguration()));
        _ = try harness.expectEvent();
        _ = try harness.expectEvent();
        _ = try harness.expectEvent();

        // A failing second swap keeps the loaded model serving.
        try harness.sendCommand(.swapModel(
            modelDirectory: "/fictional/models/unloadable",
            modelConfiguration: Self.autoregressiveModelConfiguration()));
        let secondFailure: WorkerEvent = try harness.expectEvent();
        guard case let .modelSwapFailed(secondRemainsReady, _) = secondFailure else {
            Issue.record("expected a second swap failure, got \(secondFailure)");
            return;
        }
        #expect(secondRemainsReady == true);
    }

    // MARK: - Generation event sequences

    @Test(.timeLimit(.minutes(1)))
    func should_stream_the_full_generation_sequence_to_end_of_sequence() throws {
        let harness: WorkerHarness = try WorkerHarness.start(
            factory: JourneyChatRuntimeFactory(script: JourneyEngineScript()));
        defer { harness.finish(); }
        _ = try harness.expectBootstrappedLifecycle();
        try harness.swapJourneyModel();

        try harness.sendCommand(.generate(Self.chatGenerationCommand(
            requestId: 31, maximumOutputTokens: 8)));

        let promptTokenCount: UInt32 = UInt32(Self.romeoAndJulietPrompt.utf8.count);
        let initialPrefill: WorkerEvent = try harness.expectEvent();
        #expect(initialPrefill == .prefillProgress(
            requestId: RequestId(rawRequestId: 31),
            promptProcessingPhase: .target,
            processedTokens: 0,
            totalTokens: promptTokenCount,
            elapsedMillis: 0,
            forwardPrefillChunkElapsedMillis: nil,
            completedPrefillChunkTokens: nil,
            mlxMemorySnapshot: nil,
            expertResidency: nil));

        let chunkPrefill: WorkerEvent = try harness.expectEvent();
        guard case let .prefillProgress(_, _, processedTokens, _, _, chunkElapsed, chunkTokens, _, _) = chunkPrefill else {
            Issue.record("expected the chunk prefill progress, got \(chunkPrefill)");
            return;
        }
        #expect(processedTokens == promptTokenCount);
        #expect(chunkElapsed != nil);
        #expect(chunkTokens == promptTokenCount);

        let preparation: WorkerEvent = try harness.expectEvent();
        #expect(preparation == .generationPreparationStarted(
            requestId: RequestId(rawRequestId: 31),
            totalLayerCount: 2,
            residentExpertCount: 0,
            residentExpertPayloadBytes: 0,
            mlxMemorySnapshot: nil));

        let firstDecode: WorkerEvent = try harness.expectEvent();
        #expect(firstDecode == .firstDecodeCompleted(
            requestId: RequestId(rawRequestId: 31), elapsedMillis: 0));

        let firstOutput: WorkerEvent = try harness.expectEvent();
        guard case let .output(_, firstSequence, firstTokenCount, firstOutputs, _, _) = firstOutput else {
            Issue.record("expected the first output batch, got \(firstOutput)");
            return;
        }
        #expect(firstSequence == 0);
        #expect(firstTokenCount == 1);
        #expect(firstOutputs == [.text(text: "Two")]);

        let secondOutput: WorkerEvent = try harness.expectEvent();
        guard case let .output(_, secondSequence, secondTokenCount, secondOutputs, _, _) = secondOutput else {
            Issue.record("expected the second output batch, got \(secondOutput)");
            return;
        }
        #expect(secondSequence == 1);
        #expect(secondTokenCount == 2);
        #expect(secondOutputs == [.text(text: "households,")]);

        let thirdOutput: WorkerEvent = try harness.expectEvent();
        guard case let .output(_, thirdSequence, thirdTokenCount, thirdOutputs, _, _) = thirdOutput else {
            Issue.record("expected the third output batch, got \(thirdOutput)");
            return;
        }
        #expect(thirdSequence == 2);
        #expect(thirdTokenCount == 3);
        #expect(thirdOutputs == [.text(text: "both")]);

        let finalization: WorkerEvent = try harness.expectEvent();
        guard case let .generationFinalized(_, _, finalSnapshot, _) = finalization else {
            Issue.record("expected the generation finalization, got \(finalization)");
            return;
        }
        #expect(finalSnapshot?.source == MlxMemorySnapshotSource.finalized);

        let completion: WorkerEvent = try harness.expectEvent();
        #expect(completion == .completed(
            requestId: RequestId(rawRequestId: 31),
            promptTokenCount: promptTokenCount,
            generatedTokenCount: 3,
            reasoningTokenCount: 0,
            cachedTokenCount: 0,
            persistentPromptCacheDiagnostics: nil,
            reason: .endOfSequence));
    }

    @Test(.timeLimit(.minutes(1)))
    func should_complete_at_the_maximum_output_token_boundary() throws {
        let harness: WorkerHarness = try WorkerHarness.start(
            factory: JourneyChatRuntimeFactory(script: JourneyEngineScript()));
        defer { harness.finish(); }
        _ = try harness.expectBootstrappedLifecycle();
        try harness.swapJourneyModel();

        try harness.sendCommand(.generate(Self.chatGenerationCommand(
            requestId: 41, maximumOutputTokens: 2)));

        // Two tokens stream as outputs, then the bounded budget ends the
        // request without any engine EOS.
        var outputBatches: Int = 0;
        var completionSeen: Bool = false;
        for _ in 0..<12 {
            let workerEvent: WorkerEvent = try harness.expectEvent();
            switch workerEvent {
            case .output:
                outputBatches += 1;
            case let .completed(_, _, generatedTokenCount, _, _, _, reason):
                #expect(outputBatches == 2);
                #expect(generatedTokenCount == 2);
                #expect(reason == .maximumOutputTokens);
                completionSeen = true;
            default:
                continue;
            }
            if completionSeen {
                break;
            }
        }
        #expect(completionSeen);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_cancel_an_active_generation_and_finalize_it() throws {
        let harness: WorkerHarness = try WorkerHarness.start(factory: JourneyChatRuntimeFactory(
            script: JourneyEngineScript(decodeDelayMilliseconds: 2)));
        defer { harness.finish(); }
        _ = try harness.expectBootstrappedLifecycle();
        try harness.swapJourneyModel();

        try harness.sendCommand(.generate(Self.chatGenerationCommand(
            requestId: 51, maximumOutputTokens: 64)));

        var seenFirstOutput: Bool = false;
        var cancelledCompletion: Bool = false;
        for _ in 0..<80 {
            let workerEvent: WorkerEvent = try harness.expectEvent();
            switch workerEvent {
            case .output:
                if seenFirstOutput == false {
                    seenFirstOutput = true;
                    try harness.sendCommand(.cancel(requestId: RequestId(rawRequestId: 51)));
                }
            case let .completed(_, _, _, _, _, _, reason):
                #expect(reason == .cancelled);
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
    }

    // MARK: - Containment and failure isolation

    @Test(.timeLimit(.minutes(1)))
    func should_reject_a_second_generation_while_the_engine_is_busy() throws {
        let harness: WorkerHarness = try WorkerHarness.start(factory: JourneyChatRuntimeFactory(
            script: JourneyEngineScript(
                decodeDelayMilliseconds: 2, emitEndOfSequence: false)));
        defer { harness.finish(); }
        _ = try harness.expectBootstrappedLifecycle();
        try harness.swapJourneyModel();

        try harness.sendCommand(.generate(Self.chatGenerationCommand(
            requestId: 61, maximumOutputTokens: 64)));

        var busyRejection: Bool = false;
        var outputCount: Int = 0;
        var cancelledCompletion: Bool = false;
        for _ in 0..<80 {
            let workerEvent: WorkerEvent = try harness.expectEvent();
            switch workerEvent {
            case .output:
                outputCount += 1;
                if busyRejection == false {
                    try harness.sendCommand(.generate(Self.chatGenerationCommand(
                        requestId: 62, maximumOutputTokens: 4)));
                    busyRejection = true;
                } else if outputCount == 3 {
                    try harness.sendCommand(.cancel(requestId: RequestId(rawRequestId: 61)));
                }
            case .failed(_, let reason):
                #expect(reason == .engineBusy);
            case let .completed(requestId: matchedRequestId, _, _, _, _, _, reason):
                #expect(matchedRequestId == RequestId(rawRequestId: 61));
                #expect(reason == .cancelled);
                cancelledCompletion = true;
            default:
                continue;
            }
            if cancelledCompletion {
                break;
            }
        }
        #expect(busyRejection);
        #expect(cancelledCompletion);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_fail_a_fatal_engine_error_and_exit_the_worker_after_the_wire_failure() throws {
        let harness: WorkerHarness = try WorkerHarness.start(
            factory: JourneyChatRuntimeFactory(script: JourneyEngineScript(fatalOnFirstDecode: true)));
        defer { harness.finish(); }
        _ = try harness.expectBootstrappedLifecycle();
        try harness.swapJourneyModel();

        try harness.sendCommand(.generate(Self.chatGenerationCommand(
            requestId: 71, maximumOutputTokens: 8)));

        var fatalFailure: Bool = false;
        for _ in 0..<8 {
            let workerEvent: WorkerEvent = try harness.expectEvent();
            if case let .failed(_, reason) = workerEvent {
                guard case let .fatalExecution(fatalReason) = reason else {
                    Issue.record("expected a fatal execution reason, got \(reason)");
                    return;
                }
                #expect(fatalReason.contains("journey fatal") == true);
                fatalFailure = true;
                break;
            }
        }
        #expect(fatalFailure);
        let loopFailure: (any Error)? = harness.joinWithin(seconds: 10);
        #expect(loopFailure != nil);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_reject_malformed_model_output_and_keep_the_worker_responsive() throws {
        let harness: WorkerHarness = try WorkerHarness.start(
            factory: JourneyChatRuntimeFactory(script: JourneyEngineScript(malformedOutput: true)));
        defer { harness.finish(); }
        _ = try harness.expectBootstrappedLifecycle();
        try harness.swapJourneyModel();

        try harness.sendCommand(.generate(Self.chatGenerationCommand(
            requestId: 81, maximumOutputTokens: 8)));

        var malformedFailure: Bool = false;
        for _ in 0..<8 {
            let workerEvent: WorkerEvent = try harness.expectEvent();
            if case .failed(_, .malformedModelOutput) = workerEvent {
                malformedFailure = true;
                break;
            }
        }
        #expect(malformedFailure);

        try harness.sendCommand(.sampleMlxMemory);
        let pollAnswer: WorkerEvent = try harness.expectEvent();
        guard case .mlxMemorySample = pollAnswer else {
            Issue.record("expected a memory sample after the malformed failure, got \(pollAnswer)");
            return;
        }
    }

    // MARK: - Memory limits under a loaded model

    @Test(.timeLimit(.minutes(1)))
    func should_forward_memory_limits_to_a_loaded_engine_and_reject_outside_the_machine() throws {
        let harness: WorkerHarness = try WorkerHarness.start(
            factory: JourneyChatRuntimeFactory(script: JourneyEngineScript()),
            machineMlxMemoryCeilingBytes: 1_000_000);
        defer { harness.finish(); }
        _ = try harness.expectBootstrappedLifecycle();
        try harness.swapJourneyModel();

        try harness.sendCommand(.updateMlxMemoryLimit(
            effectiveMlxMemoryCeilingBytes: 2_000_000,
            configurationGeneration: "gen-beyond"));
        let beyondReject: WorkerEvent = try harness.expectEvent();
        #expect(beyondReject == .mlxMemoryLimitRejected(
            requestedMlxMemoryCeilingBytes: 2_000_000,
            minimumMlxMemoryCeilingBytes: 1,
            machineMlxMemoryCeilingBytes: 1_000_000,
            reason: "requested memory ceiling is outside the worker machine limit"));

        try harness.sendCommand(.updateMlxMemoryLimit(
            effectiveMlxMemoryCeilingBytes: 500_000,
            configurationGeneration: "gen-accepted"));
        let acceptedChange: WorkerEvent = try harness.expectEvent();
        guard case let .mlxMemoryLimitChanged(effectiveCeiling, minimumCeiling, _, snapshot, _) = acceptedChange else {
            Issue.record("expected an accepted memory change, got \(acceptedChange)");
            return;
        }
        #expect(effectiveCeiling == 500_000);
        #expect(minimumCeiling == 1);
        #expect(snapshot?.source == .memoryLimitAdjusted);
        #expect(snapshot?.activeMemoryBytes == 4_096);
    }

    // MARK: - Harness

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
        maximumOutputTokens: UInt16
    ) -> ChatGenerationCommand {
        return ChatGenerationCommand(
            requestId: RequestId(rawRequestId: requestId),
            model: "qwen3.5",
            messages: [.user(content: Self.romeoAndJulietPrompt, images: [])],
            tools: [],
            toolChoice: .auto,
            settings: ChatGenerationSettings(
                maxOutputTokens: maximumOutputTokens,
                temperatureThousandths: nil,
                topPThousandths: nil,
                seed: nil,
                thinkingBudget: nil),
            qwenThinkingChannelSeed: nil,
            structuredGeneration: nil);
    }
}
