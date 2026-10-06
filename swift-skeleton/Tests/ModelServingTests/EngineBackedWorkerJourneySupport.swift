import Foundation;

import IpcProtocol;
import ModelServing;

/// Scripted journey runtime for the engine-backed worker journeys: a chat
/// processor and engine pair whose outputs come from the Romeo and Juliet
/// fixture and whose failure modes the journey selects up front.
struct JourneyEngineScript: Sendable {
    var fatalOnFirstDecode: Bool = false;
    var malformedOutput: Bool = false;
    /// Per-step pause so cancellation commands interleave deterministically
    /// between decode steps, exactly as they do against real model decode
    /// latency.
    var decodeDelayMilliseconds: UInt32? = nil;
    /// Keeps the end-of-sequence marker out of the token stream so a journey
    /// can exercise the budget and cancellation boundaries instead.
    var emitEndOfSequence: Bool = true;
}

/// Deterministic chat processor: the prompt size is the fixture text's byte
/// count, each generated token decodes to one fixture word, and token 9 is
/// the end-of-sequence marker.
final class JourneyChatProcessor: ModelGenerationProcessor {

    static let modelIdentifier: String = "journey-chat";

    private let script: JourneyEngineScript;

    init(script: JourneyEngineScript) {
        self.script = script;
    }

    static func chatCapabilities() -> WorkerModelCapabilities {
        return WorkerModelCapabilities(
            chat: ChatModelCapabilities(
                supportsReasoning: false,
                supportsToolCalls: false,
                hasVision: false,
                maxInputTokens: 3072,
                maxOutputTokens: 1024,
                contextWindow: 4096),
            imageGeneration: nil,
            embeddings: nil);
    }

    func readyEvent() -> WorkerEvent {
        return .ready(
            modelId: JourneyChatProcessor.modelIdentifier,
            capabilities: JourneyChatProcessor.chatCapabilities());
    }

    func prepareChatGeneration(
        _ chatGenerationCommand: ChatGenerationCommand
    ) throws -> any ActiveChatGeneration {
        let promptByteCount: Int = chatGenerationCommand.messages.reduce(0) {
            promptByteCount, chatMessage in
            return promptByteCount + chatMessage.plainTextContent().utf8.count;
        };
        return JourneyActiveGeneration(
            script: self.script,
            promptTokenCount: max(1, promptByteCount),
            maximumOutputTokens: chatGenerationCommand.settings.maxOutputTokens);
    }
}

/// Request-local translator over the scripted fixture words.
final class JourneyActiveGeneration: ActiveChatGeneration {

    private let script: JourneyEngineScript;
    private let maximumOutputTokens: UInt16;
    private var producedTokenCount: Int = 0;

    let promptTokenCount: Int;
    let preparedRequest: JourneyPreparedRequest;

    var inferenceRequest: any PreparedInferenceRequest {
        return self.preparedRequest;
    }

    init(script: JourneyEngineScript, promptTokenCount: Int, maximumOutputTokens: UInt16) {
        self.script = script;
        self.promptTokenCount = promptTokenCount;
        self.maximumOutputTokens = maximumOutputTokens;
        self.preparedRequest = JourneyPreparedRequest(promptTokenCount: promptTokenCount);
    }

    func isEndOfSequenceToken(_ generatedTokenId: UInt32) -> Bool {
        return generatedTokenId == EngineBackedWorkerTests.endOfSequenceTokenId;
    }

    func translateGeneratedToken(
        _ generatedTokenId: UInt32
    ) throws -> ModelGeneratedTokenTranslation {
        self.producedTokenCount += 1;
        if self.script.malformedOutput {
            throw ModelGenerationOutputError.malformedOutput;
        }
        let fixtureWords: Array<String> = EngineBackedWorkerTests.romeoAndJulietOutputWords;
        let word: String = fixtureWords[(self.producedTokenCount - 1) % fixtureWords.count];
        return ModelGeneratedTokenTranslation(publicOutputs: [.text(text: word)]);
    }

    func finishOutputs() throws -> Array<ChatGenerationOutput> {
        return [];
    }
}

/// The prepared fixture prompt the journey engine consumes.
final class JourneyPreparedRequest: PreparedInferenceRequest {

    let promptTokenCount: Int;

    init(promptTokenCount: Int) {
        self.promptTokenCount = promptTokenCount;
    }
}

/// Scripted engine: one prefill chunk covering the whole prompt, a
/// preparation boundary, then fixture token IDs; token 9 ends the stream
/// only when the budget allows it first.
final class JourneyChatEngine: InferenceEngine {

    private let script: JourneyEngineScript;
    private var activeRequestActive: Bool = false;
    private var decodeStepCount: Int = 0;
    private var activePromptTokenCount: Int = 0;

    init(script: JourneyEngineScript) {
        self.script = script;
    }

    func load() throws -> EngineLoadResult {
        return EngineLoadResult(minimumMlxMemoryCeilingBytes: 1, expertMemoryMode: nil);
    }

    func startGeneration(
        _ inferenceRequest: any PreparedInferenceRequest
    ) throws -> EngineGenerationStart {
        if self.activeRequestActive {
            throw InferenceEngineError.engineBusy;
        }
        self.activeRequestActive = true;
        self.activePromptTokenCount = inferenceRequest.promptTokenCount;
        self.decodeStepCount = 0;
        return EngineGenerationStart(
            cachedTokenCount: 0, restoredPromptPrefixTokenCount: 0,
            expertMemoryMode: nil, promptProcessingPhase: .target);
    }

    func decodeNextToken(requestId: RequestId) throws -> GeneratedToken {
        self.decodeStepCount += 1;
        if let decodeDelayMilliseconds = self.script.decodeDelayMilliseconds {
            Thread.sleep(forTimeInterval: Double(decodeDelayMilliseconds) / 1000.0);
        }
        if self.decodeStepCount == 1 {
            return .prefillProgress(
                processedTokenCount: UInt32(self.activePromptTokenCount),
                elapsedMillis: 1,
                forwardPrefillChunkElapsedMillis: 1,
                completedPrefillChunkTokens: UInt32(self.activePromptTokenCount),
                mlxMemorySnapshot: nil,
                expertResidencyTelemetry: nil,
                expertMemoryMode: nil,
                promptWorkReuse: WorkerPromptWorkReuse(
                    targetEligibleTokenCount: 0, targetRestoredTokenCount: 0));
        }
        if self.decodeStepCount == 2 {
            return .generationPreparationStarted(
                totalLayerCount: 2,
                residentExpertCount: 0,
                residentExpertPayloadBytes: 0,
                mlxMemorySnapshot: nil);
        }
        if self.script.fatalOnFirstDecode {
            throw InferenceEngineError.fatalExecution(
                reason: "journey fatal decode step \(self.decodeStepCount)");
        }
        // A stable fixture token stream whose third token is the
        // end-of-sequence marker; the processor decides EOS. The marker-free
        // stream serves budget and cancellation journeys.
        let fixtureTokenIds: Array<UInt32> = self.script.emitEndOfSequence
            ? [5, 6, 9, 7, 8, 10, 11, 12] : [5, 6, 7, 8, 10, 11, 12, 13];
        let generatedTokenId: UInt32 = fixtureTokenIds[(self.decodeStepCount - 3)
            % fixtureTokenIds.count];
        return .tokenId(
            generatedTokenId: generatedTokenId,
            isReasoningToken: false,
            expertMemoryMode: nil,
            mlxMemorySnapshot: nil,
            firstDecodeForwardElapsedMillis: self.decodeStepCount == 3 ? 0 : nil,
            generationFinalization: nil);
    }

    func injectInputTokens(requestId: RequestId, inputTokenIds: Array<UInt32>) throws {
        return;
    }

    func cancelGeneration(requestId: RequestId) throws -> GenerationFinalization {
        self.activeRequestActive = false;
        return GenerationFinalization(
            expertMemoryMode: nil,
            mlxMemorySnapshot: WorkerMlxMemorySnapshot(
                source: .finalized,
                activeMemoryBytes: 512,
                allocatorCacheMemoryBytes: 0,
                peakMemoryBytes: 512,
                expertPayloadBytes: 0,
                modelCorePayloadBytes: 512,
                contextStatePayloadBytes: 0,
                memoryCeilingUtilization: nil),
            expertResidencyTelemetry: nil);
    }

    func collectMlxMemorySnapshot() -> WorkerMlxMemorySnapshot? {
        return WorkerMlxMemorySnapshot(
            source: .idlePoll,
            activeMemoryBytes: 4_096,
            allocatorCacheMemoryBytes: 1_024,
            peakMemoryBytes: 8_192,
            expertPayloadBytes: 0,
            modelCorePayloadBytes: 4_096,
            contextStatePayloadBytes: 0,
            memoryCeilingUtilization: nil);
    }

    func applyMlxMemoryLimit(_ requestedMlxMemoryCeilingBytes: UInt64) throws {
        return;
    }
}

/// Factory pairing the journey processor and engine; directories that do
/// not name the loadable journey model fail the swap the way real factories
/// fail a rejected selection.
struct JourneyChatRuntimeFactory: ChatModelRuntimeFactory {

    private let script: JourneyEngineScript;

    init(script: JourneyEngineScript) {
        self.script = script;
    }

    func createChatRuntime(
        modelDirectory: String,
        modelConfiguration: WorkerModelConfiguration
    ) throws -> LoadedChatRuntime {
        if modelDirectory.hasSuffix("/qwen3.5") == false {
            throw InferenceEngineError.modelLoad(
                reason: "journey factory cannot load \(modelDirectory)");
        }
        return LoadedChatRuntime(
            processor: JourneyChatProcessor(script: self.script),
            engine: JourneyChatEngine(script: self.script));
    }
}

extension ChatMessage {

    /// The plain text the journey prompt sizing reads; vision inputs stay
    /// out of the journey surface.
    func plainTextContent() -> String {
        switch self {
        case let .system(content): return content;
        case let .user(content, _): return content;
        case let .assistant(content, _, _): return content ?? "";
        case let .tool(_, content): return content;
        }
    }
}
