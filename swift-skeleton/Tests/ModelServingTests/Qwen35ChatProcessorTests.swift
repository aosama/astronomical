import Foundation;

import Testing;

import MLXLMCommon;
import Tokenizers;

import IpcProtocol;
import ModelServing;

@testable import ModelServing;

/**
 * Hermetic journeys for the real Qwen3.5 chat processor: chat-template
 * preparation over a synthesized on-disk tokenizer, the tagged reasoning
 * channel routed through the upstream collector, end-of-sequence ids from
 * the artifact config, and the fail-closed rejection surface for slices
 * that land later. The final journey pairs the processor and the dense
 * engine through the real model directory builder inside the worker loop.
 */
@Suite(.serialized)
final class Qwen35ChatProcessorTests {

    init() {
        signal(SIGPIPE, SIG_IGN);
        MLXMetallibLocator.overrideMetallibPathIfNecessary();
    }

    // MARK: - Template preparation

    @Test(.timeLimit(.minutes(1)))
    func should_render_the_chat_template_into_a_tokenized_prompt() throws {
        let processor: Qwen35ChatProcessor = try Self.makeProcessor();
        let preparedGeneration: any ActiveChatGeneration = try processor.prepareChatGeneration(
            Self.chatCommand(requestId: 111));

        #expect(preparedGeneration.promptTokenCount > 0);
        let preparedRequest: Qwen35PreparedInferenceRequest = try #require(
            preparedGeneration.inferenceRequest as? Qwen35PreparedInferenceRequest);
        #expect(preparedRequest.promptTokenIds.count == preparedGeneration.promptTokenCount);
        // Every prompt id stays inside the tiny vocabulary and the template
        // renders the assistant generation prompt at the end.
        #expect(preparedRequest.promptTokenIds.allSatisfy { $0 < 512 });
        // The rendered prompt decodes back to the conversation with the
        // assistant generation prompt present; the minimal BPE decoder joins
        // without the stripped whitespace, so words are asserted directly.
        let decodedPrompt: String = Self.decodedPrompt(
            promptTokenIds: preparedRequest.promptTokenIds);
        #expect(decodedPrompt.contains("What"));
        #expect(decodedPrompt.contains("play"));
        #expect(decodedPrompt.contains("assistant"));
    }

    @Test(.timeLimit(.minutes(1)))
    func should_reject_the_seed_budget_and_structured_paths_with_bounded_reasons() throws {
        let processor: Qwen35ChatProcessor = try Self.makeProcessor();

        var seededCommand: ChatGenerationCommand = Self.chatCommand(requestId: 121);
        seededCommand = ChatGenerationCommand(
            requestId: seededCommand.requestId, model: seededCommand.model,
            messages: seededCommand.messages, tools: seededCommand.tools,
            toolChoice: seededCommand.toolChoice, settings: seededCommand.settings,
            qwenThinkingChannelSeed: "remember the balcony",
            structuredGeneration: nil);
        #expect(throws: ChatPreparationRejection.self) {
            _ = try processor.prepareChatGeneration(seededCommand);
        }

        var budgetedCommand: ChatGenerationCommand = Self.chatCommand(requestId: 122);
        budgetedCommand = ChatGenerationCommand(
            requestId: budgetedCommand.requestId, model: budgetedCommand.model,
            messages: budgetedCommand.messages, tools: budgetedCommand.tools,
            toolChoice: budgetedCommand.toolChoice,
            settings: ChatGenerationSettings(
                maxOutputTokens: 16, temperatureThousandths: nil, topPThousandths: nil,
                seed: nil, thinkingBudget: 8),
            qwenThinkingChannelSeed: nil, structuredGeneration: nil);
        #expect(throws: ChatPreparationRejection.self) {
            _ = try processor.prepareChatGeneration(budgetedCommand);
        }
    }

    // MARK: - Reasoning channel routing

    @Test(.timeLimit(.minutes(1)))
    func should_route_reasoning_and_text_channels_from_generated_tokens() throws {
        let processor: Qwen35ChatProcessor = try Self.makeProcessor();
        let activeGeneration: any ActiveChatGeneration = try processor.prepareChatGeneration(
            Self.chatCommand(requestId: 131));

        var reasoningText: String = "";
        var responseText: String = "";
        for generatedTokenId: UInt32 in Self.reasoningAndAnswerTokenIds() {
            let translation: ModelGeneratedTokenTranslation = try activeGeneration
                .translateGeneratedToken(generatedTokenId);
            for output: ChatGenerationOutput in translation.publicOutputs {
                switch output {
                case let .reasoning(text): reasoningText += text;
                case let .text(text): responseText += text;
                case .toolCall:
                    Issue.record("the dense journey emitted no tool calls");
                    return;
                }
            }
        }
        let flushedOutputs: Array<ChatGenerationOutput> = try activeGeneration.finishOutputs();
        for output: ChatGenerationOutput in flushedOutputs {
            if case let .text(text) = output { responseText += text; }
            if case let .reasoning(text) = output { reasoningText += text; }
        }
        #expect(reasoningText.contains("What") || reasoningText.isEmpty == false);
        #expect(responseText.isEmpty == false);
    }

    @Test(.timeLimit(.minutes(1)))
    func should_declare_end_of_sequence_only_on_the_configured_ids() throws {
        let processor: Qwen35ChatProcessor = try Self.makeProcessor();
        let activeGeneration: any ActiveChatGeneration = try processor.prepareChatGeneration(
            Self.chatCommand(requestId: 141));
        #expect(activeGeneration.isEndOfSequenceToken(UInt32(TinyTokenizerFixture.endOfSequenceTokenId)));
        #expect(activeGeneration.isEndOfSequenceToken(UInt32(TinyTokenizerFixture.endOfSequenceTokenId + 1)) == false);
        #expect(activeGeneration.isEndOfSequenceToken(20) == false);
    }

    // MARK: - Runtime pairing from a real model directory

    @Test(.timeLimit(.minutes(1)))
    func should_pair_the_tokenizer_and_engine_from_a_model_directory() throws {
        let runtime: LoadedChatRuntime = try Self.makeRuntimeFromDirectory();
        let loadResult: EngineLoadResult = try runtime.engine.load();
        #expect(loadResult.minimumMlxMemoryCeilingBytes == 1);

        let readyEvent: WorkerEvent = runtime.processor.readyEvent();
        guard case let .ready(modelId, capabilities) = readyEvent else {
            Issue.record("expected a ready event, got \(readyEvent)");
            return;
        }
        #expect(modelId == "qwen3.5");
        let chatCapabilities: ChatModelCapabilities? = capabilities.chat;
        #expect(chatCapabilities?.contextWindow == 4096);
        #expect(chatCapabilities?.supportsReasoning == true);
        #expect(chatCapabilities?.supportsToolCalls == false);
    }

    @Test(.timeLimit(.minutes(2)))
    func should_serve_real_tokenized_text_through_the_worker_loop() throws {
        let runtime: LoadedChatRuntime = try Self.makeRuntimeFromDirectory();
        let harness: WorkerHarness = try WorkerHarness.start(
            factory: DirectoryChatRuntimeFactory(runtime: runtime));
        defer { harness.finish(); }
        _ = try harness.expectBootstrappedLifecycle();
        try harness.swapModelIn(
            modelDirectory: "/synthesized/models/qwen3.5",
            modelConfiguration: WorkerModelConfiguration.autoregressive(
                Self.autoregressiveConfiguration()));

        try harness.sendCommand(.generate(Self.chatCommand(requestId: 151)));

        // The in-memory weights are the pinned random initialization, so the
        // sampled ids decode through the real tokenizer to whatever the
        // vocabulary covers; the journey asserts pipeline integrity (ordered
        // batches, budget-bound counters, completion), and the channel
        // routing with real vocabulary ids is proven in the channel journey.
        var nextSequenceNumber: UInt16 = 0;
        var emittedOutputCount: Int = 0;
        var generatedTokenCount: UInt16 = 0;
        var completed: Bool = false;
        for _ in 0..<40 {
            let workerEvent: WorkerEvent = try harness.expectEvent();
            switch workerEvent {
            case let .output(_, sequenceNumber, reportedGeneratedTokenCount, outputs, _, _):
                #expect(sequenceNumber == nextSequenceNumber);
                #expect(reportedGeneratedTokenCount >= sequenceNumber);
                nextSequenceNumber += UInt16(outputs.count);
                emittedOutputCount += outputs.count;
                generatedTokenCount = reportedGeneratedTokenCount;
            case let .completed(_, promptTokenCount, reportedGeneratedTokenCount, _, _, _, completionReason):
                #expect(promptTokenCount > 0);
                #expect(generatedTokenCount <= reportedGeneratedTokenCount);
                #expect(reportedGeneratedTokenCount <= 16);
                // Either boundary ends the journey: the config's end-of-
                // sequence id (sampled from random weights) or the budget.
                #expect(completionReason == .endOfSequence || completionReason == .maximumOutputTokens);
                if completionReason == .maximumOutputTokens {
                    #expect(reportedGeneratedTokenCount == 16);
                }
                completed = true;
            case .failed:
                Issue.record("the tokenized generation journey failed: \(workerEvent)");
                return;
            default:
                continue;
            }
            if completed {
                break;
            }
        }
        #expect(completed);
        #expect(emittedOutputCount <= Int(generatedTokenCount));
    }

    // MARK: - Fixtures

    /// Decodes a prepared prompt back to text for round-trip assertions
    /// through a second tokenizer instance loaded from the same synthesized
    /// directory shape.
    private static func decodedPrompt(
        promptTokenIds: Array<UInt32>
    ) -> String {
        let modelDirectoryUrl: URL
        do {
            modelDirectoryUrl = try Self.synthesizedModelDirectory();
        } catch {
            Issue.record("the tokenizer round-trip fixture could not be written");
            return "";
        }
        let roundTripOutcome: Result<String, Error> = Self.blockingRoundTripDecode(
            modelDirectoryUrl: modelDirectoryUrl, promptTokenIds: promptTokenIds);
        do {
            return try roundTripOutcome.get();
        } catch {
            Issue.record("the round-trip tokenizer failed: \(error)");
            return "";
        }
    }

    private static func makeProcessor() throws -> Qwen35ChatProcessor {
        let runtime: LoadedChatRuntime = try Self.makeRuntimeFromDirectory();
        guard let processor = runtime.processor as? Qwen35ChatProcessor else {
            Issue.record("expected the dense chat processor");
            throw WorkerHarness.harnessFailure("unexpected processor type");
        }
        return processor;
    }

    /// Synthesizes the model directory: the tiny dense config plus the tiny
    /// BPE tokenizer files. No shard files by design; the in-memory path
    /// constructs the engine from the config document.
    private static func synthesizedModelDirectory() throws -> URL {
        let modelDirectoryUrl: URL = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen35-chat-journey-\(UUID().uuidString)");
        try FileManager.default.createDirectory(at: modelDirectoryUrl, withIntermediateDirectories: true);
        try Data(Self.tinyDenseConfigJson.utf8).write(
            to: modelDirectoryUrl.appendingPathComponent("config.json"));
        try TinyTokenizerFixture.writeFiles(modelDirectoryUrl: modelDirectoryUrl);
        return modelDirectoryUrl;
    }

    /// Builds the matched runtime from a synthesized model directory.
    private static func makeRuntimeFromDirectory() throws -> LoadedChatRuntime {
        return try Qwen35ChatRuntime.buildInMemoryRuntime(
            modelDirectory: Self.synthesizedModelDirectory().path,
            modelConfiguration: WorkerModelConfiguration.autoregressive(
                Self.autoregressiveConfiguration()));
    }

    /// `What is the play about?` tokenized through the fixture vocabulary,
    /// then the tagged reasoning open, three reasoning words, the close, and
    /// three answer words — ids straight from the fixture vocabulary.
    private static func reasoningAndAnswerTokenIds() -> Array<UInt32> {
        return [
            UInt32(TinyTokenizerFixture.vocabulary()["What"]!),
            UInt32(TinyTokenizerFixture.vocabulary()["is"]!),
            UInt32(TinyTokenizerFixture.vocabulary()["the"]!),
            UInt32(TinyTokenizerFixture.vocabulary()["play"]!),
            UInt32(TinyTokenizerFixture.vocabulary()["about"]!),
            UInt32(TinyTokenizerFixture.thinkOpenTokenId),
            UInt32(TinyTokenizerFixture.vocabulary()["Two"]!),
            UInt32(TinyTokenizerFixture.vocabulary()["households"]!),
            UInt32(TinyTokenizerFixture.vocabulary()["both"]!),
            UInt32(TinyTokenizerFixture.thinkCloseTokenId),
            UInt32(TinyTokenizerFixture.vocabulary()["alike"]!),
            UInt32(TinyTokenizerFixture.vocabulary()["in"]!),
            UInt32(TinyTokenizerFixture.vocabulary()["dignity"]!),
        ];
    }

    private static func chatCommand(requestId: UInt64) -> ChatGenerationCommand {
        return ChatGenerationCommand(
            requestId: RequestId(rawRequestId: requestId),
            model: "qwen3.5",
            messages: [
                .system(content: "answer plainly"),
                .user(content: "What is the play about?", images: []),
            ],
            tools: [],
            toolChoice: .auto,
            settings: ChatGenerationSettings(
                maxOutputTokens: 16,
                temperatureThousandths: nil,
                topPThousandths: nil,
                seed: 7,
                thinkingBudget: nil),
            qwenThinkingChannelSeed: nil,
            structuredGeneration: nil);
    }

    private static func autoregressiveConfiguration() -> WorkerAutoregressiveModelConfiguration {
        return WorkerAutoregressiveModelConfiguration(
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
                experimentalFusedMoeDecodeEnabled: false));
    }

    /**
     * The same tiny dense config as the dense engine journeys, with the
     * end-of-sequence ids matching the fixture tokenizer's control marker.
     */
    private static let tinyDenseConfigJson: String = """
        {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "dtype": "bfloat16",
            "eos_token_id": [3],
            "tie_word_embeddings": false,
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": 64,
                "attention_bias": false,
                "hidden_act": "silu",
                "rms_norm_eps": 1e-6,
                "layer_types": ["full_attention", "full_attention"],
                "vocab_size": 512,
                "full_attention_interval": 2,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 32,
                "linear_conv_kernel_dim": 4,
                "max_position_embeddings": 4096,
                "rope_parameters": {
                    "type": "default",
                    "mrope_interleaved": true,
                    "mrope_section": [11, 11, 10],
                    "rope_theta": 100000.0,
                    "partial_rotary_factor": 1.0
                }
            }
        }
        """;
}

/// Adapter factory handing the pre-built directory runtime to the worker
/// harness; the directory itself was synthesized before the loop started.
struct DirectoryChatRuntimeFactory: ChatModelRuntimeFactory {

    let runtime: LoadedChatRuntime;

    func createChatRuntime(
        modelDirectory: String,
        modelConfiguration: WorkerModelConfiguration
    ) throws -> LoadedChatRuntime {
        return self.runtime;
    }
}

extension Qwen35ChatProcessorTests {

    /// Blocks the sync journey on the async tokenizer load for the decode
    /// round trip, mirroring the runtime's own factory boundary.
    fileprivate static func blockingRoundTripDecode(
        modelDirectoryUrl: URL,
        promptTokenIds: Array<UInt32>
    ) -> Result<String, Error> {
        final class DecodeOutcome: @unchecked Sendable {
            var result: Result<String, Error>?
        }
        let decodeOutcome: DecodeOutcome = DecodeOutcome();
        let decodeSemaphore: DispatchSemaphore = DispatchSemaphore(value: 0);
        Task.detached(priority: .userInitiated) {
            do {
                let roundTripTokenizer = try await Tokenizers.AutoTokenizer.from(
                    modelFolder: modelDirectoryUrl);
                decodeOutcome.result = .success(roundTripTokenizer.decode(
                    tokens: promptTokenIds.map(Int.init), skipSpecialTokens: false));
            } catch {
                decodeOutcome.result = .failure(error);
            }
            decodeSemaphore.signal();
        }
        decodeSemaphore.wait();
        return decodeOutcome.result!;
    }
}
