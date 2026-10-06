import Foundation;

import Testing;

import MLX;

import IpcProtocol;
import ModelServing;
import ModelServingTestSupport;
import JourneyCategories;

@testable import ModelServing;

/**
 * Hermetic journeys for artifact-weight streaming: the validated shard
 * descriptors of a synthesized tiny dense artifact stream into the upstream
 * dense model, the bound weights are the artifact payload (not the pinned
 * random initialization), descriptor ownership transfers exactly once, and
 * the production runtime builder serves one tokenized chat generation from
 * the streamed weights.
 *
 * Payload bytes are a deterministic non-zero fill, so every forward pass
 * runs on real streamed weights. The suite is serialized so the MLX
 * journeys never overlap (the repository's one-model-at-a-time rule).
 */
@Suite(.serialized, .tags(.hermeticMlxJourney))
final class Qwen35ArtifactWeightLoadingTests {

    private static let ROMEO_AND_JULIET_PROMPT: String = "What is the play about?";

    init() {
        signal(SIGPIPE, SIG_IGN);
        MLXMetallibLocator.overrideMetallibPathIfNecessary();
    }

    /**
     * The validated artifact's shard descriptors stream into the upstream
     * model and serve a real forward pass: the load reports the engine
     * floor, prefill boundaries sum to the prompt, every generated token
     * stays inside the config vocabulary, and the memory snapshot reports
     * the artifact payload as the model core payload.
     */
    @Test(.timeLimit(.minutes(2)))
    func should_stream_validated_artifact_weights_into_a_real_dense_forward_pass() throws {
        let (modelDirectoryUrl, modelLayout): (URL, TinyDenseArtifactFixture.SynthesizedLayout) =
            try TinyDenseArtifactFixture.writeModelDirectory();
        defer { try? FileManager.default.removeItem(at: modelDirectoryUrl); }
        let validatedArtifact: ValidatedQwen35Artifact = try Qwen35ArtifactValidator()
            .validate(
                modelDirectory: modelDirectoryUrl.path,
                maxOutputTokens: TinyDenseArtifactFixture.MAX_OUTPUT_TOKENS);

        let engine: Qwen35DenseEngine = Qwen35DenseEngine();
        try engine.loadValidatedArtifact(validatedArtifact);
        let loadResult: EngineLoadResult = try engine.load();
        #expect(loadResult.minimumMlxMemoryCeilingBytes == 1);
        #expect(loadResult.expertMemoryMode == nil);

        let promptTokenIds: Array<UInt32> = Qwen35ArtifactWeightLoadingTests.fixturePromptTokenIds();
        let requestId: RequestId = RequestId(rawRequestId: 91);
        let generationStart: EngineGenerationStart = try engine.startGeneration(
            Qwen35PreparedInferenceRequest(
                promptTokenIds: promptTokenIds,
                samplingSettings: Qwen35SamplingSettings(
                    chatGenerationSettings: Qwen35ArtifactWeightLoadingTests.settings(seed: 7))));

        #expect(generationStart.cachedTokenCount == 0);
        #expect(generationStart.promptProcessingPhase == .target);

        var processedTokenCount: UInt32 = 0;
        var preparationSeen: Bool = false;
        var generatedTokenIds: Array<UInt32> = [];
        while generatedTokenIds.count < 5 {
            let boundary: GeneratedToken = try engine.decodeNextToken(requestId: requestId);
            switch boundary {
            case let .prefillProgress(chunkProcessed, _, _, _, _, _, _, _):
                processedTokenCount += chunkProcessed;
                #expect(chunkProcessed > 0);
            case .generationPreparationStarted:
                preparationSeen = true;
                #expect(processedTokenCount == UInt32(promptTokenIds.count));
            case let .tokenId(generatedTokenId, false, _, _, _, nil):
                #expect(generatedTokenId < 256);
                generatedTokenIds.append(generatedTokenId);
            default:
                Issue.record("unexpected dense engine boundary \(boundary)");
                return;
            }
        }
        #expect(processedTokenCount == UInt32(promptTokenIds.count));
        #expect(preparationSeen);
        #expect(generatedTokenIds.isEmpty == false);

        let finalization: GenerationFinalization = try engine.cancelGeneration(
            requestId: requestId);
        #expect(finalization.mlxMemorySnapshot != nil);
        let finalSnapshot: WorkerMlxMemorySnapshot = finalization.mlxMemorySnapshot!;
        #expect(finalSnapshot.modelCorePayloadBytes == modelLayout.totalPayloadBytes);
    }

    /**
     * The bound weights are the artifact payload: two engines loaded from
     * the same directory (validated independently) reproduce the identical
     * seeded stream, while a directory whose payload bytes differ produces a
     * different stream — the pinned random initialization would make all
     * three identical.
     */
    @Test(.timeLimit(.minutes(2)))
    func should_bind_the_artifact_payload_rather_than_the_pinned_initialization() throws {
        let (firstDirectoryUrl, _): (URL, TinyDenseArtifactFixture.SynthesizedLayout) =
            try TinyDenseArtifactFixture.writeModelDirectory();
        defer { try? FileManager.default.removeItem(at: firstDirectoryUrl); }
        let (repeatDirectoryUrl, _): (URL, TinyDenseArtifactFixture.SynthesizedLayout) =
            try TinyDenseArtifactFixture.writeModelDirectory();
        defer { try? FileManager.default.removeItem(at: repeatDirectoryUrl); }
        let (alternateDirectoryUrl, _): (URL, TinyDenseArtifactFixture.SynthesizedLayout) =
            try TinyDenseArtifactFixture.writeModelDirectory(
                weightVariantSalt: TinyDenseArtifactFixture.ALTERNATE_WEIGHT_VARIANT_SALT);
        defer { try? FileManager.default.removeItem(at: alternateDirectoryUrl); }

        let promptTokenIds: Array<UInt32> = Qwen35ArtifactWeightLoadingTests.fixturePromptTokenIds();
        let firstTokenIds: Array<UInt32> = try Qwen35ArtifactWeightLoadingTests.collectTokenIds(
            modelDirectoryUrl: firstDirectoryUrl, promptTokenIds: promptTokenIds, seed: 42);
        let repeatTokenIds: Array<UInt32> = try Qwen35ArtifactWeightLoadingTests.collectTokenIds(
            modelDirectoryUrl: repeatDirectoryUrl, promptTokenIds: promptTokenIds, seed: 42);
        let alternateTokenIds: Array<UInt32> = try Qwen35ArtifactWeightLoadingTests.collectTokenIds(
            modelDirectoryUrl: alternateDirectoryUrl, promptTokenIds: promptTokenIds, seed: 42);

        #expect(firstTokenIds == repeatTokenIds);
        #expect(firstTokenIds.isEmpty == false);
        #expect(firstTokenIds != alternateTokenIds);
    }

    /**
     * Descriptor ownership transfers exactly once: a validated artifact
     * whose shard sources were already taken fails the second load closed
     * with a bounded model-load reason.
     */
    @Test(.timeLimit(.minutes(1)))
    func should_fail_a_second_load_of_one_validated_artifact() throws {
        let (modelDirectoryUrl, _): (URL, TinyDenseArtifactFixture.SynthesizedLayout) =
            try TinyDenseArtifactFixture.writeModelDirectory();
        defer { try? FileManager.default.removeItem(at: modelDirectoryUrl); }
        let validatedArtifact: ValidatedQwen35Artifact = try Qwen35ArtifactValidator()
            .validate(
                modelDirectory: modelDirectoryUrl.path,
                maxOutputTokens: TinyDenseArtifactFixture.MAX_OUTPUT_TOKENS);

        let engine: Qwen35DenseEngine = Qwen35DenseEngine();
        try engine.loadValidatedArtifact(validatedArtifact);
        do {
            try engine.loadValidatedArtifact(validatedArtifact);
            Issue.record("a second load of one validated artifact must fail closed");
        } catch let loadError as InferenceEngineError {
            guard case .modelLoad = loadError else {
                Issue.record("expected a model load failure, got \(loadError)");
                return;
            }
        }
    }

    /**
     * The production runtime builder validates the directory, streams the
     * artifact weights, bridges the tokenizer, and serves one tokenized
     * chat generation end to end: the prompt renders through the chat
     * template, every generated token stays inside the config vocabulary,
     * and the request-local translator finishes cleanly.
     */
    @Test(.timeLimit(.minutes(2)))
    func should_serve_a_tokenized_chat_generation_from_the_artifact_runtime() throws {
        let (modelDirectoryUrl, _): (URL, TinyDenseArtifactFixture.SynthesizedLayout) =
            try TinyDenseArtifactFixture.writeModelDirectory(includeTokenizerFiles: true);
        defer { try? FileManager.default.removeItem(at: modelDirectoryUrl); }
        let runtime: LoadedChatRuntime = try Qwen35ChatRuntime.buildArtifactRuntime(
            modelDirectory: modelDirectoryUrl.path,
            modelConfiguration: WorkerModelConfiguration.autoregressive(
                Qwen35ArtifactWeightLoadingTests.autoregressiveConfiguration()));

        let activeGeneration: any ActiveChatGeneration = try runtime.processor.prepareChatGeneration(
            Qwen35ArtifactWeightLoadingTests.chatCommand(requestId: 151));
        #expect(activeGeneration.promptTokenCount > 0);

        let requestId: RequestId = RequestId(rawRequestId: 151);
        _ = try runtime.engine.startGeneration(activeGeneration.inferenceRequest);
        var generatedTokenCount: Int = 0;
        var sawEndOfSequence: Bool = false;
        for _ in 0..<48 {
            let boundary: GeneratedToken = try runtime.engine.decodeNextToken(requestId: requestId);
            guard case let .tokenId(generatedTokenId, _, _, _, _, _) = boundary else {
                continue;
            };
            #expect(generatedTokenId < 256);
            generatedTokenCount += 1;
            if activeGeneration.isEndOfSequenceToken(generatedTokenId) {
                sawEndOfSequence = true;
                break;
            }
            if generatedTokenCount >= 12 {
                break;
            }
        }
        #expect(generatedTokenCount > 0);

        let flushedOutputs: Array<ChatGenerationOutput> = try activeGeneration.finishOutputs();
        _ = try runtime.engine.cancelGeneration(requestId: requestId);
        // The flushed batch is bounded by the emitted markers; with no tool
        // parser the journey only proves the flush stays consistent.
        #expect(flushedOutputs.count <= generatedTokenCount);
        #expect(sawEndOfSequence || generatedTokenCount >= 12);
    }

    /**
     * A directory whose shard file disappears after synthesis fails the
     * runtime build closed at validation, naming the absent shard file.
     */
    @Test(.timeLimit(.minutes(1)))
    func should_fail_the_artifact_runtime_when_validation_rejects_the_directory() throws {
        let (modelDirectoryUrl, _): (URL, TinyDenseArtifactFixture.SynthesizedLayout) =
            try TinyDenseArtifactFixture.writeModelDirectory(includeTokenizerFiles: true);
        defer { try? FileManager.default.removeItem(at: modelDirectoryUrl); }
        try FileManager.default.removeItem(
            at: modelDirectoryUrl.appendingPathComponent(TinyDenseArtifactFixture.SHARD_FILE_NAME));

        do {
            _ = try Qwen35ChatRuntime.buildArtifactRuntime(
                modelDirectory: modelDirectoryUrl.path,
                modelConfiguration: WorkerModelConfiguration.autoregressive(
                    Qwen35ArtifactWeightLoadingTests.autoregressiveConfiguration()));
            Issue.record("an artifact without its shard file must fail the runtime build");
        } catch let validationError as Qwen35ArtifactValidationError {
            guard case .artifact(.inspectRequiredFile(let missingFileName, _)) = validationError else {
                Issue.record("expected the missing shard file to be named, got \(validationError)");
                return;
            }
            #expect(missingFileName == TinyDenseArtifactFixture.SHARD_FILE_NAME);
        }
    }

    // MARK: - Fixtures

    /// Romeo and Juliet prompt bytes are the model-visible token ids; every
    /// byte value stays inside the tiny vocabulary.
    private static func fixturePromptTokenIds() -> Array<UInt32> {
        return ROMEO_AND_JULIET_PROMPT.utf8.map { (promptByte: UInt8) -> UInt32 in
            return UInt32(promptByte) % 256;
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

    /// Streams one artifact directory's weights and collects a seeded
    /// token stream, proving each directory yields its own model.
    private static func collectTokenIds(
        modelDirectoryUrl: URL, promptTokenIds: Array<UInt32>, seed: UInt64
    ) throws -> Array<UInt32> {
        let validatedArtifact: ValidatedQwen35Artifact = try Qwen35ArtifactValidator()
            .validate(
                modelDirectory: modelDirectoryUrl.path,
                maxOutputTokens: TinyDenseArtifactFixture.MAX_OUTPUT_TOKENS);
        let engine: Qwen35DenseEngine = Qwen35DenseEngine();
        try engine.loadValidatedArtifact(validatedArtifact);
        let requestId: RequestId = RequestId(rawRequestId: 97);
        _ = try engine.startGeneration(Qwen35PreparedInferenceRequest(
            promptTokenIds: promptTokenIds,
            samplingSettings: Qwen35SamplingSettings(
                chatGenerationSettings: Qwen35ArtifactWeightLoadingTests.settings(seed: seed))));
        var collectedTokenIds: Array<UInt32> = [];
        while collectedTokenIds.count < 5 {
            let boundary: GeneratedToken = try engine.decodeNextToken(requestId: requestId);
            if case let .tokenId(generatedTokenId, _, _, _, _, _) = boundary {
                collectedTokenIds.append(generatedTokenId);
            }
        }
        _ = try engine.cancelGeneration(requestId: requestId);
        return collectedTokenIds;
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

    private static func chatCommand(requestId: UInt64) -> ChatGenerationCommand {
        return ChatGenerationCommand(
            requestId: RequestId(rawRequestId: requestId),
            model: "qwen3.5",
            messages: [
                .system(content: "answer plainly"),
                .user(content: ROMEO_AND_JULIET_PROMPT, images: []),
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
}
