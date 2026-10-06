import Foundation;

import IpcProtocol;
import MLXHuggingFace;
import MLXLMCommon;
import Tokenizers;

/// Builds the matched dense Qwen3.5 chat runtime from one model directory:
/// the artifact config drives the in-memory engine construction and the
/// directory's tokenizer files drive the bridged upstream tokenizer.
///
/// The production family factory wraps this builder behind the weight-
/// streaming gate; until validated shards stream into the engine, weights
/// are the pinned in-memory initialization, so the production swap path
/// keeps failing closed while the pairing itself is fully exercised here.
public enum Qwen35ChatRuntime {

    public static func buildInMemoryRuntime(
        modelDirectory: String,
        modelConfiguration: WorkerModelConfiguration,
        prefillChunkTokenCount: Int = 512
    ) throws -> LoadedChatRuntime {
        guard let autoregressiveConfiguration = modelConfiguration.autoregressive() else {
            throw InferenceEngineError.modelLoad(
                reason: "the dense Qwen3.5 runtime requires an autoregressive model policy");
        }
        let directoryUrl: URL = URL(fileURLWithPath: modelDirectory);
        let configBytes: Data;
        do {
            configBytes = try Data(contentsOf: directoryUrl.appendingPathComponent("config.json"));
        } catch {
            throw InferenceEngineError.modelLoad(
                reason: "the dense model configuration file could not be read");
        }
        let repositoryConfiguration: Qwen3_5Config;
        do {
            repositoryConfiguration = try Qwen3_5Config.fromJsonBytes(
                configBytes: Array(configBytes));
        } catch {
            throw InferenceEngineError.modelLoad(
                reason: "the dense model configuration could not be decoded");
        }
        let engine: Qwen35DenseEngine = Qwen35DenseEngine(
            prefillChunkTokenCount: prefillChunkTokenCount);
        do {
            try engine.loadInMemoryModel(configBytes: configBytes);
        } catch let engineError as InferenceEngineError {
            throw engineError;
        } catch {
            throw InferenceEngineError.modelLoad(
                reason: "the dense model could not be constructed");
        }
        let bridgedTokenizer: any MLXLMCommon.Tokenizer;
        do {
            bridgedTokenizer = #adaptHuggingFaceTokenizer(
                try Qwen35ChatRuntime.blockingTokenizerLoad(directoryUrl: directoryUrl));
        } catch {
            throw InferenceEngineError.modelLoad(
                reason: "the model tokenizer files could not be loaded: \(error)");
        }
        let processor: Qwen35ChatProcessor = Qwen35ChatProcessor(
            tokenizer: bridgedTokenizer,
            modelId: autoregressiveConfiguration.modelId,
            endOfSequenceTokenIds: Set(
                repositoryConfiguration.endOfSequenceTokenIds()),
            capabilities: Qwen35ChatProcessor.capabilities(
                autoregressiveConfiguration: autoregressiveConfiguration,
                maximumPositionCount: repositoryConfiguration.maximumPositionCount()));
        return LoadedChatRuntime(processor: processor, engine: engine);
    }

    /// Blocks the synchronous factory boundary on the async upstream
    /// tokenizer load, exactly as the Rust factory's spawn_blocking boundary
    /// does; the worker loop stays the single MLX owner around it. The boxed
    /// result is the task's only shared state and is @unchecked Sendable by
    /// the semaphore handshake.
    private static func blockingTokenizerLoad(
        directoryUrl: URL
    ) throws -> any Tokenizers.Tokenizer {
        final class LoadOutcome: @unchecked Sendable {
            var result: Result<any Tokenizers.Tokenizer, Error>?
        }
        let loadOutcome: LoadOutcome = LoadOutcome();
        let loadSemaphore: DispatchSemaphore = DispatchSemaphore(value: 0);
        let detachedTask: Task<Void, Never> = Task.detached(priority: .userInitiated) {
            do {
                loadOutcome.result = .success(
                    try await Tokenizers.AutoTokenizer.from(modelFolder: directoryUrl));
            } catch {
                loadOutcome.result = .failure(error);
            }
            loadSemaphore.signal();
        }
        loadSemaphore.wait();
        detachedTask.cancel();
        let outcome: Result<any Tokenizers.Tokenizer, Error> = loadOutcome.result!;
        return try outcome.get();
    }
}
