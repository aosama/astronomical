import Foundation;

import IpcProtocol;

/// One matched chat runtime the model family factory produced: the
/// translation seam and the paired engine, loaded as one unit. The runtime
/// is owned by the single-threaded worker loop; Sendable conformance is the
/// factory hand-off contract, not a sharing promise.
public struct LoadedChatRuntime: @unchecked Sendable {

    public let processor: any ModelGenerationProcessor;
    public let engine: any InferenceEngine;

    public init(processor: any ModelGenerationProcessor, engine: any InferenceEngine) {
        self.processor = processor;
        self.engine = engine;
    }
}

/// Creates a matched chat runtime for one model directory and the supervisor
/// model configuration that selected it.
///
/// Mirrors the autoregressive arm of the Rust worker `ModelFactory`
/// (crates/model-serving/src/engine_backed_worker/support.rs). The factory
/// derives family selection from the configuration, never from filenames,
/// and returns a bounded failure reason the `modelSwapFailed` event carries.
public protocol ChatModelRuntimeFactory: Sendable {

    /// Builds the runtime for the requested model; throws a bounded,
    /// wire-safe reason when the family or payload is unusable.
    func createChatRuntime(
        modelDirectory: String,
        modelConfiguration: WorkerModelConfiguration
    ) throws -> LoadedChatRuntime;

    /// Validates a replacement selection without requiring it to load MLX
    /// weights before the worker retires the previous runtime.
    func createChatRuntimeCandidate(
        modelDirectory: String,
        modelConfiguration: WorkerModelConfiguration
    ) throws -> ChatRuntimeCandidate;

    /// Builds the runtime against the effective machine-adaptive MLX ceiling.
    func createChatRuntime(
        modelDirectory: String,
        modelConfiguration: WorkerModelConfiguration,
        effectiveMlxMemoryCeilingBytes: UInt64
    ) throws -> LoadedChatRuntime;

    /// Validates a replacement against the effective ceiling and defers MLX
    /// weight loading until `ChatRuntimeCandidate.load()`.
    func createChatRuntimeCandidate(
        modelDirectory: String,
        modelConfiguration: WorkerModelConfiguration,
        effectiveMlxMemoryCeilingBytes: UInt64
    ) throws -> ChatRuntimeCandidate;
}

extension ChatModelRuntimeFactory {

    public func createChatRuntime(
        modelDirectory: String,
        modelConfiguration: WorkerModelConfiguration,
        effectiveMlxMemoryCeilingBytes: UInt64
    ) throws -> LoadedChatRuntime {
        let runtimeCandidate: ChatRuntimeCandidate = try self.createChatRuntimeCandidate(
            modelDirectory: modelDirectory,
            modelConfiguration: modelConfiguration,
            effectiveMlxMemoryCeilingBytes: effectiveMlxMemoryCeilingBytes);
        return try runtimeCandidate.load();
    }

    /// Factories whose candidate does not consume memory-policy input can
    /// keep their candidate creation path and still defer runtime loading.
    public func createChatRuntimeCandidate(
        modelDirectory: String,
        modelConfiguration: WorkerModelConfiguration,
        effectiveMlxMemoryCeilingBytes: UInt64
    ) throws -> ChatRuntimeCandidate {
        return try self.createChatRuntimeCandidate(
            modelDirectory: modelDirectory,
            modelConfiguration: modelConfiguration);
    }
}
