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
}
