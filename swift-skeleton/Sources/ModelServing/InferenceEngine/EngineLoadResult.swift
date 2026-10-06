import Foundation;

import IpcProtocol;

/// Engine resources loaded before the worker reports readiness.
///
/// Mirrors crates/model-serving/src/inference_engine/load_result.rs. The
/// minimum MLX ceiling is the one-byte floor a freshly loaded engine starts
/// from before its first request raises it; dense engines carry no expert
/// memory mode.
public struct EngineLoadResult: Equatable {

    public let minimumMlxMemoryCeilingBytes: UInt64;
    public let expertMemoryMode: ExpertMemoryMode?;

    public init(
        minimumMlxMemoryCeilingBytes: UInt64,
        expertMemoryMode: ExpertMemoryMode? = nil
    ) {
        self.minimumMlxMemoryCeilingBytes = minimumMlxMemoryCeilingBytes;
        self.expertMemoryMode = expertMemoryMode;
    }
}
