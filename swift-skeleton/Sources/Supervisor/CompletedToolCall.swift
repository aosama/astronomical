import Foundation

/**
 * Raw accumulated tool call, captured at emission time and bounded at write
 * time, mirroring the Rust type from apps/supervisor/src/completion_attribution_log.rs.
 *
 * Stored on the active request so the completion event can attribute the
 * emitted content without re-reading the stream.
 */
public struct CompletedToolCall: Equatable {

    /// Emission order of this tool call within the generation.
    public let toolCallIndex: UInt16

    /// The function name the model emitted.
    public let functionName: String

    /// The raw arguments JSON exactly as the model emitted it.
    public let argumentsJson: String

    public init(toolCallIndex: UInt16, functionName: String, argumentsJson: String) {
        self.toolCallIndex = toolCallIndex
        self.functionName = functionName
        self.argumentsJson = argumentsJson
    }
}
