import Foundation;

/// A model-output event translated from Qwen3.5 syntax into neutral structured
/// output. Dense and MoE share the parser that emits these events.
///
/// Mirrors crates/model-serving/src/qwen3_5/text/output_parser.rs
/// `Qwen3_5OutputEvent`. The Rust variant for model-visible corrections has no
/// Swift counterpart because the Swift worker does not yet feed tokens back
/// into the model context mid-generation.
public enum Qwen35OutputEvent: Equatable, Sendable {
    /// Reasoning content with think markers removed.
    case reasoningDelta(String);
    /// Normal assistant response content with control markers removed.
    case textDelta(String);
    /// One complete function call, including well-formed names the request did not declare.
    case toolCall(Qwen35ToolCall);
}

/// One validated function call parsed from a complete Qwen3.5 XML block.
public struct Qwen35ToolCall: Equatable, Sendable {
    /// The zero-based output order of this function call.
    public let index: UInt16;
    /// The selected function name.
    public let functionName: String;
    /// Canonical JSON object arguments for the selected function.
    public let argumentsJson: String;

    public init(index: UInt16, functionName: String, argumentsJson: String) {
        self.index = index;
        self.functionName = functionName;
        self.argumentsJson = argumentsJson;
    }
}
