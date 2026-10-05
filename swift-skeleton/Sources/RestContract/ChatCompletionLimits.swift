import Foundation;

/// Public bounds shared by the Chat Completions wire types. Port of the
/// constants at the top of
/// crates/rest-contract/src/openai_chat_completion_request.rs.
public enum ChatCompletionLimits {

    /// The maximum accepted nesting depth of a function JSON Schema.
    public static let MAX_OPENAI_TOOL_SCHEMA_NESTING_DEPTH: Int = 32;

    /// Maximum generated-token budget representable by the current worker protocol.
    /// The model-serving layer still validates prompt plus output against model context.
    public static let MAX_OPENAI_OUTPUT_TOKENS: UInt32 = UInt32(UInt16.max);

    /// The fallback generated-token budget when a client does not send one.
    public static let DEFAULT_OPENAI_OUTPUT_TOKENS: UInt32 = 1_024;
}
