import Foundation

import IpcProtocol

/**
 * One emitted tool call with bounded arguments, mirroring the Rust record
 * from apps/supervisor/src/completion_attribution_log.rs.
 */
public struct CompletionToolCallRecord: Equatable {

    /// Emission order of this tool call within the generation.
    public let toolCallIndex: UInt16

    /// The function name the model emitted.
    public let functionName: String

    /// The arguments JSON, bounded so the log never grows without bound.
    public let arguments: CompletionArgumentsRecord

    /**
     * Builds one bounded tool-call record from the raw arguments JSON.
     *
     * Arguments at or under the cap are recorded verbatim. Larger arguments
     * are truncated to the cap and the full original is hashed so identical
     * payloads correlate regardless of truncation.
     */
    public static func fromArguments(
        toolCallIndex: UInt16,
        functionName: String,
        argumentsJson: String
    ) -> CompletionToolCallRecord {
        return CompletionToolCallRecord(
            toolCallIndex: toolCallIndex,
            functionName: functionName,
            arguments: CompletionArgumentsRecord.fromArgumentsJson(argumentsJson))
    }

    public init(
        toolCallIndex: UInt16,
        functionName: String,
        arguments: CompletionArgumentsRecord
    ) {
        self.toolCallIndex = toolCallIndex
        self.functionName = functionName
        self.arguments = arguments
    }

    /// The serde-shaped JSON object for the completion log's `tool_calls` entry.
    public func jsonlWireValue() -> JsonWireValue {
        var wireObject: JsonWireObject = JsonWireObject(entries: [])
        wireObject.appendEntry(key: "tool_call_index", value: .unsignedInteger(UInt64(self.toolCallIndex)))
        wireObject.appendEntry(key: "function_name", value: .string(self.functionName))
        var argumentsObject: JsonWireObject = JsonWireObject(entries: [])
        argumentsObject.appendEntry(key: "size_bytes", value: .unsignedInteger(UInt64(self.arguments.sizeBytes)))
        argumentsObject.appendEntry(key: "sha256", value: .string(self.arguments.sha256))
        argumentsObject.appendEntry(key: "json", value: .string(self.arguments.json))
        argumentsObject.appendEntry(key: "truncated", value: .boolean(self.arguments.truncated))
        wireObject.appendEntry(key: "arguments", value: .object(argumentsObject))
        return .object(wireObject)
    }
}
