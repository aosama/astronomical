import Foundation

import IpcProtocol

/**
 * One row in the completion attribution log, written when a chat generation
 * completes, mirroring the Rust record from
 * apps/supervisor/src/completion_attribution_log.rs.
 *
 * Fields are chosen to answer "what did the model emit?" at a glance:
 * - Identity: `request_id`, `model_id`
 * - Outcome: `completion_reason`
 * - Content: `tool_calls` (names + bounded arguments)
 */
public struct CompletionAttributionRecord: Equatable {

    /// Unix epoch milliseconds when the record was written (request completion time).
    public let timestampMillis: UInt64

    /// Supervisor-local monotonic request identifier.
    public let requestId: UInt64

    /// The model that produced this generation.
    public let modelId: String

    /// Why the generation stopped: `end_of_sequence`, `tool_calls`,
    /// `maximum_output_tokens`, or `cancelled`.
    public let completionReason: String

    /// The tool calls the model emitted, in emission order. Empty for
    /// non-tool-call completions.
    public let toolCalls: Array<CompletionToolCallRecord>

    public init(
        timestampMillis: UInt64,
        requestId: UInt64,
        modelId: String,
        completionReason: String,
        toolCalls: Array<CompletionToolCallRecord>
    ) {
        self.timestampMillis = timestampMillis
        self.requestId = requestId
        self.modelId = modelId
        self.completionReason = completionReason
        self.toolCalls = toolCalls
    }

    /// The serde-shaped JSON object written to `completion.jsonl`.
    public func jsonlWireValue() -> JsonWireValue {
        var wireObject: JsonWireObject = JsonWireObject(entries: [])
        wireObject.appendEntry(key: "timestamp_millis", value: .unsignedInteger(self.timestampMillis))
        wireObject.appendEntry(key: "request_id", value: .unsignedInteger(self.requestId))
        wireObject.appendEntry(key: "model_id", value: .string(self.modelId))
        wireObject.appendEntry(key: "completion_reason", value: .string(self.completionReason))
        wireObject.appendEntry(
            key: "tool_calls",
            value: JsonWireValue.mappedArray(self.toolCalls, mappedWireValue: { (toolCallRecord: CompletionToolCallRecord) -> JsonWireValue in
                return toolCallRecord.jsonlWireValue()
            }))
        return .object(wireObject)
    }
}
