// ResponsesStream.swift — RestContract
//
// Port of crates/rest-contract/src/openai_responses_stream.rs.

import Foundation;
import IpcProtocol;

/// One semantic Server-Sent Events payload from the local Responses endpoint.
public enum OpenAiResponseStreamEvent: Equatable {
    case created(sequenceNumber: UInt64, response: OpenAiResponse);
    case inProgress(sequenceNumber: UInt64, response: OpenAiResponse);
    case outputItemAdded(sequenceNumber: UInt64, outputIndex: Int, item: OpenAiResponseOutputItem);
    case outputItemDone(sequenceNumber: UInt64, outputIndex: Int, item: OpenAiResponseOutputItem);
    case contentPartAdded(
        sequenceNumber: UInt64, itemId: String, outputIndex: Int, contentIndex: Int,
        part: OpenAiResponseOutputContent);
    case contentPartDone(
        sequenceNumber: UInt64, itemId: String, outputIndex: Int, contentIndex: Int,
        part: OpenAiResponseOutputContent);
    case reasoningSummaryTextDelta(
        sequenceNumber: UInt64, itemId: String, outputIndex: Int, summaryIndex: Int,
        delta: String);
    case reasoningSummaryTextDone(
        sequenceNumber: UInt64, itemId: String, outputIndex: Int, summaryIndex: Int,
        text: String);
    case outputTextDelta(
        sequenceNumber: UInt64, itemId: String, outputIndex: Int, contentIndex: Int,
        delta: String, logprobs: Array<JsonWireValue>);
    case outputTextDone(
        sequenceNumber: UInt64, itemId: String, outputIndex: Int, contentIndex: Int,
        text: String, logprobs: Array<JsonWireValue>);
    case functionCallArgumentsDelta(
        sequenceNumber: UInt64, itemId: String, outputIndex: Int, delta: String);
    case functionCallArgumentsDone(
        sequenceNumber: UInt64, itemId: String, outputIndex: Int, name: String,
        arguments: String);
    case completed(sequenceNumber: UInt64, response: OpenAiResponse);
    case incomplete(sequenceNumber: UInt64, response: OpenAiResponse);
    case failed(sequenceNumber: UInt64, response: OpenAiResponse);
    case error(sequenceNumber: UInt64, code: String?, message: String, param: String?);

    /// The exact serde `rename` spelling used as the wire event type tag.
    public func eventType() -> String {
        switch self {
        case .created: return "response.created";
        case .inProgress: return "response.in_progress";
        case .outputItemAdded: return "response.output_item.added";
        case .outputItemDone: return "response.output_item.done";
        case .contentPartAdded: return "response.content_part.added";
        case .contentPartDone: return "response.content_part.done";
        case .reasoningSummaryTextDelta: return "response.reasoning_summary_text.delta";
        case .reasoningSummaryTextDone: return "response.reasoning_summary_text.done";
        case .outputTextDelta: return "response.output_text.delta";
        case .outputTextDone: return "response.output_text.done";
        case .functionCallArgumentsDelta: return "response.function_call_arguments.delta";
        case .functionCallArgumentsDone: return "response.function_call_arguments.done";
        case .completed: return "response.completed";
        case .incomplete: return "response.incomplete";
        case .failed: return "response.failed";
        case .error: return "error";
        }
    }

    /// The ordered stream position carried by every event variant.
    public func sequenceNumber() -> UInt64 {
        switch self {
        case .created(let sequenceNumber, _): return sequenceNumber;
        case .inProgress(let sequenceNumber, _): return sequenceNumber;
        case .outputItemAdded(let sequenceNumber, _, _): return sequenceNumber;
        case .outputItemDone(let sequenceNumber, _, _): return sequenceNumber;
        case .contentPartAdded(let sequenceNumber, _, _, _, _): return sequenceNumber;
        case .contentPartDone(let sequenceNumber, _, _, _, _): return sequenceNumber;
        case .reasoningSummaryTextDelta(let sequenceNumber, _, _, _, _): return sequenceNumber;
        case .reasoningSummaryTextDone(let sequenceNumber, _, _, _, _): return sequenceNumber;
        case .outputTextDelta(let sequenceNumber, _, _, _, _, _): return sequenceNumber;
        case .outputTextDone(let sequenceNumber, _, _, _, _, _): return sequenceNumber;
        case .functionCallArgumentsDelta(let sequenceNumber, _, _, _): return sequenceNumber;
        case .functionCallArgumentsDone(let sequenceNumber, _, _, _, _): return sequenceNumber;
        case .completed(let sequenceNumber, _): return sequenceNumber;
        case .incomplete(let sequenceNumber, _): return sequenceNumber;
        case .failed(let sequenceNumber, _): return sequenceNumber;
        case .error(let sequenceNumber, _, _, _): return sequenceNumber;
        }
    }

    /// Serializes the internally tagged event exactly as the serde derive
    /// does: the `type` tag first, then the variant fields in declaration
    /// order, with `None` Options emitted as JSON null.
    public func wireValue() -> JsonWireValue {
        var eventObject: JsonWireObject = JsonWireObject(entries: Array());
        eventObject.appendEntry(key: "type", value: .string(self.eventType()));
        switch self {
        case .created(let sequenceNumber, let response):
            eventObject.appendEntry(key: "sequence_number", value: .unsignedInteger(sequenceNumber));
            eventObject.appendEntry(key: "response", value: response.wireValue());
        case .inProgress(let sequenceNumber, let response):
            eventObject.appendEntry(key: "sequence_number", value: .unsignedInteger(sequenceNumber));
            eventObject.appendEntry(key: "response", value: response.wireValue());
        case .outputItemAdded(let sequenceNumber, let outputIndex, let item):
            eventObject.appendEntry(key: "sequence_number", value: .unsignedInteger(sequenceNumber));
            eventObject.appendEntry(key: "output_index", value: .unsignedInteger(UInt64(outputIndex)));
            eventObject.appendEntry(key: "item", value: item.wireValue());
        case .outputItemDone(let sequenceNumber, let outputIndex, let item):
            eventObject.appendEntry(key: "sequence_number", value: .unsignedInteger(sequenceNumber));
            eventObject.appendEntry(key: "output_index", value: .unsignedInteger(UInt64(outputIndex)));
            eventObject.appendEntry(key: "item", value: item.wireValue());
        case .contentPartAdded(
            let sequenceNumber, let itemId, let outputIndex, let contentIndex, let part):
            eventObject.appendEntry(key: "sequence_number", value: .unsignedInteger(sequenceNumber));
            eventObject.appendEntry(key: "item_id", value: .string(itemId));
            eventObject.appendEntry(key: "output_index", value: .unsignedInteger(UInt64(outputIndex)));
            eventObject.appendEntry(key: "content_index", value: .unsignedInteger(UInt64(contentIndex)));
            eventObject.appendEntry(key: "part", value: part.wireValue());
        case .contentPartDone(
            let sequenceNumber, let itemId, let outputIndex, let contentIndex, let part):
            eventObject.appendEntry(key: "sequence_number", value: .unsignedInteger(sequenceNumber));
            eventObject.appendEntry(key: "item_id", value: .string(itemId));
            eventObject.appendEntry(key: "output_index", value: .unsignedInteger(UInt64(outputIndex)));
            eventObject.appendEntry(key: "content_index", value: .unsignedInteger(UInt64(contentIndex)));
            eventObject.appendEntry(key: "part", value: part.wireValue());
        case .reasoningSummaryTextDelta(
            let sequenceNumber, let itemId, let outputIndex, let summaryIndex, let delta):
            eventObject.appendEntry(key: "sequence_number", value: .unsignedInteger(sequenceNumber));
            eventObject.appendEntry(key: "item_id", value: .string(itemId));
            eventObject.appendEntry(key: "output_index", value: .unsignedInteger(UInt64(outputIndex)));
            eventObject.appendEntry(key: "summary_index", value: .unsignedInteger(UInt64(summaryIndex)));
            eventObject.appendEntry(key: "delta", value: .string(delta));
        case .reasoningSummaryTextDone(
            let sequenceNumber, let itemId, let outputIndex, let summaryIndex, let text):
            eventObject.appendEntry(key: "sequence_number", value: .unsignedInteger(sequenceNumber));
            eventObject.appendEntry(key: "item_id", value: .string(itemId));
            eventObject.appendEntry(key: "output_index", value: .unsignedInteger(UInt64(outputIndex)));
            eventObject.appendEntry(key: "summary_index", value: .unsignedInteger(UInt64(summaryIndex)));
            eventObject.appendEntry(key: "text", value: .string(text));
        case .outputTextDelta(
            let sequenceNumber, let itemId, let outputIndex, let contentIndex, let delta,
            let logprobs):
            eventObject.appendEntry(key: "sequence_number", value: .unsignedInteger(sequenceNumber));
            eventObject.appendEntry(key: "item_id", value: .string(itemId));
            eventObject.appendEntry(key: "output_index", value: .unsignedInteger(UInt64(outputIndex)));
            eventObject.appendEntry(key: "content_index", value: .unsignedInteger(UInt64(contentIndex)));
            eventObject.appendEntry(key: "delta", value: .string(delta));
            eventObject.appendEntry(key: "logprobs", value: .array(logprobs));
        case .outputTextDone(
            let sequenceNumber, let itemId, let outputIndex, let contentIndex, let text,
            let logprobs):
            eventObject.appendEntry(key: "sequence_number", value: .unsignedInteger(sequenceNumber));
            eventObject.appendEntry(key: "item_id", value: .string(itemId));
            eventObject.appendEntry(key: "output_index", value: .unsignedInteger(UInt64(outputIndex)));
            eventObject.appendEntry(key: "content_index", value: .unsignedInteger(UInt64(contentIndex)));
            eventObject.appendEntry(key: "text", value: .string(text));
            eventObject.appendEntry(key: "logprobs", value: .array(logprobs));
        case .functionCallArgumentsDelta(let sequenceNumber, let itemId, let outputIndex, let delta):
            eventObject.appendEntry(key: "sequence_number", value: .unsignedInteger(sequenceNumber));
            eventObject.appendEntry(key: "item_id", value: .string(itemId));
            eventObject.appendEntry(key: "output_index", value: .unsignedInteger(UInt64(outputIndex)));
            eventObject.appendEntry(key: "delta", value: .string(delta));
        case .functionCallArgumentsDone(
            let sequenceNumber, let itemId, let outputIndex, let name, let arguments):
            eventObject.appendEntry(key: "sequence_number", value: .unsignedInteger(sequenceNumber));
            eventObject.appendEntry(key: "item_id", value: .string(itemId));
            eventObject.appendEntry(key: "output_index", value: .unsignedInteger(UInt64(outputIndex)));
            eventObject.appendEntry(key: "name", value: .string(name));
            eventObject.appendEntry(key: "arguments", value: .string(arguments));
        case .completed(let sequenceNumber, let response):
            eventObject.appendEntry(key: "sequence_number", value: .unsignedInteger(sequenceNumber));
            eventObject.appendEntry(key: "response", value: response.wireValue());
        case .incomplete(let sequenceNumber, let response):
            eventObject.appendEntry(key: "sequence_number", value: .unsignedInteger(sequenceNumber));
            eventObject.appendEntry(key: "response", value: response.wireValue());
        case .failed(let sequenceNumber, let response):
            eventObject.appendEntry(key: "sequence_number", value: .unsignedInteger(sequenceNumber));
            eventObject.appendEntry(key: "response", value: response.wireValue());
        case .error(let sequenceNumber, let code, let message, let param):
            eventObject.appendEntry(key: "sequence_number", value: .unsignedInteger(sequenceNumber));
            eventObject.appendEntry(
                key: "code", value: ResponsesStreamWireSupport.optionalString(code));
            eventObject.appendEntry(key: "message", value: .string(message));
            eventObject.appendEntry(
                key: "param", value: ResponsesStreamWireSupport.optionalString(param));
        }
        return .object(eventObject);
    }
}

/// Module-private serialization helpers for the stream event payloads.
fileprivate enum ResponsesStreamWireSupport {

    fileprivate static func optionalString(_ optionalValue: String?) -> JsonWireValue {
        guard let unwrappedValue: String = optionalValue else {
            return .null;
        }
        return .string(unwrappedValue);
    }
}
