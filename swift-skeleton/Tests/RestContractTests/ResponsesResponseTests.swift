import XCTest;
import RestContract;
import IpcProtocol;

/// Ported from crates/rest-contract/tests/rest_api/openai_responses_response.rs.
final class ResponsesResponseTests: XCTestCase {

    func testShouldSerializeRawModelReasoningAsAPlaintextSummaryWithoutEncryptedContent() throws {
        let response: OpenAiResponse = OpenAiResponse.completed(
            responseId: "resp_instance-7",
            createdAt: 1_753_000_000,
            completedAt: 1_753_000_007,
            modelId: "astronomical/fake-mixture-of-experts",
            instructions: "Be precise.",
            output: [
                OpenAiResponseOutputItem.reasoning(id: "rs_instance-7", reasoningText: "Inspect first."),
                OpenAiResponseOutputItem.message(id: "msg_instance-7", outputText: "Done."),
                OpenAiResponseOutputItem.functionCall(
                    id: "fc_instance-7-0",
                    callId: "call_instance-7-0",
                    functionName: "read",
                    argumentsJson: #"{"filePath":"README.md"}"#),
            ],
            usage: try XCTUnwrap(OpenAiResponseUsage.new(
                inputTokens: 100, outputTokens: 20, cachedTokens: 64, reasoningTokens: 7)));
        let responseDocument: JsonWireValue = response.wireValue();
        XCTAssertEqual(
            ResponseFormatValueAccess.field(responseDocument, key: "object"),
            .string("response"));
        XCTAssertEqual(
            ResponseFormatValueAccess.field(responseDocument, key: "status"),
            .string("completed"));
        XCTAssertEqual(
            ResponseFormatValueAccess.field(responseDocument, key: "created_at"),
            .unsignedInteger(1_753_000_000));
        XCTAssertEqual(
            ResponseFormatValueAccess.field(responseDocument, key: "completed_at"),
            .unsignedInteger(1_753_000_007));
        guard case .array(let outputItems) = ResponseFormatValueAccess.field(responseDocument, key: "output") else {
            XCTFail("expected an output array");
            return;
        }
        XCTAssertEqual(outputItems.count, 3);
        let reasoningItem: JsonWireValue = outputItems[0];
        XCTAssertEqual(
            ResponseFormatValueAccess.field(reasoningItem, key: "type"),
            .string("reasoning"));
        guard case .array(let reasoningSummaries) = ResponseFormatValueAccess.field(reasoningItem, key: "summary") else {
            XCTFail("expected a reasoning summary array");
            return;
        }
        XCTAssertEqual(
            ResponseFormatValueAccess.field(reasoningSummaries[0], key: "type"),
            .string("summary_text"));
        XCTAssertEqual(
            ResponseFormatValueAccess.field(reasoningSummaries[0], key: "text"),
            .string("Inspect first."));
        XCTAssertEqual(
            ResponseFormatValueAccess.field(reasoningItem, key: "content"),
            .array(Array<JsonWireValue>()));
        // serde_json's Index returns Null for a missing object key, so the
        // Rust is_null() check passes when the absent encrypted_content field
        // is simply not serialized; assert the same absent-or-null shape.
        if let encryptedContent: JsonWireValue = ResponseFormatValueAccess.field(reasoningItem, key: "encrypted_content") {
            guard case .null = encryptedContent else {
                XCTFail("encrypted_content must serialize as null for raw reasoning");
                return;
            }
        }
        let messageItem: JsonWireValue = outputItems[1];
        XCTAssertEqual(
            ResponseFormatValueAccess.field(messageItem, key: "type"),
            .string("message"));
        guard case .array(let messageContent) = ResponseFormatValueAccess.field(messageItem, key: "content") else {
            XCTFail("expected a message content array");
            return;
        }
        XCTAssertEqual(
            ResponseFormatValueAccess.field(messageContent[0], key: "text"),
            .string("Done."));
        let functionCallItem: JsonWireValue = outputItems[2];
        XCTAssertEqual(
            ResponseFormatValueAccess.field(functionCallItem, key: "type"),
            .string("function_call"));
        XCTAssertEqual(
            ResponseFormatValueAccess.field(functionCallItem, key: "name"),
            .string("read"));
        guard case .object(let usageObject) = ResponseFormatValueAccess.field(responseDocument, key: "usage") else {
            XCTFail("expected a usage object");
            return;
        }
        XCTAssertEqual(
            usageObject.value(forKey: "input_tokens"),
            .unsignedInteger(100));
        XCTAssertEqual(
            ResponseFormatValueAccess.field(usageObject.value(forKey: "input_tokens_details"), key: "cached_tokens"),
            .unsignedInteger(64));
        XCTAssertEqual(
            ResponseFormatValueAccess.field(usageObject.value(forKey: "output_tokens_details"), key: "reasoning_tokens"),
            .unsignedInteger(7));
        XCTAssertEqual(
            usageObject.value(forKey: "total_tokens"),
            .unsignedInteger(120));
    }
}
