import Foundation;
import RestContract;
import IpcProtocol;
import Testing;
import JourneyCategories;

/// Ported from crates/rest-contract/tests/rest_api/openai_responses_response.rs.
@Suite(.tags(.hermeticJourney))
final class ResponsesResponseTests {

    @Test
    func should_serialize_raw_model_reasoning_as_a_plaintext_summary_without_encrypted_content() throws {
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
            usage: try #require(OpenAiResponseUsage.new(
                inputTokens: 100, outputTokens: 20, cachedTokens: 64, reasoningTokens: 7)));
        let responseDocument: JsonWireValue = response.wireValue();
        #expect(
            ResponseFormatValueAccess.field(responseDocument, key: "object")
                == JsonWireValue.string("response"));
        #expect(
            ResponseFormatValueAccess.field(responseDocument, key: "status")
                == JsonWireValue.string("completed"));
        #expect(
            ResponseFormatValueAccess.field(responseDocument, key: "created_at")
                == JsonWireValue.unsignedInteger(1_753_000_000));
        #expect(
            ResponseFormatValueAccess.field(responseDocument, key: "completed_at")
                == JsonWireValue.unsignedInteger(1_753_000_007));
        guard case .array(let outputItems) = ResponseFormatValueAccess.field(responseDocument, key: "output") else {
            Issue.record("expected an output array");
            return;
        }
        #expect(outputItems.count == 3);
        let reasoningItem: JsonWireValue = outputItems[0];
        #expect(
            ResponseFormatValueAccess.field(reasoningItem, key: "type")
                == JsonWireValue.string("reasoning"));
        guard case .array(let reasoningSummaries) = ResponseFormatValueAccess.field(reasoningItem, key: "summary") else {
            Issue.record("expected a reasoning summary array");
            return;
        }
        #expect(
            ResponseFormatValueAccess.field(reasoningSummaries[0], key: "type")
                == JsonWireValue.string("summary_text"));
        #expect(
            ResponseFormatValueAccess.field(reasoningSummaries[0], key: "text")
                == JsonWireValue.string("Inspect first."));
        #expect(
            ResponseFormatValueAccess.field(reasoningItem, key: "content")
                == JsonWireValue.array(Array<JsonWireValue>()));
        // serde_json's Index returns Null for a missing object key, so the
        // Rust is_null() check passes when the absent encrypted_content field
        // is simply not serialized; assert the same absent-or-null shape.
        if let encryptedContent: JsonWireValue = ResponseFormatValueAccess.field(reasoningItem, key: "encrypted_content") {
            guard case .null = encryptedContent else {
                Issue.record("encrypted_content must serialize as null for raw reasoning");
                return;
            }
        }
        let messageItem: JsonWireValue = outputItems[1];
        #expect(
            ResponseFormatValueAccess.field(messageItem, key: "type")
                == JsonWireValue.string("message"));
        guard case .array(let messageContent) = ResponseFormatValueAccess.field(messageItem, key: "content") else {
            Issue.record("expected a message content array");
            return;
        }
        #expect(
            ResponseFormatValueAccess.field(messageContent[0], key: "text")
                == JsonWireValue.string("Done."));
        let functionCallItem: JsonWireValue = outputItems[2];
        #expect(
            ResponseFormatValueAccess.field(functionCallItem, key: "type")
                == JsonWireValue.string("function_call"));
        #expect(
            ResponseFormatValueAccess.field(functionCallItem, key: "name")
                == JsonWireValue.string("read"));
        guard case .object(let usageObject) = ResponseFormatValueAccess.field(responseDocument, key: "usage") else {
            Issue.record("expected a usage object");
            return;
        }
        #expect(
            usageObject.value(forKey: "input_tokens")
                == JsonWireValue.unsignedInteger(100));
        #expect(
            ResponseFormatValueAccess.field(usageObject.value(forKey: "input_tokens_details"), key: "cached_tokens")
                == JsonWireValue.unsignedInteger(64));
        #expect(
            ResponseFormatValueAccess.field(usageObject.value(forKey: "output_tokens_details"), key: "reasoning_tokens")
                == JsonWireValue.unsignedInteger(7));
        #expect(
            usageObject.value(forKey: "total_tokens")
                == JsonWireValue.unsignedInteger(120));
    }
}
