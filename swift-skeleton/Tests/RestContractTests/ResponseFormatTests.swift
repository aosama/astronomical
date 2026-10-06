import Foundation;
import RestContract;
import IpcProtocol;
import Testing;
import JourneyCategories;

/// Shared JSON-text fixture loader for the RestContract test target.
enum RestContractTestFixture {

    static func wireValue(_ jsonText: String) throws -> JsonWireValue {
        return try JsonWireParser.parseDocument(documentBytes: Data(jsonText.utf8));
    }
}

/// Ported from crates/rest-contract/tests/rest_api/openai_response_format.rs.
@Suite(.tags(.hermeticJourney))
final class ResponseFormatTests {

    @Test
    func should_accept_json_object_response_format_on_chat_completions() throws {
        let request: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
            {
                "model": "mlx-community/Qwen3.5-2B-4bit",
                "messages": [{"role": "user", "content": "O Romeo, Romeo, wherefore art thou Romeo?"}],
                "response_format": {"type": "json_object"}
            }
            """));
        let requestParts: OpenAiChatCompletionRequestParts = try request.intoParts();
        #expect(requestParts.structuredOutput == .jsonObject);
    }

    @Test
    func should_accept_json_schema_response_format_on_chat_completions() throws {
        let request: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
            {
                "model": "mlx-community/Qwen3.5-2B-4bit",
                "messages": [{"role": "user", "content": "O Romeo, Romeo, wherefore art thou Romeo?"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "romeo_line",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "speaker": {"type": "string"},
                                "play": {"type": "string"}
                            },
                            "required": ["speaker", "play"]
                        },
                        "strict": true
                    }
                }
            }
            """));
        let requestParts: OpenAiChatCompletionRequestParts = try request.intoParts();
        guard case .jsonSchema(let schemaName, _, _, let strict) = requestParts.structuredOutput else {
            Issue.record("expected json_schema, got \(String(describing: requestParts.structuredOutput))");
            return;
        }
        #expect(schemaName == "romeo_line");
        #expect(strict);
    }

    @Test
    func should_reject_an_unsupported_response_format_type() throws {
        let request: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
            {
                "model": "mlx-community/Qwen3.5-2B-4bit",
                "messages": [{"role": "user", "content": "O Romeo, Romeo, wherefore art thou Romeo?"}],
                "response_format": {"type": "xml"}
            }
            """));
        do {
            _ = try request.intoParts();
            Issue.record("unsupported response_format types must fail before worker admission");
        } catch let validationError as OpenAiChatCompletionValidationError {
            #expect(
                validationError.errorDescription
                    == OpenAiStructuredOutputValidationError.unsupportedType(formatType: "xml").errorDescription);
        }
    }

    @Test
    func should_accept_responses_text_format_json_schema() throws {
        let request: OpenAiResponsesRequest = try OpenAiResponsesRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
            {
                "model": "mlx-community/Qwen3.5-2B-4bit",
                "input": "O Romeo, Romeo, wherefore art thou Romeo?",
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "romeo_line",
                        "schema": {"type": "object"}
                    }
                }
            }
            """));
        let requestParts: OpenAiResponsesRequestParts = try request.intoParts();
        guard case .jsonSchema = requestParts.structuredOutput else {
            Issue.record("expected json_schema from Responses text.format");
            return;
        }
    }

    @Test
    func should_extract_json_from_fenced_model_text_without_filling_fields() throws {
        let extractedJson: JsonWireValue = try #require(
            OpenAiStructuredOutput.extract_json_value_from_text(
                "Juliet says:\n```json\n{\"speaker\":\"Juliet\",\"play\":\"Romeo and Juliet\"}\n```\n"));
        #expect(
            try extractedJson
                == RestContractTestFixture.wireValue(
                    #"{"speaker": "Juliet", "play": "Romeo and Juliet"}"#));
        #expect(OpenAiStructuredOutput.compact_extracted_json_text("not json at all") == nil);
    }

    @Test
    func should_name_unenforced_grammar_in_the_warning_header() {
        #expect(ResponseFormatConstants.UNENFORCED_RESPONSE_FORMAT_WARNING
            .contains("grammar-constrained decoding unavailable"));
    }

    @Test
    func should_accept_structured_outputs_choice() throws {
        let request: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
            {
                "model": "mlx-community/Qwen3.5-2B-4bit",
                "messages": [{"role": "user", "content": "O Romeo, Romeo, wherefore art thou Romeo?"}],
                "structured_outputs": {"choice": ["Juliet", "Romeo"]}
            }
            """));
        let requestParts: OpenAiChatCompletionRequestParts = try request.intoParts();
        guard case .choice = requestParts.enforcedStructuredGeneration else {
            Issue.record("expected a choice-enforced generation");
            return;
        }
    }

    @Test
    func should_enforce_structured_outputs_regex() throws {
        let request: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
            {
                "model": "mlx-community/Qwen3.5-2B-4bit",
                "messages": [{"role": "user", "content": "O Romeo, Romeo, wherefore art thou Romeo?"}],
                "structured_outputs": {"regex": "[A-Z]+"}
            }
            """));
        let requestParts: OpenAiChatCompletionRequestParts = try request.intoParts();
        #expect(requestParts.enforcedStructuredGeneration == .regex(pattern: "[A-Z]+"));
    }

    @Test
    func should_reject_an_uncompilable_regex_pattern() throws {
        let request: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
            {
                "model": "mlx-community/Qwen3.5-2B-4bit",
                "messages": [{"role": "user", "content": "O Romeo, Romeo, wherefore art thou Romeo?"}],
                "structured_outputs": {"regex": "(["}
            }
            """));
        do {
            _ = try request.intoParts();
            Issue.record("an uncompilable regex must fail closed");
        } catch let validationError as OpenAiChatCompletionValidationError {
            #expect(validationError.errorDescription?.contains("regex") == true,
                "unexpected error: \(String(describing: validationError.errorDescription))");
        }
    }

    @Test
    func should_reject_an_oversized_regex_pattern() throws {
        let oversizedPattern: String = String(
            repeating: "a",
            count: OpenAiStructuredOutputs.MAXIMUM_STRUCTURED_REGEX_PATTERN_BYTES + 1);
        let request: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
            {
                "model": "mlx-community/Qwen3.5-2B-4bit",
                "messages": [{"role": "user", "content": "O Romeo, Romeo, wherefore art thou Romeo?"}],
                "structured_outputs": {"regex": "\(oversizedPattern)"}
            }
            """));
        do {
            _ = try request.intoParts();
            Issue.record("an oversized regex pattern must fail closed");
        } catch let validationError as OpenAiChatCompletionValidationError {
            #expect(validationError.errorDescription?.contains("bounded") == true,
                "unexpected error: \(String(describing: validationError.errorDescription))");
        }
    }

    @Test
    func should_reject_guided_grammar_until_enforced() throws {
        let request: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
            {
                "model": "mlx-community/Qwen3.5-2B-4bit",
                "messages": [{"role": "user", "content": "O Romeo, Romeo, wherefore art thou Romeo?"}],
                "guided_grammar": "root ::= \\"Juliet\\" | \\"Romeo\\""
            }
            """));
        do {
            try request.validate();
            Issue.record("unenforced guided_grammar must fail closed");
        } catch let validationError as OpenAiChatCompletionValidationError {
            #expect(validationError.errorDescription?.contains("guided_grammar") == true,
                "unexpected error: \(String(describing: validationError.errorDescription))");
        }
    }

    @Test
    func should_reject_structured_outputs_and_guided_grammar_together() throws {
        let request: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
            {
                "model": "mlx-community/Qwen3.5-2B-4bit",
                "messages": [{"role": "user", "content": "O Romeo, Romeo, wherefore art thou Romeo?"}],
                "structured_outputs": {"choice": ["Juliet"]},
                "guided_grammar": "root ::= \\"Juliet\\""
            }
            """));
        do {
            try request.validate();
            Issue.record("both extra-body fields must fail closed");
        } catch let validationError as OpenAiChatCompletionValidationError {
            #expect(validationError.errorDescription?.contains("only one") == true,
                "unexpected error: \(String(describing: validationError.errorDescription))");
        }
    }

    @Test
    func should_accept_structured_outputs_json_object() throws {
        let request: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
            {
                "model": "mlx-community/Qwen3.5-2B-4bit",
                "messages": [{"role": "user", "content": "O Romeo, Romeo, wherefore art thou Romeo?"}],
                "structured_outputs": {"json": {}}
            }
            """));
        let requestParts: OpenAiChatCompletionRequestParts = try request.intoParts();
        #expect(requestParts.enforcedStructuredGeneration == .jsonObject);
    }
}
