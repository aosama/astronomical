import XCTest;
import RestContract;
import IpcProtocol;

/// Shared JSON-text fixture loader for the RestContract test target.
enum RestContractTestFixture {

    static func wireValue(_ jsonText: String) throws -> JsonWireValue {
        return try JsonWireParser.parseDocument(documentBytes: Data(jsonText.utf8));
    }
}

/// Ported from crates/rest-contract/tests/rest_api/openai_response_format.rs.
final class ResponseFormatTests: XCTestCase {

    func testShouldAcceptJsonObjectResponseFormatOnChatCompletions() throws {
        let request: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
            {
                "model": "mlx-community/Qwen3.5-2B-4bit",
                "messages": [{"role": "user", "content": "O Romeo, Romeo, wherefore art thou Romeo?"}],
                "response_format": {"type": "json_object"}
            }
            """));
        let requestParts: OpenAiChatCompletionRequestParts = try request.intoParts();
        XCTAssertEqual(requestParts.structuredOutput, .jsonObject);
    }

    func testShouldAcceptJsonSchemaResponseFormatOnChatCompletions() throws {
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
            XCTFail("expected json_schema, got \(String(describing: requestParts.structuredOutput))");
            return;
        }
        XCTAssertEqual(schemaName, "romeo_line");
        XCTAssertTrue(strict);
    }

    func testShouldRejectAnUnsupportedResponseFormatType() throws {
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
            XCTFail("unsupported response_format types must fail before worker admission");
            return;
        } catch let validationError as OpenAiChatCompletionValidationError {
            XCTAssertEqual(
                validationError.errorDescription,
                OpenAiStructuredOutputValidationError.unsupportedType(formatType: "xml").errorDescription);
        }
    }

    func testShouldAcceptResponsesTextFormatJsonSchema() throws {
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
            XCTFail("expected json_schema from Responses text.format");
            return;
        }
    }

    func testShouldExtractJsonFromFencedModelTextWithoutFillingFields() throws {
        let extractedJson: JsonWireValue = try XCTUnwrap(
            OpenAiStructuredOutput.extract_json_value_from_text(
                "Juliet says:\n```json\n{\"speaker\":\"Juliet\",\"play\":\"Romeo and Juliet\"}\n```\n"));
        XCTAssertEqual(
            extractedJson,
            try RestContractTestFixture.wireValue(
                #"{"speaker": "Juliet", "play": "Romeo and Juliet"}"#));
        XCTAssertNil(OpenAiStructuredOutput.compact_extracted_json_text("not json at all"));
    }

    func testShouldNameUnenforcedGrammarInTheWarningHeader() {
        XCTAssertTrue(ResponseFormatConstants.UNENFORCED_RESPONSE_FORMAT_WARNING
            .contains("grammar-constrained decoding unavailable"));
    }

    func testShouldAcceptStructuredOutputsChoice() throws {
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
            XCTFail("expected a choice-enforced generation");
            return;
        }
    }

    func testShouldEnforceStructuredOutputsRegex() throws {
        let request: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
            {
                "model": "mlx-community/Qwen3.5-2B-4bit",
                "messages": [{"role": "user", "content": "O Romeo, Romeo, wherefore art thou Romeo?"}],
                "structured_outputs": {"regex": "[A-Z]+"}
            }
            """));
        let requestParts: OpenAiChatCompletionRequestParts = try request.intoParts();
        XCTAssertEqual(requestParts.enforcedStructuredGeneration, .regex(pattern: "[A-Z]+"));
    }

    func testShouldRejectAnUncompilableRegexPattern() throws {
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
            XCTFail("an uncompilable regex must fail closed");
            return;
        } catch let validationError as OpenAiChatCompletionValidationError {
            XCTAssertTrue(validationError.errorDescription?.contains("regex") == true,
                "unexpected error: \(String(describing: validationError.errorDescription))");
        }
    }

    func testShouldRejectAnOversizedRegexPattern() throws {
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
            XCTFail("an oversized regex pattern must fail closed");
            return;
        } catch let validationError as OpenAiChatCompletionValidationError {
            XCTAssertTrue(validationError.errorDescription?.contains("bounded") == true,
                "unexpected error: \(String(describing: validationError.errorDescription))");
        }
    }

    func testShouldRejectGuidedGrammarUntilEnforced() throws {
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
            XCTFail("unenforced guided_grammar must fail closed");
            return;
        } catch let validationError as OpenAiChatCompletionValidationError {
            XCTAssertTrue(validationError.errorDescription?.contains("guided_grammar") == true,
                "unexpected error: \(String(describing: validationError.errorDescription))");
        }
    }

    func testShouldRejectStructuredOutputsAndGuidedGrammarTogether() throws {
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
            XCTFail("both extra-body fields must fail closed");
            return;
        } catch let validationError as OpenAiChatCompletionValidationError {
            XCTAssertTrue(validationError.errorDescription?.contains("only one") == true,
                "unexpected error: \(String(describing: validationError.errorDescription))");
        }
    }

    func testShouldAcceptStructuredOutputsJsonObject() throws {
        let request: OpenAiChatCompletionRequest = try OpenAiChatCompletionRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
            {
                "model": "mlx-community/Qwen3.5-2B-4bit",
                "messages": [{"role": "user", "content": "O Romeo, Romeo, wherefore art thou Romeo?"}],
                "structured_outputs": {"json": {}}
            }
            """));
        let requestParts: OpenAiChatCompletionRequestParts = try request.intoParts();
        XCTAssertEqual(requestParts.enforcedStructuredGeneration, .jsonObject);
    }
}
