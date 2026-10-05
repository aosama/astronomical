import XCTest;
import RestContract;
import IpcProtocol;

/// Ported from crates/rest-contract/tests/rest_api/openai_responses_request.rs.
final class ResponsesRequestTests: XCTestCase {

    func testShouldParseANonStreamingResponseRequestWithStringInput() throws {
        let request: OpenAiResponsesRequest = try OpenAiResponsesRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
        {
            "model": "astronomical/fake-mixture-of-experts",
            "input": "Explain the repository.",
            "max_output_tokens": 512,
            "temperature": 0.6,
            "top_p": 0.95
        }
        """));
        let requestParts: OpenAiResponsesRequestParts = try request.intoParts();
        XCTAssertEqual(requestParts.model, "astronomical/fake-mixture-of-experts");
        guard case .text(let inputText) = requestParts.input else {
            XCTFail("expected string input");
            return;
        }
        XCTAssertEqual(inputText, "Explain the repository.");
        XCTAssertEqual(requestParts.maximumOutputTokens, 512);
        XCTAssertEqual(requestParts.requestedMaximumOutputTokens, 512);
        XCTAssertEqual(requestParts.temperature, 0.6);
        XCTAssertEqual(requestParts.topP, 0.95);
        XCTAssertFalse(requestParts.stream);
    }

    func testShouldPreserveOmittedResponsesGenerationSettingsAndRequestOnlyMetadata() throws {
        let request: OpenAiResponsesRequest = try OpenAiResponsesRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue(
                #"{"model":"organization/model","input":"hello"}"#));
        let requestParts: OpenAiResponsesRequestParts = try request.intoParts();
        let responseConfiguration: OpenAiResponseRequestConfiguration = requestParts.responseConfiguration();
        XCTAssertNil(requestParts.requestedMaximumOutputTokens);
        XCTAssertNil(requestParts.temperature);
        XCTAssertNil(requestParts.topP);
        XCTAssertNil(responseConfiguration.maxOutputTokens);
        XCTAssertNil(responseConfiguration.temperature);
        XCTAssertNil(responseConfiguration.topP);
    }

    func testShouldPreserveOrderedResponseItemsForManualFunctionLoopReplay() throws {
        let request: OpenAiResponsesRequest = try OpenAiResponsesRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
        {
            "model": "astronomical/fake-mixture-of-experts",
            "instructions": "Be precise.",
            "input": [
                {"role":"user","content":[{"type":"input_text","text":"Inspect the repository."}]},
                {"type":"reasoning","id":"rs_prior","summary":[],"content":[{"type":"reasoning_text","text":"I should inspect files."}]},
                {"type":"message","id":"msg_prior","role":"assistant","status":"completed","content":[{"type":"output_text","text":"I will inspect it.","annotations":[],"logprobs":[]}]},
                {"type":"function_call","id":"fc_prior","call_id":"call_prior","name":"glob","arguments":"{\\"pattern\\":\\"**/*.rs\\"}","status":"completed"},
                {"type":"function_call_output","call_id":"call_prior","output":"src/lib.rs"}
            ]
        }
        """));
        let requestParts: OpenAiResponsesRequestParts = try request.intoParts();
        XCTAssertEqual(requestParts.instructions, "Be precise.");
        guard case .items(let responseInputItems) = requestParts.input else {
            XCTFail("expected ordered response input items");
            return;
        }
        XCTAssertEqual(responseInputItems.count, 5);
        XCTAssertEqual(responseInputItems[0].kindName(), "user_message");
        XCTAssertEqual(responseInputItems[1].kindName(), "reasoning");
        XCTAssertEqual(responseInputItems[2].kindName(), "assistant_message");
        XCTAssertEqual(responseInputItems[3].kindName(), "function_call");
        XCTAssertEqual(responseInputItems[4].kindName(), "function_call_output");
    }

    func testShouldDecodeAResponsesDataUriImageInUserContentOrder() throws {
        let redPixelPngBase64: String = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg==";
        let requestJson: String = """
        {
            "model":"astronomical/fake-mixture-of-experts",
            "input":[{
                "role":"user",
                "content":[
                    {"type":"input_text","text":"Describe this image."},
                    {"type":"input_image","image_url":"data:image/png;base64,\(redPixelPngBase64)","detail":"auto"}
                ]
            }]
        }
        """;
        let request: OpenAiResponsesRequest = try OpenAiResponsesRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue(requestJson));
        let requestParts: OpenAiResponsesRequestParts = try request.intoParts();
        guard case .items(let responseInputItems) = requestParts.input else {
            XCTFail("expected response input items");
            return;
        }
        guard case .userMessage(let content, let images) = responseInputItems[0] else {
            XCTFail("expected a user message");
            return;
        }
        XCTAssertEqual(content, "Describe this image.");
        XCTAssertEqual(images.count, 1);
        XCTAssertEqual(images[0].mimeType(), "image/png");
    }

    func testShouldAcceptNativeFunctionToolsAndHarmlessCompatibilityFields() throws {
        let request: OpenAiResponsesRequest = try OpenAiResponsesRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
        {
            "model":"astronomical/fake-mixture-of-experts",
            "input":"List Rust files.",
            "tools":[{
                "type":"function",
                "name":"glob",
                "description":"List matching files.",
                "parameters":{"type":"object","properties":{"pattern":{"type":"string"}}},
                "strict":false
            }],
            "tool_choice":"none",
            "metadata":{"session":"local"},
            "store":false,
            "background":false,
            "truncation":"disabled",
            "service_tier":"auto",
            "user":"single-user",
            "safety_identifier":"local-user",
            "prompt_cache_key":"session-prefix"
        }
        """));
        let requestParts: OpenAiResponsesRequestParts = try request.intoParts();
        XCTAssertEqual(requestParts.tools.count, 1);
        XCTAssertEqual(requestParts.tools[0].name, "glob");
        XCTAssertEqual(requestParts.toolChoice.kindName(), "none");
        XCTAssertEqual(
            requestParts.metadata.first(where: { (entry: OpenAiMetadataEntry) -> Bool in entry.metadataKey == "session" })?.metadataValue,
            "local");
    }

    func testShouldParseRecognizedBehaviorChangingFieldsBeforeRejectingThem() throws {
        let unsupportedRequests: Array<(requestJson: String, expectedOptionName: String)> = [
            (
                #"{"model":"ornith","input":"hello","previous_response_id":"resp_prior"}"#,
                "previous_response_id"
            ),
            (
                #"{"model":"ornith","input":"hello","text":{"verbosity":"high"}}"#,
                "text.verbosity"
            ),
            (
                #"{"model":"ornith","input":"hello","tools":[{"type":"web_search"}]}"#,
                "tools[].type"
            ),
        ];
        for unsupportedRequest in unsupportedRequests {
            let request: OpenAiResponsesRequest = try OpenAiResponsesRequest.decoded(
                wireValue: try RestContractTestFixture.wireValue(unsupportedRequest.requestJson));
            do {
                _ = try request.intoParts();
                XCTFail("behavior-changing unsupported fields must be rejected");
                return;
            } catch let validationError as OpenAiResponsesValidationError {
                XCTAssertTrue(
                    validationError.errorDescription?.contains(unsupportedRequest.expectedOptionName) == true,
                    "expected \(unsupportedRequest.expectedOptionName) in \(String(describing: validationError.errorDescription))");
            }
        }
    }

    func testShouldParseAForeignResponseItemBeforeRejectingItsType() throws {
        let request: OpenAiResponsesRequest = try OpenAiResponsesRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
        {
            "model":"ornith",
            "input":[{"type":"file_search_call","id":"fs_1","queries":["docs"]}]
        }
        """));
        do {
            _ = try request.intoParts();
            XCTFail("hosted file-search items are not locally executable");
            return;
        } catch let validationError as OpenAiResponsesValidationError {
            XCTAssertTrue(validationError.errorDescription?.contains("file_search_call") == true);
        }
    }

    func testShouldAcceptTheIncludeFieldAsAHarmlessNoop() throws {
        let request: OpenAiResponsesRequest = try OpenAiResponsesRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
        {
            "model":"ornith",
            "input":"hello",
            "include":["reasoning.encrypted_content"]
        }
        """));
        let requestParts: OpenAiResponsesRequestParts = try request.intoParts();
        XCTAssertEqual(requestParts.model, "ornith");
    }

    func testShouldRejectResponsesGuidedGrammarUntilEnforced() throws {
        let request: OpenAiResponsesRequest = try OpenAiResponsesRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
        {
            "model":"mlx-community/Qwen3.5-2B-4bit",
            "input":"O Romeo, Romeo, wherefore art thou Romeo?",
            "guided_grammar":"root ::= \\"Juliet\\" | \\"Romeo\\""
        }
        """));
        do {
            _ = try request.intoParts();
            XCTFail("unenforced guided_grammar must fail closed on Responses");
            return;
        } catch let validationError as OpenAiResponsesValidationError {
            XCTAssertTrue(validationError.errorDescription?.contains("guided_grammar") == true);
        }
    }

    func testShouldResolveCopilotReasoningEffortIntoTheThinkingBudget() throws {
        let request: OpenAiResponsesRequest = try OpenAiResponsesRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
        {
            "model":"ornith",
            "input":"hello",
            "reasoning":{"effort":"medium"}
        }
        """));
        let requestParts: OpenAiResponsesRequestParts = try request.intoParts();
        XCTAssertEqual(requestParts.model, "ornith");
        XCTAssertEqual(requestParts.thinkingBudget, 8192);
    }
}
