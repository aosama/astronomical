import Foundation;
import RestContract;
import IpcProtocol;
import Testing;
import JourneyCategories;

/// Ported from crates/rest-contract/tests/rest_api/openai_responses_request.rs.
@Suite(.tags(.hermeticJourney))
final class ResponsesRequestTests {

    @Test
    func should_parse_a_non_streaming_response_request_with_string_input() throws {
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
        #expect(requestParts.model == "astronomical/fake-mixture-of-experts");
        guard case .text(let inputText) = requestParts.input else {
            Issue.record("expected string input");
            return;
        }
        #expect(inputText == "Explain the repository.");
        #expect(requestParts.maximumOutputTokens == 512);
        #expect(requestParts.requestedMaximumOutputTokens == 512);
        #expect(requestParts.temperature == 0.6);
        #expect(requestParts.topP == 0.95);
        #expect(requestParts.stream == false);
    }

    @Test
    func should_preserve_omitted_responses_generation_settings_and_request_only_metadata() throws {
        let request: OpenAiResponsesRequest = try OpenAiResponsesRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue(
                #"{"model":"organization/model","input":"hello"}"#));
        let requestParts: OpenAiResponsesRequestParts = try request.intoParts();
        let responseConfiguration: OpenAiResponseRequestConfiguration = requestParts.responseConfiguration();
        #expect(requestParts.requestedMaximumOutputTokens == nil);
        #expect(requestParts.temperature == nil);
        #expect(requestParts.topP == nil);
        #expect(responseConfiguration.maxOutputTokens == nil);
        #expect(responseConfiguration.temperature == nil);
        #expect(responseConfiguration.topP == nil);
    }

    @Test
    func should_preserve_ordered_response_items_for_manual_function_loop_replay() throws {
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
        #expect(requestParts.instructions == "Be precise.");
        guard case .items(let responseInputItems) = requestParts.input else {
            Issue.record("expected ordered response input items");
            return;
        }
        #expect(responseInputItems.count == 5);
        #expect(responseInputItems[0].kindName() == "user_message");
        #expect(responseInputItems[1].kindName() == "reasoning");
        #expect(responseInputItems[2].kindName() == "assistant_message");
        #expect(responseInputItems[3].kindName() == "function_call");
        #expect(responseInputItems[4].kindName() == "function_call_output");
    }

    @Test
    func should_decode_a_responses_data_uri_image_in_user_content_order() throws {
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
            Issue.record("expected response input items");
            return;
        }
        guard case .userMessage(let content, let images) = responseInputItems[0] else {
            Issue.record("expected a user message");
            return;
        }
        #expect(content == "Describe this image.");
        #expect(images.count == 1);
        #expect(images[0].mimeType() == "image/png");
    }

    @Test
    func should_accept_native_function_tools_and_harmless_compatibility_fields() throws {
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
        #expect(requestParts.tools.count == 1);
        #expect(requestParts.tools[0].name == "glob");
        #expect(requestParts.toolChoice.kindName() == "none");
        #expect(
            requestParts.metadata.first(where: { (entry: OpenAiMetadataEntry) -> Bool in entry.metadataKey == "session" })?.metadataValue
                == "local");
    }

    @Test
    func should_parse_recognized_behavior_changing_fields_before_rejecting_them() throws {
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
                Issue.record("behavior-changing unsupported fields must be rejected");
            } catch let validationError as OpenAiResponsesValidationError {
                #expect(
                    validationError.errorDescription?.contains(unsupportedRequest.expectedOptionName) == true,
                    "expected \(unsupportedRequest.expectedOptionName) in \(String(describing: validationError.errorDescription))");
            }
        }
    }

    @Test
    func should_parse_a_foreign_response_item_before_rejecting_its_type() throws {
        let request: OpenAiResponsesRequest = try OpenAiResponsesRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
        {
            "model":"ornith",
            "input":[{"type":"file_search_call","id":"fs_1","queries":["docs"]}]
        }
        """));
        do {
            _ = try request.intoParts();
            Issue.record("hosted file-search items are not locally executable");
        } catch let validationError as OpenAiResponsesValidationError {
            #expect(validationError.errorDescription?.contains("file_search_call") == true);
        }
    }

    @Test
    func should_accept_the_include_field_as_a_harmless_noop() throws {
        let request: OpenAiResponsesRequest = try OpenAiResponsesRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
        {
            "model":"ornith",
            "input":"hello",
            "include":["reasoning.encrypted_content"]
        }
        """));
        let requestParts: OpenAiResponsesRequestParts = try request.intoParts();
        #expect(requestParts.model == "ornith");
    }

    @Test
    func should_reject_responses_guided_grammar_until_enforced() throws {
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
            Issue.record("unenforced guided_grammar must fail closed on Responses");
        } catch let validationError as OpenAiResponsesValidationError {
            #expect(validationError.errorDescription?.contains("guided_grammar") == true);
        }
    }

    @Test
    func should_resolve_copilot_reasoning_effort_into_the_thinking_budget() throws {
        let request: OpenAiResponsesRequest = try OpenAiResponsesRequest.decoded(
            wireValue: try RestContractTestFixture.wireValue("""
        {
            "model":"ornith",
            "input":"hello",
            "reasoning":{"effort":"medium"}
        }
        """));
        let requestParts: OpenAiResponsesRequestParts = try request.intoParts();
        #expect(requestParts.model == "ornith");
        #expect(requestParts.thinkingBudget == 8192);
    }
}
