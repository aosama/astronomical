import XCTest;
import RestContract;
import IpcProtocol;

/// Ported from crates/rest-contract/tests/rest_api/openai_models_and_errors.rs.
final class ModelsAndErrorsTests: XCTestCase {

    func testShouldSerializeCompleteReadyModelCapabilityMetadataWithoutLosingOpenaiFields() throws {
        let modelList: OpenAiModelList = try OpenAiModelList.singleModel(
            modelParts: OpenAiModelParts(
                modelId: "astronomical/fake-mixture-of-experts",
                created: 1_784_231_803,
                ownedBy: "astronomical",
                contextWindow: 262_144,
                maxInputTokens: 241_664,
                maxOutputTokens: 20_480,
                inputModalities: ["text", "image"],
                outputModalities: ["text"],
                supportsStreaming: true,
                supportsReasoning: true,
                reasoningFormat: "openai_chat_reasoning_content_and_responses_reasoning_summary_text",
                supportsToolCalls: true,
                toolCallFormat: "openai_function_call",
                supportedEndpoints: ["/v1/chat/completions", "/v1/responses"],
                supportsStructuredOutputs: true,
                structuredOutputEnforcement: "logits_mask"));
        XCTAssertEqual(
            try modelList.wireValue().serializedText,
            #"{"object":"list","data":[{"id":"astronomical/fake-mixture-of-experts","object":"model","created":1784231803,"owned_by":"astronomical","context_window":262144,"max_input_tokens":241664,"max_output_tokens":20480,"input_modalities":["text","image"],"output_modalities":["text"],"supports_streaming":true,"supports_reasoning":true,"reasoning_format":"openai_chat_reasoning_content_and_responses_reasoning_summary_text","supports_tool_calls":true,"tool_call_format":"openai_function_call","supported_endpoints":["/v1/chat/completions","/v1/responses"],"supports_structured_outputs":true,"structured_output_enforcement":"logits_mask"}]}"#);
    }

    func testShouldOmitReasoningAndToolFormatsWhenModelDoesNotSupportThem() throws {
        let modelList: OpenAiModelList = try OpenAiModelList.singleModel(
            modelParts: OpenAiModelParts(
                modelId: "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit",
                created: 1_784_231_803,
                ownedBy: "astronomical",
                contextWindow: 131_072,
                maxInputTokens: 110_592,
                maxOutputTokens: 20_480,
                inputModalities: ["text"],
                outputModalities: ["text"],
                supportsStreaming: true,
                supportsReasoning: false,
                reasoningFormat: nil,
                supportsToolCalls: false,
                toolCallFormat: nil,
                supportedEndpoints: ["/v1/chat/completions", "/v1/responses"],
                supportsStructuredOutputs: true,
                structuredOutputEnforcement: "logits_mask"));
        XCTAssertEqual(
            try modelList.wireValue().serializedText,
            #"{"object":"list","data":[{"id":"mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit","object":"model","created":1784231803,"owned_by":"astronomical","context_window":131072,"max_input_tokens":110592,"max_output_tokens":20480,"input_modalities":["text"],"output_modalities":["text"],"supports_streaming":true,"supports_reasoning":false,"supports_tool_calls":false,"supported_endpoints":["/v1/chat/completions","/v1/responses"],"supports_structured_outputs":true,"structured_output_enforcement":"logits_mask"}]}"#);
    }

    func testShouldAcceptIndependentInputAndOutputMaximaThatShareOneContextWindow() {
        do {
            _ = try OpenAiModelList.singleModel(
                modelParts: OpenAiModelParts(
                    modelId: "astronomical/independent-token-limit-model",
                    created: 1_784_231_803,
                    ownedBy: "astronomical",
                    contextWindow: 10,
                    maxInputTokens: 9,
                    maxOutputTokens: 9,
                    inputModalities: ["text"],
                    outputModalities: ["text"],
                    supportsStreaming: true,
                    supportsReasoning: false,
                    reasoningFormat: nil,
                    supportsToolCalls: false,
                    toolCallFormat: nil,
                    supportedEndpoints: ["/v1/chat/completions"],
                    supportsStructuredOutputs: true,
                    structuredOutputEnforcement: "logits_mask"));
        } catch {
            XCTFail("independent maxima must remain valid because request admission enforces the shared context: \(error)");
        }
    }

    func testShouldRejectAnIndependentMaximumThatLeavesNoPositionForTheOtherSide() {
        do {
            _ = try OpenAiModelList.singleModel(
                modelParts: OpenAiModelParts(
                    modelId: "astronomical/exhausted-token-limit-model",
                    created: 1_784_231_803,
                    ownedBy: "astronomical",
                    contextWindow: 10,
                    maxInputTokens: 10,
                    maxOutputTokens: 1,
                    inputModalities: ["text"],
                    outputModalities: ["text"],
                    supportsStreaming: true,
                    supportsReasoning: false,
                    reasoningFormat: nil,
                    supportsToolCalls: false,
                    toolCallFormat: nil,
                    supportedEndpoints: ["/v1/chat/completions"],
                    supportsStructuredOutputs: true,
                    structuredOutputEnforcement: "logits_mask"));
            XCTFail("each independent maximum must leave one context position for the other side");
            return;
        } catch let validationError as OpenAiModelValidationError {
            XCTAssertEqual(
                validationError,
                .inputTokenBudgetMustLeaveGenerationPosition(maxInputTokens: 10, contextWindow: 10));
        } catch {
            XCTFail("unexpected error type: \(error)");
        }
    }

    func testShouldRejectAnOutputMaximumThatLeavesNoPromptPosition() {
        do {
            _ = try OpenAiModelList.singleModel(
                modelParts: OpenAiModelParts(
                    modelId: "astronomical/exhausted-output-limit-model",
                    created: 1_784_231_803,
                    ownedBy: "astronomical",
                    contextWindow: 10,
                    maxInputTokens: 1,
                    maxOutputTokens: 10,
                    inputModalities: ["text"],
                    outputModalities: ["text"],
                    supportsStreaming: true,
                    supportsReasoning: false,
                    reasoningFormat: nil,
                    supportsToolCalls: false,
                    toolCallFormat: nil,
                    supportedEndpoints: ["/v1/chat/completions"],
                    supportsStructuredOutputs: true,
                    structuredOutputEnforcement: "logits_mask"));
            XCTFail("the output maximum must leave one context position for prompt input");
            return;
        } catch let validationError as OpenAiModelValidationError {
            XCTAssertEqual(
                validationError,
                .outputTokenBudgetMustLeavePromptPosition(maxOutputTokens: 10, contextWindow: 10));
        } catch {
            XCTFail("unexpected error type: \(error)");
        }
    }

    func testShouldAdvertiseAnImageOutputModelWithoutFakeTokenLimits() throws {
        let imageModel: OpenAiModel = try OpenAiModel.fromImageParts(
            imageModelParts: OpenAiImageModelParts(
                modelId: "black-forest-labs/FLUX.2-klein-4B",
                created: 1_787_010_400,
                ownedBy: "astronomical",
                inputModalities: ["text"],
                outputModalities: ["image"],
                supportedEndpoints: ["/v1/images/generations"]));
        let modelList: OpenAiModelList = OpenAiModelList.fromModels(models: [imageModel]);
        XCTAssertEqual(
            try modelList.wireValue().serializedText,
            #"{"object":"list","data":[{"id":"black-forest-labs/FLUX.2-klein-4B","object":"model","created":1787010400,"owned_by":"astronomical","input_modalities":["text"],"output_modalities":["image"],"supports_streaming":false,"supports_reasoning":false,"supports_tool_calls":false,"supported_endpoints":["/v1/images/generations"],"supports_structured_outputs":false}]}"#);
    }

    func testShouldOmitCapabilityFieldsWhenNoWorkerIsReady() throws {
        let modelList: OpenAiModelList = OpenAiModelList.empty();
        let serializedModelList: String = try modelList.wireValue().serializedText;
        XCTAssertEqual(serializedModelList, #"{"object":"list","data":[]}"#);
        XCTAssertFalse(serializedModelList.contains("context_window"));
        XCTAssertFalse(serializedModelList.contains("max_input_tokens"));
        XCTAssertFalse(serializedModelList.contains("max_output_tokens"));
        XCTAssertFalse(serializedModelList.contains("supports_reasoning"));
        XCTAssertFalse(serializedModelList.contains("supports_tool_calls"));
        XCTAssertFalse(serializedModelList.contains("supports_streaming"));
        XCTAssertFalse(serializedModelList.contains("input_modalities"));
        XCTAssertFalse(serializedModelList.contains("output_modalities"));
        XCTAssertFalse(serializedModelList.contains("reasoning_format"));
        XCTAssertFalse(serializedModelList.contains("tool_call_format"));
        XCTAssertFalse(serializedModelList.contains("supported_endpoints"));
    }

    func testShouldSerializeAnEmptyOpenaiModelListWhenNoWorkerIsReady() throws {
        let modelList: OpenAiModelList = OpenAiModelList.empty();
        XCTAssertEqual(
            try modelList.wireValue().serializedText,
            #"{"object":"list","data":[]}"#);
    }

    func testShouldSerializeAStandardOpenaiInvalidRequestError() throws {
        let errorResponse: OpenAiErrorResponse = OpenAiErrorResponse.invalidRequest(
            message: "model is not loaded by the local worker",
            parameter: "model",
            code: "model_not_found");
        XCTAssertEqual(
            try errorResponse.wireValue().serializedText,
            #"{"error":{"message":"model is not loaded by the local worker","type":"invalid_request_error","param":"model","code":"model_not_found"}}"#);
    }
}
