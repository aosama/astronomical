import Foundation;
import RestContract;
import IpcProtocol;
import Testing;
import JourneyCategories;

/// Ported from crates/rest-contract/tests/rest_api/openai_models_and_errors.rs.
@Suite(.tags(.hermeticJourney))
final class ModelsAndErrorsTests {

    @Test
    func should_serialize_complete_ready_model_capability_metadata_without_losing_openai_fields() throws {
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
        #expect(
            try modelList.wireValue().serializedText
                == #"{"object":"list","data":[{"id":"astronomical/fake-mixture-of-experts","object":"model","created":1784231803,"owned_by":"astronomical","context_window":262144,"max_input_tokens":241664,"max_output_tokens":20480,"input_modalities":["text","image"],"output_modalities":["text"],"supports_streaming":true,"supports_reasoning":true,"reasoning_format":"openai_chat_reasoning_content_and_responses_reasoning_summary_text","supports_tool_calls":true,"tool_call_format":"openai_function_call","supported_endpoints":["/v1/chat/completions","/v1/responses"],"supports_structured_outputs":true,"structured_output_enforcement":"logits_mask"}]}"#);
    }

    @Test
    func should_omit_reasoning_and_tool_formats_when_model_does_not_support_them() throws {
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
        #expect(
            try modelList.wireValue().serializedText
                == #"{"object":"list","data":[{"id":"mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit","object":"model","created":1784231803,"owned_by":"astronomical","context_window":131072,"max_input_tokens":110592,"max_output_tokens":20480,"input_modalities":["text"],"output_modalities":["text"],"supports_streaming":true,"supports_reasoning":false,"supports_tool_calls":false,"supported_endpoints":["/v1/chat/completions","/v1/responses"],"supports_structured_outputs":true,"structured_output_enforcement":"logits_mask"}]}"#);
    }

    @Test
    func should_accept_independent_input_and_output_maxima_that_share_one_context_window() throws {
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
    }

    @Test
    func should_reject_an_independent_maximum_that_leaves_no_position_for_the_other_side() throws {
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
            Issue.record("each independent maximum must leave one context position for the other side");
        } catch let validationError as OpenAiModelValidationError {
            #expect(
                validationError
                    == OpenAiModelValidationError.inputTokenBudgetMustLeaveGenerationPosition(
                        maxInputTokens: 10, contextWindow: 10));
        }
    }

    @Test
    func should_reject_an_output_maximum_that_leaves_no_prompt_position() throws {
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
            Issue.record("the output maximum must leave one context position for prompt input");
        } catch let validationError as OpenAiModelValidationError {
            #expect(
                validationError
                    == OpenAiModelValidationError.outputTokenBudgetMustLeavePromptPosition(
                        maxOutputTokens: 10, contextWindow: 10));
        }
    }

    @Test
    func should_advertise_an_image_output_model_without_fake_token_limits() throws {
        let imageModel: OpenAiModel = try OpenAiModel.fromImageParts(
            imageModelParts: OpenAiImageModelParts(
                modelId: "black-forest-labs/FLUX.2-klein-4B",
                created: 1_787_010_400,
                ownedBy: "astronomical",
                inputModalities: ["text"],
                outputModalities: ["image"],
                supportedEndpoints: ["/v1/images/generations"]));
        let modelList: OpenAiModelList = OpenAiModelList.fromModels(models: [imageModel]);
        #expect(
            try modelList.wireValue().serializedText
                == #"{"object":"list","data":[{"id":"black-forest-labs/FLUX.2-klein-4B","object":"model","created":1787010400,"owned_by":"astronomical","input_modalities":["text"],"output_modalities":["image"],"supports_streaming":false,"supports_reasoning":false,"supports_tool_calls":false,"supported_endpoints":["/v1/images/generations"],"supports_structured_outputs":false}]}"#);
    }

    @Test
    func should_omit_capability_fields_when_no_worker_is_ready() throws {
        let modelList: OpenAiModelList = OpenAiModelList.empty();
        let serializedModelList: String = try modelList.wireValue().serializedText;
        #expect(serializedModelList == #"{"object":"list","data":[]}"#);
        #expect(serializedModelList.contains("context_window") == false);
        #expect(serializedModelList.contains("max_input_tokens") == false);
        #expect(serializedModelList.contains("max_output_tokens") == false);
        #expect(serializedModelList.contains("supports_reasoning") == false);
        #expect(serializedModelList.contains("supports_tool_calls") == false);
        #expect(serializedModelList.contains("supports_streaming") == false);
        #expect(serializedModelList.contains("input_modalities") == false);
        #expect(serializedModelList.contains("output_modalities") == false);
        #expect(serializedModelList.contains("reasoning_format") == false);
        #expect(serializedModelList.contains("tool_call_format") == false);
        #expect(serializedModelList.contains("supported_endpoints") == false);
    }

    @Test
    func should_serialize_an_empty_openai_model_list_when_no_worker_is_ready() throws {
        let modelList: OpenAiModelList = OpenAiModelList.empty();
        #expect(
            try modelList.wireValue().serializedText
                == #"{"object":"list","data":[]}"#);
    }

    @Test
    func should_serialize_a_standard_openai_invalid_request_error() throws {
        let errorResponse: OpenAiErrorResponse = OpenAiErrorResponse.invalidRequest(
            message: "model is not loaded by the local worker",
            parameter: "model",
            code: "model_not_found");
        #expect(
            try errorResponse.wireValue().serializedText
                == #"{"error":{"message":"model is not loaded by the local worker","type":"invalid_request_error","param":"model","code":"model_not_found"}}"#);
    }
}
