import Foundation;

import Testing;

import AstronomicalConfig;
import IpcProtocol;
import RestContract;

@testable import Supervisor;

/**
 * Acceptance journeys for the non-streaming POST /v1/chat/completions
 * surface: a scripted executor's events assemble into one OpenAI-compatible
 * JSON body (text, reasoning, tool calls, cached-token usage), every worker
 * failure maps to its stable error envelope, the admission failures map to
 * 429/503/413 with their codes, and a structured-output request answers with
 * extracted JSON plus the unenforced Warning header. The executor is
 * scripted, so no worker process runs.
 */
@Suite(.serialized, .tags(.hermeticJourney))
final class RestChatCompletionEndpointTests {

    // MARK: Non-streaming assembly journeys

    @Test
    func should_return_a_non_streaming_chat_completion_json_response() throws {
        let routeTable: RestRouteTable = try RestChatCompletionEndpointTests.routeTable(
            modelId: RestChatJourneySupport.nonStreamingModelId,
            streamEvents: [
                .textFragment("done"),
                .completed(
                    promptTokenCount: 3,
                    generatedTokenCount: 2,
                    reasoningTokenCount: 0,
                    cachedTokenCount: 0,
                    reason: .endOfSequence),
            ]);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(RestChatJourneySupport.nonStreamingModelId)\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":false}");

        #expect(chatResponse.statusCode == 200);
        #expect(chatResponse.contentType.hasPrefix("application/json"));
        let responseText: String = RestChatJourneySupport.responseText(chatResponse);
        #expect(responseText.contains("data: ") == false,
            "the non-streaming response must not contain SSE frames");
        #expect(responseText.contains("[DONE]") == false,
            "the non-streaming response must not contain the SSE terminator");
        let envelope: [String: Any] = try RestChatJourneySupport.decodeObjectEnvelope(chatResponse);
        #expect(envelope["object"] as? String == "chat.completion");
        let choice: [String: Any] = try RestChatJourneySupport.requireChoice(envelope);
        let assistantMessage: [String: Any] = try RestChatJourneySupport.requireMessage(choice);
        #expect(assistantMessage["content"] as? String == "done");
        #expect(choice["finish_reason"] as? String == "stop");
        let usage: [String: Any] = try RestChatJourneySupport.requireUsage(envelope);
        #expect(usage["prompt_tokens"] as? UInt64 == 3);
        #expect(usage["completion_tokens"] as? UInt64 == 2);
        #expect(usage["total_tokens"] as? UInt64 == 5);
    }

    @Test
    func should_accept_a_chat_request_with_the_stream_field_omitted() throws {
        let routeTable: RestRouteTable = try RestChatCompletionEndpointTests.routeTable(
            modelId: RestChatJourneySupport.nonStreamingModelId,
            streamEvents: [
                .textFragment("done"),
                .completed(
                    promptTokenCount: 3,
                    generatedTokenCount: 2,
                    reasoningTokenCount: 0,
                    cachedTokenCount: 0,
                    reason: .endOfSequence),
            ]);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(RestChatJourneySupport.nonStreamingModelId)\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}");

        #expect(chatResponse.statusCode == 200);
        let envelope: [String: Any] = try RestChatJourneySupport.decodeObjectEnvelope(chatResponse);
        #expect(envelope["object"] as? String == "chat.completion");
    }

    @Test
    func should_assemble_reasoning_and_text_in_a_non_streaming_response() throws {
        let routeTable: RestRouteTable = try RestChatCompletionEndpointTests.routeTable(
            modelId: RestChatJourneySupport.nonStreamingModelId,
            streamEvents: [
                .reasoningFragment("inspect first"),
                .textFragment("done"),
                .completed(
                    promptTokenCount: 3,
                    generatedTokenCount: 2,
                    reasoningTokenCount: 1,
                    cachedTokenCount: 0,
                    reason: .endOfSequence),
            ]);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(RestChatJourneySupport.nonStreamingModelId)\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":false}");

        #expect(chatResponse.statusCode == 200);
        let envelope: [String: Any] = try RestChatJourneySupport.decodeObjectEnvelope(chatResponse);
        let assistantMessage: [String: Any] = try RestChatJourneySupport.requireMessage(
            try RestChatJourneySupport.requireChoice(envelope));
        #expect(assistantMessage["content"] as? String == "done");
        #expect(assistantMessage["reasoning_content"] as? String == "inspect first");
        #expect(assistantMessage["content"] as? String != assistantMessage["reasoning_content"] as? String);
    }

    @Test
    func should_not_fallback_reasoning_only_non_streaming_output_to_message_content() throws {
        let routeTable: RestRouteTable = try RestChatCompletionEndpointTests.routeTable(
            modelId: RestChatJourneySupport.streamingModelId,
            streamEvents: [
                .reasoningFragment("thinking only"),
                .completed(
                    promptTokenCount: 3,
                    generatedTokenCount: 2,
                    reasoningTokenCount: 0,
                    cachedTokenCount: 0,
                    reason: .endOfSequence),
            ]);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(RestChatJourneySupport.streamingModelId)\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":false}");

        #expect(chatResponse.statusCode == 200);
        let envelope: [String: Any] = try RestChatJourneySupport.decodeObjectEnvelope(chatResponse);
        let assistantMessage: [String: Any] = try RestChatJourneySupport.requireMessage(
            try RestChatJourneySupport.requireChoice(envelope));
        #expect(assistantMessage["content"] is NSNull || assistantMessage["content"] == nil,
            "reasoning-only responses must not expose reasoning as visible assistant content");
        #expect(assistantMessage["reasoning_content"] as? String == "thinking only");
    }

    @Test
    func should_assemble_tool_calls_in_a_non_streaming_response() throws {
        let routeTable: RestRouteTable = try RestChatCompletionEndpointTests.routeTable(
            modelId: RestChatJourneySupport.nonStreamingModelId,
            streamEvents: [
                .toolCall(
                    toolCallIndex: 0,
                    functionName: "read",
                    argumentsJson: "{\"filePath\":\"README.md\"}"),
                .completed(
                    promptTokenCount: 1,
                    generatedTokenCount: 1,
                    reasoningTokenCount: 0,
                    cachedTokenCount: 0,
                    reason: .toolCalls),
            ]);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(RestChatJourneySupport.nonStreamingModelId)\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":false}");

        #expect(chatResponse.statusCode == 200);
        let envelope: [String: Any] = try RestChatJourneySupport.decodeObjectEnvelope(chatResponse);
        let choice: [String: Any] = try RestChatJourneySupport.requireChoice(envelope);
        #expect(choice["finish_reason"] as? String == "tool_calls");
        let assistantMessage: [String: Any] = try RestChatJourneySupport.requireMessage(choice);
        let toolCalls: Array<Any> = try #require(assistantMessage["tool_calls"] as? Array<Any>);
        let toolCall: [String: Any] = try #require(toolCalls[0] as? [String: Any]);
        let toolFunction: [String: Any] = try #require(toolCall["function"] as? [String: Any]);
        #expect(toolFunction["name"] as? String == "read");
        #expect(toolFunction["arguments"] as? String == "{\"filePath\":\"README.md\"}");
        let toolCallId: String = try #require(toolCall["id"] as? String);
        #expect(toolCallId.hasPrefix("call_"),
            "the tool call id should follow the streaming-path format");
    }

    @Test
    func should_include_cached_tokens_in_non_streaming_usage_when_nonzero() throws {
        let routeTable: RestRouteTable = try RestChatCompletionEndpointTests.routeTable(
            modelId: RestChatJourneySupport.nonStreamingModelId,
            streamEvents: [
                .completed(
                    promptTokenCount: 4_096,
                    generatedTokenCount: 100,
                    reasoningTokenCount: 0,
                    cachedTokenCount: 2_048,
                    reason: .endOfSequence),
            ]);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(RestChatJourneySupport.nonStreamingModelId)\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":false}");

        #expect(chatResponse.statusCode == 200);
        let envelope: [String: Any] = try RestChatJourneySupport.decodeObjectEnvelope(chatResponse);
        let usage: [String: Any] = try RestChatJourneySupport.requireUsage(envelope);
        let promptTokenDetails: [String: Any] = try #require(
            usage["prompt_tokens_details"] as? [String: Any]);
        #expect(promptTokenDetails["cached_tokens"] as? UInt64 == 2_048);
    }

    @Test
    func should_withhold_reasoning_content_from_non_streaming_responses_when_excluded() throws {
        let routeTable: RestRouteTable = try RestChatCompletionEndpointTests.routeTable(
            modelId: RestChatJourneySupport.streamingModelId,
            streamEvents: [
                .reasoningFragment("internal thought"),
                .textFragment("Visible title"),
                .completed(
                    promptTokenCount: 3,
                    generatedTokenCount: 2,
                    reasoningTokenCount: 0,
                    cachedTokenCount: 0,
                    reason: .endOfSequence),
            ]);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(RestChatJourneySupport.streamingModelId)\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":false,\"reasoning\":{\"effort\":\"low\",\"exclude\":true}}");

        #expect(chatResponse.statusCode == 200);
        let envelope: [String: Any] = try RestChatJourneySupport.decodeObjectEnvelope(chatResponse);
        let assistantMessage: [String: Any] = try RestChatJourneySupport.requireMessage(
            try RestChatJourneySupport.requireChoice(envelope));
        #expect(assistantMessage["reasoning_content"] == nil,
            "excluded reasoning must not be assembled into the non-streaming response");
        #expect(assistantMessage["content"] as? String == "Visible title");
    }

    // MARK: Worker-failure mapping journeys

    @Test
    func should_return_a_service_unavailable_error_body_for_a_failed_non_streaming_chat() throws {
        let routeTable: RestRouteTable = try RestChatCompletionEndpointTests.routeTable(
            modelId: RestChatJourneySupport.nonStreamingModelId,
            streamEvents: [
                .failed(reason: .invalidRequest(
                    reason: "rendered prompt exceeds the 262144-byte worker limit")),
            ]);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(RestChatJourneySupport.nonStreamingModelId)\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":false}");

        #expect(chatResponse.statusCode == 503);
        let errorObject: [String: Any] = try RestChatJourneySupport.requireErrorObject(
            try RestChatJourneySupport.decodeObjectEnvelope(chatResponse));
        #expect(errorObject["code"] as? String == "chat_invalid_request");
        #expect(errorObject["message"] as? String
            == "the local worker rejected the chat request: rendered prompt exceeds the 262144-byte worker limit");
    }

    @Test
    func should_return_a_service_unavailable_error_body_when_the_worker_becomes_unavailable_mid_chat() throws {
        let routeTable: RestRouteTable = try RestChatCompletionEndpointTests.routeTable(
            modelId: RestChatJourneySupport.nonStreamingModelId,
            streamEvents: [.streamError(.workerUnavailable)]);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(RestChatJourneySupport.nonStreamingModelId)\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":false}");

        #expect(chatResponse.statusCode == 503);
        let errorObject: [String: Any] = try RestChatJourneySupport.requireErrorObject(
            try RestChatJourneySupport.decodeObjectEnvelope(chatResponse));
        #expect(errorObject["code"] as? String == "chat_worker_unavailable");
        #expect(errorObject["message"] as? String
            == "the local worker became unavailable while processing the chat request");
    }

    @Test
    func should_report_fatal_worker_execution_as_unavailable_for_a_non_streaming_chat() throws {
        let boundedFatalExecutionReason: String =
            "GPU allocation exceeded the platform buffer limit; reduce the prompt size";
        let routeTable: RestRouteTable = try RestChatCompletionEndpointTests.routeTable(
            modelId: RestChatJourneySupport.nonStreamingModelId,
            streamEvents: [.failed(reason: .fatalExecution(reason: boundedFatalExecutionReason))]);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(RestChatJourneySupport.nonStreamingModelId)\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":false}");

        #expect(chatResponse.statusCode == 503);
        let errorObject: [String: Any] = try RestChatJourneySupport.requireErrorObject(
            try RestChatJourneySupport.decodeObjectEnvelope(chatResponse));
        #expect(errorObject["code"] as? String == "chat_worker_unavailable");
        #expect(errorObject["message"] as? String
            == "the local worker stopped after a fatal model execution error: "
                + boundedFatalExecutionReason);
    }

    @Test
    func should_return_context_length_exceeded_for_a_non_streaming_chat() throws {
        let routeTable: RestRouteTable = try RestChatCompletionEndpointTests.routeTable(
            modelId: RestChatJourneySupport.nonStreamingModelId,
            streamEvents: [.failed(reason: .contextLengthExceeded(
                actualTotalContextTokens: 262_145,
                maximumContextTokens: 262_144))]);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(RestChatJourneySupport.nonStreamingModelId)\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":false}");

        #expect(chatResponse.statusCode == 400);
        let errorObject: [String: Any] = try RestChatJourneySupport.requireErrorObject(
            try RestChatJourneySupport.decodeObjectEnvelope(chatResponse));
        #expect(errorObject["code"] as? String == "context_length_exceeded");
        #expect(errorObject["param"] as? String == "messages");
    }

    // MARK: Admission and gating journeys

    @Test
    func should_reject_openai_chat_model_mismatch_before_worker_admission() throws {
        let routeTable: RestRouteTable = try RestChatCompletionEndpointTests.routeTable(
            modelId: RestChatJourneySupport.negativeModelId,
            streamEvents: []);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            requestBody: "{\"model\":\"astronomical/not-loaded\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":true}");

        #expect(chatResponse.statusCode == 400);
        let errorObject: [String: Any] = try RestChatJourneySupport.requireErrorObject(
            try RestChatJourneySupport.decodeObjectEnvelope(chatResponse));
        #expect(errorObject["param"] as? String == "model");
        #expect(errorObject["code"] as? String == "model_not_found");
    }

    @Test
    func should_return_too_many_requests_when_openai_chat_capacity_is_unavailable() throws {
        let routeTable: RestRouteTable = try RestChatCompletionEndpointTests.routeTable(
            modelId: RestChatJourneySupport.negativeModelId,
            streamEvents: [],
            startError: .capacityUnavailable);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(RestChatJourneySupport.negativeModelId)\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":true}");

        #expect(chatResponse.statusCode == 429);
        let errorObject: [String: Any] = try RestChatJourneySupport.requireErrorObject(
            try RestChatJourneySupport.decodeObjectEnvelope(chatResponse));
        #expect(errorObject["code"] as? String == "server_capacity");
        #expect(errorObject["message"] as? String == "the generation queue is full");
    }

    @Test
    func should_return_service_unavailable_when_openai_chat_worker_is_unavailable() throws {
        let routeTable: RestRouteTable = try RestChatCompletionEndpointTests.routeTable(
            modelId: RestChatJourneySupport.negativeModelId,
            streamEvents: [],
            startError: .workerUnavailable);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(RestChatJourneySupport.negativeModelId)\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":true}");

        #expect(chatResponse.statusCode == 503);
        let errorObject: [String: Any] = try RestChatJourneySupport.requireErrorObject(
            try RestChatJourneySupport.decodeObjectEnvelope(chatResponse));
        #expect(errorObject["code"] as? String == "worker_unavailable");
        #expect(errorObject["message"] as? String == "the local worker is unavailable");
    }

    @Test
    func should_return_payload_too_large_when_openai_chat_exceeds_the_ipc_frame() throws {
        let routeTable: RestRouteTable = try RestChatCompletionEndpointTests.routeTable(
            modelId: RestChatJourneySupport.negativeModelId,
            streamEvents: [],
            startError: .requestTooLarge(
                actualIpcMessageBytes: 33_554_433,
                maximumIpcMessageBytes: 33_554_432));

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(RestChatJourneySupport.negativeModelId)\",\"messages\":[{\"role\":\"user\",\"content\":\"describe the image\"}],\"stream\":true}");

        #expect(chatResponse.statusCode == 413);
        let errorObject: [String: Any] = try RestChatJourneySupport.requireErrorObject(
            try RestChatJourneySupport.decodeObjectEnvelope(chatResponse));
        #expect(errorObject["code"] as? String == "request_too_large");
        let errorMessage: String = try #require(errorObject["message"] as? String);
        #expect(errorMessage.contains("reduce image sizes or conversation history"));
    }

    @Test
    func should_explain_why_the_requested_chat_model_could_not_be_loaded() throws {
        let routeTable: RestRouteTable = try RestChatCompletionEndpointTests.routeTable(
            modelId: RestChatJourneySupport.negativeModelId,
            streamEvents: [],
            startError: .modelLoadFailed(
                modelLoadFailureReason: "OptiQ metadata uses unsupported 2-bit quantization"));

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(RestChatJourneySupport.negativeModelId)\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":true}");

        #expect(chatResponse.statusCode == 503);
        let errorObject: [String: Any] = try RestChatJourneySupport.requireErrorObject(
            try RestChatJourneySupport.decodeObjectEnvelope(chatResponse));
        #expect(errorObject["code"] as? String == "model_load_failed");
        #expect(errorObject["message"] as? String
            == "the requested model could not be loaded: OptiQ metadata uses unsupported 2-bit quantization");
    }

    @Test
    func should_canonicalize_a_provider_prefixed_model_for_an_idle_worker() throws {
        let resolvedConfig: ResolvedRuntimeConfig = try RestChatJourneySupport.makeResolvedConfig(
            discoveredModels: [
                DiscoveryDiscoveredModel(
                    modelId: "requested-model",
                    providerModelId: nil,
                    modelFamily: .modernbert,
                    revision: "0000000000000000000000000000000000000000",
                    modelDirectory: FilePath(string: "/models/requested-model"),
                    capabilities: .chat(DiscoveryChatModelCapabilities(
                        contextWindowTokens: 4_096,
                        maximumInputTokens: 4_095,
                        maximumOutputTokens: 1_024,
                        supportsVision: false,
                        supportsReasoning: false,
                        supportsToolCalls: false)),
                    license: nil,
                    modelSizeBytes: 400_000_000),
            ]);
        let scriptedExecutor: ScriptedChatExecutor = ScriptedChatExecutor(
            healthSnapshot: WorkerHealthSnapshot.readyWithoutModel(
                machineMlxMemoryCeilingBytes: 1,
                effectiveMlxMemoryCeilingBytes: 1,
                minimumMlxMemoryCeilingBytes: 1),
            startError: .capacityUnavailable);
        let routeTable: RestRouteTable = try RestChatJourneySupport.chatRouteTable(
            resolvedRuntimeConfig: resolvedConfig,
            chatExecutor: scriptedExecutor);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            requestBody: "{\"model\":\"mlx-community/requested-model\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":true}");

        // The provider prefix canonicalizes onto the discovered leaf model,
        // so the request reaches admission (and its queue-full rejection).
        #expect(chatResponse.statusCode == 429);
    }

    @Test
    func should_reject_malformed_json_with_the_invalid_json_code() throws {
        let routeTable: RestRouteTable = try RestChatCompletionEndpointTests.routeTable(
            modelId: RestChatJourneySupport.negativeModelId,
            streamEvents: []);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            requestBody: "{\"model\":");

        #expect(chatResponse.statusCode == 400);
        let errorObject: [String: Any] = try RestChatJourneySupport.requireErrorObject(
            try RestChatJourneySupport.decodeObjectEnvelope(chatResponse));
        #expect(errorObject["code"] as? String == "invalid_json");
    }

    // MARK: Structured-output journeys

    @Test
    func should_return_extracted_json_and_a_warning_for_json_schema_chat() throws {
        let routeTable: RestRouteTable = try RestChatCompletionEndpointTests.routeTable(
            modelId: RestChatJourneySupport.nonStreamingModelId,
            streamEvents: [
                .textFragment("Juliet answers:\n```json\n{\"speaker\":\"Juliet\",\"play\":\"Romeo and Juliet\"}\n```"),
                .completed(
                    promptTokenCount: 12,
                    generatedTokenCount: 8,
                    reasoningTokenCount: 0,
                    cachedTokenCount: 0,
                    reason: .endOfSequence),
            ]);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(RestChatJourneySupport.nonStreamingModelId)\",\"messages\":[{\"role\":\"user\",\"content\":\"O Romeo, Romeo, wherefore art thou Romeo?\"}],\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"romeo_line\",\"schema\":{\"type\":\"object\",\"properties\":{\"speaker\":{\"type\":\"string\"},\"play\":{\"type\":\"string\"}},\"required\":[\"speaker\",\"play\"]}}},\"stream\":false}");

        #expect(chatResponse.statusCode == 200);
        let warningHeaderValue: String? = RestChatJourneySupport.responseHeaderValue(
            chatResponse, headerName: "Warning");
        #expect(warningHeaderValue == ResponseFormatConstants.UNENFORCED_RESPONSE_FORMAT_WARNING,
            "unenforced json_schema must disclose a Warning header");
        let envelope: [String: Any] = try RestChatJourneySupport.decodeObjectEnvelope(chatResponse);
        let assistantMessage: [String: Any] = try RestChatJourneySupport.requireMessage(
            try RestChatJourneySupport.requireChoice(envelope));
        let visibleContent: String = try #require(assistantMessage["content"] as? String);
        let extractedJson: [String: Any] = try #require(
            JSONSerialization.jsonObject(with: Data(visibleContent.utf8)) as? [String: Any]);
        #expect(extractedJson["speaker"] as? String == "Juliet");
        #expect(extractedJson["play"] as? String == "Romeo and Juliet");
    }

    @Test
    func should_keep_original_text_when_json_cannot_be_extracted() throws {
        let routeTable: RestRouteTable = try RestChatCompletionEndpointTests.routeTable(
            modelId: RestChatJourneySupport.nonStreamingModelId,
            streamEvents: [
                .textFragment("Two households, both alike in dignity."),
                .completed(
                    promptTokenCount: 8,
                    generatedTokenCount: 8,
                    reasoningTokenCount: 0,
                    cachedTokenCount: 0,
                    reason: .endOfSequence),
            ]);

        let chatResponse: RestHttpResponse = try RestChatJourneySupport.postChat(
            routeTable: routeTable,
            requestBody: "{\"model\":\"\(RestChatJourneySupport.nonStreamingModelId)\",\"messages\":[{\"role\":\"user\",\"content\":\"O Romeo, Romeo, wherefore art thou Romeo?\"}],\"response_format\":{\"type\":\"json_object\"},\"stream\":false}");

        #expect(chatResponse.statusCode == 200);
        let envelope: [String: Any] = try RestChatJourneySupport.decodeObjectEnvelope(chatResponse);
        let assistantMessage: [String: Any] = try RestChatJourneySupport.requireMessage(
            try RestChatJourneySupport.requireChoice(envelope));
        #expect(assistantMessage["content"] as? String == "Two households, both alike in dignity.");
        let warningHeaderValue: String? = RestChatJourneySupport.responseHeaderValue(
            chatResponse, headerName: "Warning");
        #expect(warningHeaderValue == ResponseFormatConstants.UNENFORCED_RESPONSE_FORMAT_WARNING);
    }

    // MARK: Shared fixture helpers

    static func routeTable(
        modelId: String,
        streamEvents: Array<ChatGenerationStreamEvent>,
        startError: GenerationStartError? = nil
    ) throws -> RestRouteTable {
        let scriptedExecutor: ScriptedChatExecutor = ScriptedChatExecutor(
            healthSnapshot: WorkerHealthSnapshot.readyWithModel(
                modelId: modelId,
                capabilities: RestChatJourneySupport.readyChatCapabilities()),
            streamEvents: streamEvents,
            startError: startError);
        return try RestChatJourneySupport.chatRouteTable(
            resolvedRuntimeConfig: try RestChatJourneySupport.makeResolvedConfig(),
            chatExecutor: scriptedExecutor);
    }
}
