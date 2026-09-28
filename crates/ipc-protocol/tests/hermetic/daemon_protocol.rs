//! Wire-codec round trips for the ephemeral CLI daemon protocol: the
//! request and response frames a local `astronomical` CLI process and the
//! daemon exchange over the instance unix socket.

use astronomical_ipc_protocol::{
    ChatGenerationCompletionReason, ChatGenerationFailureReason, ChatGenerationSettings,
    ChatMessage, DaemonRequest, DaemonResponse, DaemonWorkerStatus, EmbeddingsFailureReason,
    decode_daemon_request, decode_daemon_response, encode_daemon_request, encode_daemon_response,
};

fn chat_generate_request() -> DaemonRequest {
    DaemonRequest::ChatGenerate {
        model: "example/local-model".to_owned(),
        messages: vec![
            ChatMessage::System {
                content: "Be concise.".to_owned(),
            },
            ChatMessage::User {
                content: "Say hello".to_owned(),
                images: vec![],
            },
        ],
        settings: ChatGenerationSettings {
            max_output_tokens: 128,
            temperature_thousandths: None,
            top_p_thousandths: None,
            seed: None,
            thinking_budget: None,
        },
    }
}

#[test]
fn should_round_trip_daemon_requests_through_the_wire_codec() {
    let daemon_requests = [
        DaemonRequest::Handshake,
        DaemonRequest::Status,
        chat_generate_request(),
        DaemonRequest::EmbedGenerate {
            model: Some("example/local-embedder".to_owned()),
            inputs: vec!["one text".to_owned(), "another text".to_owned()],
            dimensions: Some(256),
        },
        DaemonRequest::EmbedGenerate {
            model: None,
            inputs: vec!["embed whatever is resident".to_owned()],
            dimensions: None,
        },
    ];
    for daemon_request in daemon_requests {
        let encoded_request = encode_daemon_request(&daemon_request)
            .expect("the daemon request should encode to a bounded frame");
        let decoded_request = decode_daemon_request(&encoded_request)
            .expect("the daemon request frame should decode");
        assert_eq!(
            decoded_request, daemon_request,
            "the codec should preserve the daemon request"
        );
    }
}

#[test]
fn should_round_trip_daemon_responses_through_the_wire_codec() {
    let daemon_responses = [
        DaemonResponse::HandshakeAccepted {
            protocol_version: 1,
            application_name: "Astronomical".to_owned(),
        },
        DaemonResponse::Status {
            worker_status: DaemonWorkerStatus::Ready,
            ready_model_id: Some("example/local-model".to_owned()),
        },
        DaemonResponse::Status {
            worker_status: DaemonWorkerStatus::Loading,
            ready_model_id: None,
        },
        DaemonResponse::Status {
            worker_status: DaemonWorkerStatus::Unavailable,
            ready_model_id: None,
        },
        DaemonResponse::ChatGenerationText {
            text: "Hello".to_owned(),
        },
        DaemonResponse::ChatGenerationReasoning {
            text: "Thinking about it.".to_owned(),
        },
        DaemonResponse::ChatGenerationToolCall {
            tool_call_index: 0,
            function_name: "get_weather".to_owned(),
            arguments_json: r#"{"city":"Lisbon"}"#.to_owned(),
        },
        DaemonResponse::ChatGenerationCompleted {
            prompt_token_count: 12,
            generated_token_count: 2,
            reasoning_token_count: 0,
            cached_token_count: 4,
            reason: ChatGenerationCompletionReason::EndOfSequence,
        },
        DaemonResponse::ChatGenerationFailed {
            reason: ChatGenerationFailureReason::InvalidRequest {
                reason: "the worker rejected the request".to_owned(),
            },
        },
        DaemonResponse::GenerationRejected {
            reason: "the worker is not ready".to_owned(),
        },
        DaemonResponse::EmbeddingsCompleted {
            model: "example/local-embedder".to_owned(),
            vectors: vec![vec![0.25, -0.5, 1.0]],
            input_token_counts: vec![3],
        },
        DaemonResponse::EmbeddingsFailed {
            reason: EmbeddingsFailureReason::ContextLengthExceeded {
                actual_total_context_tokens: 5_000,
                maximum_context_tokens: 2_048,
            },
        },
    ];
    for daemon_response in daemon_responses {
        let encoded_response = encode_daemon_response(&daemon_response)
            .expect("the daemon response should encode to a bounded frame");
        let decoded_response = decode_daemon_response(&encoded_response)
            .expect("the daemon response frame should decode");
        assert_eq!(
            decoded_response, daemon_response,
            "the codec should preserve the daemon response"
        );
    }
}
