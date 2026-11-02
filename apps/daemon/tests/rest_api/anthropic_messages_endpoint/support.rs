use super::*;

pub(super) fn completed_event(reason: ChatGenerationCompletionReason) -> ChatGenerationStreamEvent {
    ChatGenerationStreamEvent::Completed {
        prompt_token_count: 40,
        generated_token_count: 6,
        reasoning_token_count: 2,
        cached_token_count: 32,
        reason,
    }
}

pub(super) fn messages_request_body(stream: bool) -> String {
    format!(
        r#"{{"model":"{MODEL_ID}","max_tokens":256,"stream":{stream},"messages":[{{"role":"user","content":"Shall I compare thee to a summer's day?"}}]}}"#
    )
}

pub(super) async fn post_messages_body(
    application: axum::Router,
    request_body: &str,
) -> axum::response::Response {
    application
        .oneshot(
            Request::builder()
                .method("POST")
                // Claude Code appends `?beta=true`; the route must match regardless.
                .uri("/v1/messages?beta=true")
                .header(header::CONTENT_TYPE, "application/json")
                .header("anthropic-version", "2023-06-01")
                .body(Body::from(request_body.to_owned()))
                .expect("the Messages request should be valid"),
        )
        .await
        .expect("the application should return an HTTP response")
}

pub(super) async fn response_json(http_response: axum::response::Response) -> Value {
    let response_body = to_bytes(http_response.into_body(), 64 * 1024)
        .await
        .expect("the response body should be readable");
    serde_json::from_slice(&response_body).expect("the response should be JSON")
}

/// One parsed SSE frame: its `event:` name and its JSON `data:` payload.
pub(super) struct ParsedMessagesSseEvent {
    pub(super) event_type: String,
    pub(super) json_payload: Value,
}

pub(super) async fn parsed_sse_events(
    http_response: axum::response::Response,
) -> Vec<ParsedMessagesSseEvent> {
    let response_body = to_bytes(http_response.into_body(), 64 * 1024)
        .await
        .expect("the Messages stream should be readable");
    let response_text =
        String::from_utf8(response_body.to_vec()).expect("the stream should contain UTF-8");
    let mut parsed_events = Vec::new();
    for sse_frame in response_text.split("\n\n") {
        let mut event_type = None;
        let mut json_payload = None;
        for sse_frame_line in sse_frame.lines() {
            if let Some(parsed_event_type) = sse_frame_line.strip_prefix("event: ") {
                event_type = Some(parsed_event_type.to_owned());
            }
            if let Some(sse_data_line) = sse_frame_line.strip_prefix("data: ") {
                json_payload = Some(
                    serde_json::from_str(sse_data_line)
                        .expect("each Messages SSE data payload should be JSON"),
                );
            }
        }
        if let (Some(event_type), Some(json_payload)) = (event_type, json_payload) {
            parsed_events.push(ParsedMessagesSseEvent {
                event_type,
                json_payload,
            });
        }
    }
    parsed_events
}

pub(super) fn event_types(parsed_events: &[ParsedMessagesSseEvent]) -> Vec<&str> {
    let mut event_type_names = Vec::with_capacity(parsed_events.len());
    for parsed_event in parsed_events {
        event_type_names.push(parsed_event.event_type.as_str());
    }
    event_type_names
}
