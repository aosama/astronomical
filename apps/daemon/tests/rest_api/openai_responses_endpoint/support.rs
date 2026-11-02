use super::*;

pub(super) async fn post_response(
    application: axum::Router,
    stream: bool,
) -> axum::response::Response {
    post_response_body(
        application,
        &format!(r#"{{"model":"{MODEL_ID}","input":"hello","stream":{stream}}}"#),
    )
    .await
}

pub(super) async fn post_response_body(
    application: axum::Router,
    request_body: &str,
) -> axum::response::Response {
    application
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(request_body.to_owned()))
                .expect("the Responses request should be valid"),
        )
        .await
        .expect("the application should return an HTTP response")
}

pub(super) async fn assert_error_code(
    http_response: axum::response::Response,
    expected_error_code: &str,
) {
    let response_body = to_bytes(http_response.into_body(), 16 * 1024)
        .await
        .expect("the error body should be readable");
    let response_document: serde_json::Value =
        serde_json::from_slice(&response_body).expect("the error should be JSON");
    assert_eq!(response_document["error"]["code"], expected_error_code);
}

pub(super) struct ParsedResponsesSseStream {
    events: Vec<ParsedResponsesSseEvent>,
}

struct ParsedResponsesSseEvent {
    event_type: String,
    json_payload: Value,
}

impl ParsedResponsesSseStream {
    pub(super) fn parse(response_text: &str) -> Self {
        let events = response_text
            .split("\n\n")
            .filter_map(|sse_frame| {
                let mut event_type = None;
                let mut sse_data_lines = Vec::new();
                for sse_frame_line in sse_frame.lines() {
                    if let Some(parsed_event_type) = sse_frame_line.strip_prefix("event: ") {
                        event_type = Some(parsed_event_type.to_owned());
                    }
                    if let Some(sse_data_line) = sse_frame_line.strip_prefix("data: ") {
                        sse_data_lines.push(sse_data_line);
                    }
                }
                let event_type = event_type?;
                if sse_data_lines.is_empty() {
                    return None;
                }
                let sse_data_payload = sse_data_lines.join("\n");
                Some(ParsedResponsesSseEvent {
                    event_type,
                    json_payload: serde_json::from_str(&sse_data_payload)
                        .expect("each Responses SSE data payload should be valid JSON"),
                })
            })
            .collect();
        Self { events }
    }

    pub(super) fn event_types(&self) -> Vec<&str> {
        self.events
            .iter()
            .map(|sse_event| sse_event.event_type.as_str())
            .collect()
    }

    pub(super) fn visible_text_for_opencode(&self) -> String {
        self.events
            .iter()
            .filter(|sse_event| sse_event.event_type == "response.output_text.delta")
            .filter_map(|sse_event| sse_event.json_payload["delta"].as_str())
            .collect::<String>()
    }

    pub(super) fn reasoning_summary_text(&self) -> String {
        self.events
            .iter()
            .filter(|sse_event| sse_event.event_type == "response.reasoning_summary_text.delta")
            .filter_map(|sse_event| sse_event.json_payload["delta"].as_str())
            .collect::<String>()
    }

    pub(super) fn first_payload_for_event_type(&self, expected_event_type: &str) -> Option<&Value> {
        self.events
            .iter()
            .find(|sse_event| sse_event.event_type == expected_event_type)
            .map(|sse_event| &sse_event.json_payload)
    }

    pub(super) fn completed_response(&self) -> Option<&Value> {
        self.first_payload_for_event_type("response.completed")
            .map(|json_payload| &json_payload["response"])
    }
}
