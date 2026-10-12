use crate::common::MODEL_ID;
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use tower::ServiceExt;

pub(super) async fn get_status(application: &axum::Router, path: &str) -> StatusCode {
    application
        .clone()
        .oneshot(
            Request::builder()
                .uri(path)
                .body(Body::empty())
                .expect("the GET request should be valid"),
        )
        .await
        .expect("the application should return a response")
        .status()
}
pub(super) async fn post_chat(application: axum::Router) -> String {
    post_chat_with_message(application, "hello").await
}
pub(super) async fn post_chat_with_message(
    application: axum::Router,
    user_message: &str,
) -> String {
    let serialized_user_message =
        serde_json::to_string(user_message).expect("the user message should serialize into JSON");
    let chat_response = application.oneshot(Request::builder().method("POST").uri("/v1/chat/completions").header(header::CONTENT_TYPE, "application/json").body(Body::from(format!(r#"{{"model":"{MODEL_ID}","messages":[{{"role":"user","content":{serialized_user_message}}}],"stream":true}}"#))).expect("the chat request should be valid")).await.expect("the application should return a chat response");
    assert_eq!(chat_response.status(), StatusCode::OK);
    let response_body = to_bytes(chat_response.into_body(), 16 * 1024)
        .await
        .expect("the chat response body should be readable");
    String::from_utf8(response_body.to_vec()).expect("the SSE body should contain UTF-8")
}
pub(super) fn extract_tool_call_id(response_body: &str) -> &str {
    let id_prefix = r#""id":"call_"#;
    let id_start = response_body
        .find(id_prefix)
        .expect("the response should contain a tool-call ID")
        + r#""id":""#.len();
    let id_end = response_body[id_start..]
        .find('"')
        .map(|relative_end| id_start + relative_end)
        .expect("the tool-call ID should have a closing quote");
    &response_body[id_start..id_end]
}
