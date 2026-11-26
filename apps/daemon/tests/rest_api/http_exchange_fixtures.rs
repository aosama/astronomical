//! Issuing GET requests against an in-process daemon application and
//! decoding JSON response bodies, shared by every REST API test module.

use axum::{
    body::{Body, to_bytes},
    http::Request,
};
use tower::ServiceExt;

/// Sends a GET to a clone of the router so the caller keeps its application
/// for follow-up requests; `Router` clones share the same live state.
pub(super) async fn get_response(
    application: &axum::Router,
    request_path: &str,
) -> axum::response::Response {
    application
        .clone()
        .oneshot(
            Request::builder()
                .uri(request_path)
                .body(Body::empty())
                .expect("the GET request should be valid"),
        )
        .await
        .unwrap_or_else(|_| panic!("the application should answer GET {request_path}"))
}

pub(super) async fn read_json_body(
    response: axum::response::Response,
    body_limit_bytes: usize,
) -> serde_json::Value {
    let response_body = to_bytes(response.into_body(), body_limit_bytes)
        .await
        .expect("the response body should be readable");
    serde_json::from_slice(&response_body).expect("the response body should contain JSON")
}

pub(super) async fn read_text_body(
    response: axum::response::Response,
    body_limit_bytes: usize,
) -> String {
    let response_body = to_bytes(response.into_body(), body_limit_bytes)
        .await
        .expect("the response body should be readable");
    String::from_utf8(response_body.to_vec()).expect("the response body should be UTF-8")
}

pub(super) fn post_json_request(
    request_path: &str,
    request_document: serde_json::Value,
) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(request_path)
        .header("content-type", "application/json")
        .body(Body::from(request_document.to_string()))
        .expect("the POST request should be valid")
}
