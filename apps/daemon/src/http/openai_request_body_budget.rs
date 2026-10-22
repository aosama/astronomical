use astronomical_rest_contract::{MAX_OPENAI_REQUEST_BODY_BYTES, OpenAiErrorResponse};
use axum::{
    Json,
    body::{Body, to_bytes},
    extract::Request,
    http::{StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};

/// The mainstream ceiling the body budget mirrors, named in rejections so the
/// message satisfies the OpenAI-compatible fidelity contract: the caller learns
/// which hosted limit the local rejection protects.
const MIRRORED_MAINSTREAM_BODY_CEILING: &str =
    "mirrors the 32 MB standard-endpoint request size limit of the Anthropic Messages API";

/// Rejects OpenAI-compatible request bodies wider than the portable request
/// budget with an OpenAI-shaped 413 before the request reaches an endpoint.
///
/// The value comes from the single budget module in `astronomical-rest-contract`
/// and mirrors the 32 MB standard-endpoint request size limit Anthropic
/// documents (retrieved 2026-10-22). A declared `Content-Length` over the
/// budget is rejected before any body byte is read; a streamed body without the
/// header is buffered against the budget so the typed rejection cannot be
/// bypassed by a chunked client. `DefaultBodyLimit` stays layered beneath as
/// the transport backstop for the same value.
pub(crate) async fn enforce_openai_request_body_budget(request: Request, next: Next) -> Response {
    if let Some(declared_body_bytes) = declared_content_length(&request) {
        if declared_body_bytes > MAX_OPENAI_REQUEST_BODY_BYTES {
            return request_body_too_large_response(
                declared_body_bytes,
                MAX_OPENAI_REQUEST_BODY_BYTES,
            );
        }
        return next.run(request).await;
    }
    let (request_parts, request_body) = request.into_parts();
    // One byte past the budget makes an overflow observable while keeping the
    // buffering bounded; a body beyond that is rejected without exact overshoot.
    let read_outcome = to_bytes(request_body, MAX_OPENAI_REQUEST_BODY_BYTES + 1).await;
    match read_outcome {
        Ok(buffered_request_body) => {
            if buffered_request_body.len() > MAX_OPENAI_REQUEST_BODY_BYTES {
                return request_body_too_large_response(
                    buffered_request_body.len(),
                    MAX_OPENAI_REQUEST_BODY_BYTES,
                );
            }
            next.run(Request::from_parts(
                request_parts,
                Body::from(buffered_request_body),
            ))
            .await
        }
        Err(_) => request_body_too_large_without_overshoot_response(),
    }
}

fn declared_content_length(request: &Request) -> Option<usize> {
    request
        .headers()
        .get(header::CONTENT_LENGTH)?
        .to_str()
        .ok()?
        .parse()
        .ok()
}

fn request_body_too_large_response(actual_body_bytes: usize, budget_bytes: usize) -> Response {
    (
        StatusCode::PAYLOAD_TOO_LARGE,
        Json(OpenAiErrorResponse::invalid_request(
            format!(
                "the request body is {actual_body_bytes} bytes, exceeding the {budget_bytes}-byte request budget by {} bytes ({MIRRORED_MAINSTREAM_BODY_CEILING})",
                actual_body_bytes - budget_bytes
            ),
            None,
            Some("request_too_large"),
        )),
    )
        .into_response()
}

fn request_body_too_large_without_overshoot_response() -> Response {
    (
        StatusCode::PAYLOAD_TOO_LARGE,
        Json(OpenAiErrorResponse::invalid_request(
            format!(
                "the request body exceeds the {MAX_OPENAI_REQUEST_BODY_BYTES}-byte request budget ({MIRRORED_MAINSTREAM_BODY_CEILING})"
            ),
            None,
            Some("request_too_large"),
        )),
    )
        .into_response()
}
