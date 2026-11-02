use astronomical_rest_contract::{AnthropicError, MAX_REQUEST_BODY_BYTES, OpenAiErrorResponse};
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

/// Path prefix of the Anthropic Messages routes (`/v1/messages` and
/// `/v1/messages/count_tokens`), whose clients parse Anthropic error envelopes.
const ANTHROPIC_MESSAGES_PATH_PREFIX: &str = "/v1/messages";

/// The error envelope a rejection is written in, chosen by the protocol the
/// rejected route speaks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RejectionEnvelope {
    OpenAi,
    Anthropic,
}

/// Pure: the envelope for a request path. The budget guards both protocol
/// surfaces, so its 413 must speak the protocol of the route it rejected.
pub(crate) fn rejection_envelope_for_path(request_path: &str) -> RejectionEnvelope {
    if request_path.starts_with(ANTHROPIC_MESSAGES_PATH_PREFIX) {
        RejectionEnvelope::Anthropic
    } else {
        RejectionEnvelope::OpenAi
    }
}

/// Rejects generation request bodies wider than the portable request budget
/// with a 413 before the request reaches an endpoint, written in the error
/// envelope of the protocol the route speaks.
///
/// The value comes from the single budget module in `astronomical-rest-contract`
/// and mirrors the 32 MB standard-endpoint request size limit Anthropic
/// documents (retrieved 2026-10-22). A declared `Content-Length` over the
/// budget is rejected before any body byte is read; a streamed body without the
/// header is buffered against the budget so the typed rejection cannot be
/// bypassed by a chunked client. `DefaultBodyLimit` stays layered beneath as
/// the transport backstop for the same value.
pub(crate) async fn enforce_request_body_budget(request: Request, next: Next) -> Response {
    let rejection_envelope = rejection_envelope_for_path(request.uri().path());
    if let Some(declared_body_bytes) = declared_content_length(&request) {
        if declared_body_bytes > MAX_REQUEST_BODY_BYTES {
            return request_body_too_large_response(
                declared_body_bytes,
                MAX_REQUEST_BODY_BYTES,
                rejection_envelope,
            );
        }
        return next.run(request).await;
    }
    let (request_parts, request_body) = request.into_parts();
    // One byte past the budget makes an overflow observable while keeping the
    // buffering bounded; a body beyond that is rejected without exact overshoot.
    let read_outcome = to_bytes(request_body, MAX_REQUEST_BODY_BYTES + 1).await;
    match read_outcome {
        Ok(buffered_request_body) => {
            if buffered_request_body.len() > MAX_REQUEST_BODY_BYTES {
                return request_body_too_large_response(
                    buffered_request_body.len(),
                    MAX_REQUEST_BODY_BYTES,
                    rejection_envelope,
                );
            }
            next.run(Request::from_parts(
                request_parts,
                Body::from(buffered_request_body),
            ))
            .await
        }
        Err(_) => request_body_too_large_without_overshoot_response(rejection_envelope),
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

fn request_body_too_large_response(
    actual_body_bytes: usize,
    budget_bytes: usize,
    rejection_envelope: RejectionEnvelope,
) -> Response {
    too_large_rejection(
        format!(
            "the request body is {actual_body_bytes} bytes, exceeding the {budget_bytes}-byte request budget by {} bytes ({MIRRORED_MAINSTREAM_BODY_CEILING})",
            actual_body_bytes - budget_bytes
        ),
        rejection_envelope,
    )
}

fn request_body_too_large_without_overshoot_response(
    rejection_envelope: RejectionEnvelope,
) -> Response {
    too_large_rejection(
        format!(
            "the request body exceeds the {MAX_REQUEST_BODY_BYTES}-byte request budget ({MIRRORED_MAINSTREAM_BODY_CEILING})"
        ),
        rejection_envelope,
    )
}

/// One 413, written in the rejected route's envelope with the same message.
fn too_large_rejection(
    rejection_message: String,
    rejection_envelope: RejectionEnvelope,
) -> Response {
    match rejection_envelope {
        RejectionEnvelope::OpenAi => (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(OpenAiErrorResponse::invalid_request(
                rejection_message,
                None,
                Some("request_too_large"),
            )),
        )
            .into_response(),
        RejectionEnvelope::Anthropic => (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(AnthropicError::request_too_large(rejection_message).into_response_body()),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::{RejectionEnvelope, rejection_envelope_for_path};

    #[test]
    fn should_choose_the_anthropic_envelope_for_both_messages_routes() {
        assert_eq!(
            rejection_envelope_for_path("/v1/messages"),
            RejectionEnvelope::Anthropic
        );
        assert_eq!(
            rejection_envelope_for_path("/v1/messages/count_tokens"),
            RejectionEnvelope::Anthropic
        );
    }

    #[test]
    fn should_keep_the_openai_envelope_for_every_other_route() {
        for openai_path in [
            "/v1/chat/completions",
            "/v1/responses",
            "/v1/embeddings",
            "/",
        ] {
            assert_eq!(
                rejection_envelope_for_path(openai_path),
                RejectionEnvelope::OpenAi
            );
        }
    }

    #[test]
    fn should_keep_the_openai_envelope_for_an_empty_path() {
        assert_eq!(rejection_envelope_for_path(""), RejectionEnvelope::OpenAi);
    }
}
