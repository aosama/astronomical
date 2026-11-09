use std::io::Cursor;

use super::{
    LoopbackRequestError, MAXIMUM_JSON_BODY_BYTES, is_chunked_transfer, read_http_response,
    successful_response_body,
};

const STATUS_BODY: &str = r#"{"application":"Astronomical"}"#;

fn body_of(http_response_text: &str) -> Result<Vec<u8>, LoopbackRequestError> {
    let raw_response = read_http_response(&mut Cursor::new(http_response_text.as_bytes()))?;
    successful_response_body(&raw_response)
}

fn ok_response_with_length(body_text: &str) -> String {
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body_text}",
        body_text.len()
    )
}

#[test]
fn should_return_the_declared_body_of_a_successful_response() {
    let response_body = body_of(&ok_response_with_length(STATUS_BODY)).expect("a 200 response");
    assert_eq!(response_body, STATUS_BODY.as_bytes());
}

#[test]
fn should_cut_the_body_to_its_declared_length() {
    let response_text = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{STATUS_BODY}trailing bytes",
        STATUS_BODY.len()
    );
    assert_eq!(
        body_of(&response_text).expect("a 200 response"),
        STATUS_BODY.as_bytes()
    );
}

#[test]
fn should_read_to_the_close_when_no_length_is_declared() {
    let response_text = format!("HTTP/1.0 200 OK\r\n\r\n{STATUS_BODY}");
    assert_eq!(
        body_of(&response_text).expect("a 200 response"),
        STATUS_BODY.as_bytes()
    );
}

#[test]
fn should_accept_an_empty_declared_body() {
    assert_eq!(
        body_of(&ok_response_with_length("")).expect("a 200 response"),
        b""
    );
}

#[test]
fn should_reject_a_status_other_than_200() {
    let response_text = "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n";
    assert!(matches!(
        body_of(response_text),
        Err(LoopbackRequestError::Rejected)
    ));
}

#[test]
fn should_report_a_body_shorter_than_declared_as_an_early_close() {
    let response_text = "HTTP/1.1 200 OK\r\nContent-Length: 64\r\n\r\n{\"partial\":";
    assert!(matches!(
        body_of(response_text),
        Err(LoopbackRequestError::Unreachable)
    ));
}

#[test]
fn should_reject_a_response_that_never_finishes_its_header() {
    let response_text = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n";
    assert!(matches!(
        body_of(response_text),
        Err(LoopbackRequestError::Malformed)
    ));
}

#[test]
fn should_reject_a_declared_length_above_the_body_limit() {
    let response_text = format!(
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n",
        MAXIMUM_JSON_BODY_BYTES + 1
    );
    assert!(matches!(
        body_of(&response_text),
        Err(LoopbackRequestError::Oversized)
    ));
}

#[test]
fn should_reject_an_undeclared_body_above_the_limit() {
    let response_text = format!(
        "HTTP/1.0 200 OK\r\n\r\n{}",
        "x".repeat(MAXIMUM_JSON_BODY_BYTES + 1)
    );
    assert!(matches!(
        body_of(&response_text),
        Err(LoopbackRequestError::Oversized)
    ));
}

#[test]
fn should_refuse_chunked_responses_in_every_header_spelling() {
    for chunked_header in [
        "Transfer-Encoding: chunked",
        "transfer-encoding:chunked",
        "TRANSFER-ENCODING:  gzip, Chunked",
    ] {
        let response_text =
            format!("HTTP/1.1 200 OK\r\n{chunked_header}\r\n\r\n2\r\n{{}}\r\n0\r\n\r\n");
        assert!(
            matches!(body_of(&response_text), Err(LoopbackRequestError::Rejected)),
            "{chunked_header} must be refused"
        );
    }
}

#[test]
fn should_not_mistake_other_headers_for_chunked_transfer() {
    assert!(!is_chunked_transfer(
        "HTTP/1.1 200 OK\r\nX-Note: chunked\r\nContent-Length: 2"
    ));
    assert!(!is_chunked_transfer(
        "HTTP/1.1 200 OK\r\nTransfer-Encoding: gzip"
    ));
    assert!(!is_chunked_transfer("HTTP/1.1 200 OK"));
}
