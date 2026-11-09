//! Bounded loopback GET for launch. This layer reports transport outcomes;
//! launch maps them to one-line user errors so a running instance is never
//! described as missing.

use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpStream},
    time::{Duration, Instant},
};

use serde_json::Value;

const MAXIMUM_JSON_BODY_BYTES: usize = 1_000_000;
const HEADER_TERMINATOR: &[u8] = b"\r\n\r\n";

/// Why a loopback GET could not produce JSON.
#[derive(Debug)]
pub(crate) enum LoopbackRequestError {
    Unreachable,
    Rejected,
    Oversized,
    Malformed,
}

/// GET JSON from one Astronomical loopback path with an explicit timeout.
pub(crate) fn get_loopback_json(
    bind_address: SocketAddr,
    request_path: &str,
    timeout: Duration,
) -> Result<Value, LoopbackRequestError> {
    let request_started_at = Instant::now();
    let response_body = match fetch_http_body(bind_address, request_path, timeout) {
        Ok(response_body) => response_body,
        Err(request_error) => {
            tracing::debug!(
                bind_address = %bind_address,
                request_path,
                elapsed_milliseconds = request_started_at.elapsed().as_millis() as u64,
                "loopback GET failed"
            );
            return Err(request_error);
        }
    };
    tracing::debug!(
        bind_address = %bind_address,
        request_path,
        elapsed_milliseconds = request_started_at.elapsed().as_millis() as u64,
        body_bytes = response_body.len(),
        "loopback GET completed"
    );
    serde_json::from_slice(&response_body).map_err(|parse_error| {
        tracing::debug!(error = %parse_error, "loopback JSON was not usable");
        LoopbackRequestError::Malformed
    })
}

fn fetch_http_body(
    bind_address: SocketAddr,
    request_path: &str,
    timeout: Duration,
) -> Result<Vec<u8>, LoopbackRequestError> {
    let mut tcp_stream = open_loopback_stream(bind_address, timeout)?;
    let http_request = format!(
        "GET {request_path} HTTP/1.1\r\nHost: {bind_address}\r\nAccept: application/json\r\nConnection: close\r\n\r\n"
    );
    tcp_stream
        .write_all(http_request.as_bytes())
        .map_err(|io_error| {
            tracing::debug!(error = %io_error, "loopback write failed");
            LoopbackRequestError::Unreachable
        })?;
    let raw_response = read_http_response(&mut tcp_stream)?;
    successful_response_body(&raw_response)
}

/// Connects with `timeout` bounding the connect, every read and every
/// write, so a stalled instance can never hang `launch`.
fn open_loopback_stream(
    bind_address: SocketAddr,
    timeout: Duration,
) -> Result<TcpStream, LoopbackRequestError> {
    let tcp_stream = TcpStream::connect_timeout(&bind_address, timeout).map_err(|io_error| {
        tracing::debug!(error = %io_error, "loopback connect failed");
        LoopbackRequestError::Unreachable
    })?;
    tcp_stream
        .set_read_timeout(Some(timeout))
        .map_err(|_| LoopbackRequestError::Unreachable)?;
    tcp_stream
        .set_write_timeout(Some(timeout))
        .map_err(|_| LoopbackRequestError::Unreachable)?;
    Ok(tcp_stream)
}

/// The bytes of one response as read, where its header ends, and the body
/// length the header declared, if any.
#[derive(Debug)]
struct RawHttpResponse {
    response_bytes: Vec<u8>,
    header_end: usize,
    content_length: Option<usize>,
}

/// Where the header ends and the body length it declared, once the header
/// has fully arrived.
#[derive(Clone, Copy)]
struct ParsedResponseHead {
    header_end: usize,
    content_length: Option<usize>,
}

/// Reads until the peer closes or the declared body has arrived, refusing
/// to buffer more than one maximal body plus one read of header.
fn read_http_response(
    response_stream: &mut impl Read,
) -> Result<RawHttpResponse, LoopbackRequestError> {
    let mut response_bytes = Vec::new();
    let mut read_buffer = [0_u8; 8192];
    let mut parsed_head: Option<ParsedResponseHead> = None;
    loop {
        if response_bytes.len() > MAXIMUM_JSON_BODY_BYTES + read_buffer.len() {
            return Err(LoopbackRequestError::Oversized);
        }
        match response_stream.read(&mut read_buffer) {
            Ok(0) => break,
            Ok(bytes_read) => response_bytes.extend_from_slice(&read_buffer[..bytes_read]),
            Err(io_error) => {
                tracing::debug!(error = %io_error, "loopback read failed");
                return Err(LoopbackRequestError::Unreachable);
            }
        }
        if parsed_head.is_none()
            && let Some(header_end) = find_header_end(&response_bytes)
        {
            let content_length = parse_response_head(header_text(&response_bytes, header_end)?)?;
            parsed_head = Some(ParsedResponseHead {
                header_end,
                content_length,
            });
        }
        if let Some(ParsedResponseHead {
            header_end,
            content_length: Some(body_length),
        }) = parsed_head
            && response_bytes
                .len()
                .saturating_sub(header_end + HEADER_TERMINATOR.len())
                >= body_length
        {
            break;
        }
    }
    let ParsedResponseHead {
        header_end,
        content_length,
    } = parsed_head.ok_or(LoopbackRequestError::Malformed)?;
    Ok(RawHttpResponse {
        response_bytes,
        header_end,
        content_length,
    })
}

fn header_text(response_bytes: &[u8], header_end: usize) -> Result<&str, LoopbackRequestError> {
    std::str::from_utf8(&response_bytes[..header_end]).map_err(|_| LoopbackRequestError::Malformed)
}

/// The declared body length. Astronomical JSON responses use
/// Content-Length; a chunked response is refused rather than read to EOF,
/// where its chunk framing would corrupt the JSON.
fn parse_response_head(header_text: &str) -> Result<Option<usize>, LoopbackRequestError> {
    if is_chunked_transfer(header_text) {
        return Err(LoopbackRequestError::Rejected);
    }
    parse_content_length(header_text)
}

/// Matches the header by name and the coding as one list entry, so
/// `Transfer-Encoding:chunked` and `gzip, chunked` are both recognized.
fn is_chunked_transfer(header_text: &str) -> bool {
    for header_line in header_text.split("\r\n").skip(1) {
        let Some((header_name, header_value)) = header_line.split_once(':') else {
            continue;
        };
        if !header_name.trim().eq_ignore_ascii_case("transfer-encoding") {
            continue;
        }
        for transfer_coding in header_value.split(',') {
            if transfer_coding.trim().eq_ignore_ascii_case("chunked") {
                return true;
            }
        }
    }
    false
}

/// The body of a 200 response, cut to its declared length; a body shorter
/// than declared means the peer closed early.
fn successful_response_body(
    raw_response: &RawHttpResponse,
) -> Result<Vec<u8>, LoopbackRequestError> {
    let response_header_text = header_text(&raw_response.response_bytes, raw_response.header_end)?;
    let status_line = response_header_text
        .split("\r\n")
        .next()
        .ok_or(LoopbackRequestError::Malformed)?;
    if !status_line.starts_with("HTTP/1.1 200") && !status_line.starts_with("HTTP/1.0 200") {
        return Err(LoopbackRequestError::Rejected);
    }
    let body_start = raw_response.header_end + HEADER_TERMINATOR.len();
    let body_bytes = raw_response.response_bytes.get(body_start..).unwrap_or(&[]);
    if body_bytes.len() > MAXIMUM_JSON_BODY_BYTES {
        return Err(LoopbackRequestError::Oversized);
    }
    match raw_response.content_length {
        Some(body_length) if body_bytes.len() < body_length => {
            Err(LoopbackRequestError::Unreachable)
        }
        Some(body_length) => Ok(body_bytes[..body_length].to_vec()),
        None => Ok(body_bytes.to_vec()),
    }
}

fn find_header_end(response_bytes: &[u8]) -> Option<usize> {
    response_bytes
        .windows(HEADER_TERMINATOR.len())
        .position(|window| window == HEADER_TERMINATOR)
}

fn parse_content_length(header_text: &str) -> Result<Option<usize>, LoopbackRequestError> {
    for header_line in header_text.split("\r\n").skip(1) {
        let Some((header_name, header_value)) = header_line.split_once(':') else {
            continue;
        };
        if !header_name.eq_ignore_ascii_case("content-length") {
            continue;
        }
        let parsed_length = header_value
            .trim()
            .parse::<usize>()
            .map_err(|_| LoopbackRequestError::Malformed)?;
        if parsed_length > MAXIMUM_JSON_BODY_BYTES {
            return Err(LoopbackRequestError::Oversized);
        }
        return Ok(Some(parsed_length));
    }
    Ok(None)
}

#[cfg(test)]
#[path = "loopback_http_tests.rs"]
mod loopback_http_tests;
