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
    let mut tcp_stream =
        TcpStream::connect_timeout(&bind_address, timeout).map_err(|io_error| {
            tracing::debug!(error = %io_error, "loopback connect failed");
            LoopbackRequestError::Unreachable
        })?;
    tcp_stream
        .set_read_timeout(Some(timeout))
        .map_err(|_| LoopbackRequestError::Unreachable)?;
    tcp_stream
        .set_write_timeout(Some(timeout))
        .map_err(|_| LoopbackRequestError::Unreachable)?;

    let http_request = format!(
        "GET {request_path} HTTP/1.1\r\nHost: {bind_address}\r\nAccept: application/json\r\nConnection: close\r\n\r\n"
    );
    tcp_stream
        .write_all(http_request.as_bytes())
        .map_err(|io_error| {
            tracing::debug!(error = %io_error, "loopback write failed");
            LoopbackRequestError::Unreachable
        })?;

    let mut response_bytes = Vec::new();
    let mut read_buffer = [0_u8; 8192];
    let mut header_end = None;
    let mut content_length = None;
    loop {
        if response_bytes.len() > MAXIMUM_JSON_BODY_BYTES + 8192 {
            return Err(LoopbackRequestError::Oversized);
        }
        match tcp_stream.read(&mut read_buffer) {
            Ok(0) => break,
            Ok(bytes_read) => response_bytes.extend_from_slice(&read_buffer[..bytes_read]),
            Err(io_error) => {
                tracing::debug!(error = %io_error, "loopback read failed");
                return Err(LoopbackRequestError::Unreachable);
            }
        }
        if header_end.is_none() {
            if let Some(end) = find_header_end(&response_bytes) {
                header_end = Some(end);
                let header_text = std::str::from_utf8(&response_bytes[..end])
                    .map_err(|_| LoopbackRequestError::Malformed)?;
                if header_text
                    .to_ascii_lowercase()
                    .contains("transfer-encoding: chunked")
                {
                    // Astronomical JSON responses use Content-Length. Refuse chunked
                    // encoding rather than hanging on an incomplete stream.
                    return Err(LoopbackRequestError::Rejected);
                }
                content_length = parse_content_length(header_text)?;
            }
        }
        if let (Some(end), Some(body_length)) = (header_end, content_length) {
            let body_start = end + HEADER_TERMINATOR.len();
            if response_bytes.len().saturating_sub(body_start) >= body_length {
                break;
            }
        }
    }

    let header_end = header_end.ok_or(LoopbackRequestError::Malformed)?;
    let header_text = std::str::from_utf8(&response_bytes[..header_end])
        .map_err(|_| LoopbackRequestError::Malformed)?;
    let status_line = header_text
        .split("\r\n")
        .next()
        .ok_or(LoopbackRequestError::Malformed)?;
    if !status_line.starts_with("HTTP/1.1 200") && !status_line.starts_with("HTTP/1.0 200") {
        return Err(LoopbackRequestError::Rejected);
    }
    let body_start = header_end + HEADER_TERMINATOR.len();
    let body_bytes = response_bytes.get(body_start..).unwrap_or(&[]);
    if body_bytes.len() > MAXIMUM_JSON_BODY_BYTES {
        return Err(LoopbackRequestError::Oversized);
    }
    if let Some(body_length) = content_length {
        if body_bytes.len() < body_length {
            return Err(LoopbackRequestError::Unreachable);
        }
        return Ok(body_bytes[..body_length].to_vec());
    }
    Ok(body_bytes.to_vec())
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
