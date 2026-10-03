//! Minimal OpenAI-compatible streaming chat client for the acceptance journeys.
//!
//! The journeys only need a raw JSON request and the server-sent-event chunk
//! values over loopback HTTP, so a direct TCP client replaces the previous
//! OpenAI SDK dev-dependency and keeps its closure out of the test build graph.

use std::{fmt, net::SocketAddr, pin::Pin};

use futures_util::Stream;
use serde_json::Value;
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    net::TcpStream,
};

/// Bounds a non-success response body so an error cannot grow without limit.
const ERROR_BODY_BOUND_BYTES: u64 = 64 * 1024;

/// One streaming chat completion client bound to a local REST endpoint.
pub(crate) struct LocalOpenAiClient {
    server_address: SocketAddr,
    api_key: String,
}

impl LocalOpenAiClient {
    pub(crate) fn new(server_address: SocketAddr, api_key: &str) -> Self {
        Self {
            server_address,
            api_key: api_key.to_owned(),
        }
    }

    /// Starts one streaming chat completion and returns its chunk stream.
    ///
    /// The request document is sent verbatim to `/v1/chat/completions`; each
    /// server-sent `data:` payload is yielded as one parsed JSON value, and the
    /// stream ends at `data: [DONE]` or when the connection closes.
    pub(crate) async fn create_streaming_chat_completion(
        &self,
        request: &Value,
    ) -> Result<ChatCompletionStream, LocalOpenAiClientError> {
        let mut connection =
            TcpStream::connect(self.server_address)
                .await
                .map_err(|connect_error| {
                    LocalOpenAiClientError::Connection(format!(
                        "the local REST endpoint should accept a connection: {connect_error}"
                    ))
                })?;
        let request_body = request.to_string();
        let request_text = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nAccept: text/event-stream\r\nAuthorization: Bearer {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            self.server_address,
            self.api_key,
            request_body.len(),
            request_body,
        );
        connection
            .write_all(request_text.as_bytes())
            .await
            .map_err(|write_error| {
                LocalOpenAiClientError::Request(format!(
                    "the streaming chat request should be written: {write_error}"
                ))
            })?;
        // `read_until` lives on `AsyncBufReadExt`, so the raw stream is wrapped
        // in a `BufReader` before any line-oriented reading begins.
        let mut reader = BufReader::new(connection);
        let status_line = read_response_head(&mut reader).await?;
        if !status_line.starts_with("HTTP/1.1 2") {
            let error_body = read_bounded_body(&mut reader).await;
            return Err(LocalOpenAiClientError::HttpStatus {
                status: status_line.trim_end().to_owned(),
                body: error_body,
            });
        }
        Ok(Box::pin(futures_util::stream::unfold(
            StreamingChatState {
                reader,
                stream_finished: false,
            },
            |state| Box::pin(next_chat_chunk(state)),
        )))
    }
}

/// Reads the status line plus headers, returning the status line.
async fn read_response_head(
    reader: &mut BufReader<TcpStream>,
) -> Result<String, LocalOpenAiClientError> {
    let status_line = read_response_line(reader, "status line").await?;
    loop {
        let header_line = read_response_line(reader, "header line").await?;
        if header_line.trim_end().is_empty() {
            return Ok(status_line);
        }
    }
}

async fn read_response_line(
    reader: &mut BufReader<TcpStream>,
    line_role: &str,
) -> Result<String, LocalOpenAiClientError> {
    let mut line = Vec::new();
    let read_count = reader
        .read_until(b'\n', &mut line)
        .await
        .map_err(|read_error| {
            LocalOpenAiClientError::Stream(format!(
                "the HTTP response {line_role} should be readable: {read_error}"
            ))
        })?;
    if read_count == 0 {
        return Err(LocalOpenAiClientError::Stream(format!(
            "the HTTP response ended before its {line_role}"
        )));
    }
    Ok(String::from_utf8_lossy(&line).into_owned())
}

async fn read_bounded_body(reader: &mut BufReader<TcpStream>) -> String {
    let mut error_body = String::new();
    let _ = reader
        .take(ERROR_BODY_BOUND_BYTES)
        .read_to_string(&mut error_body)
        .await;
    error_body
}

struct StreamingChatState {
    reader: BufReader<TcpStream>,
    stream_finished: bool,
}

/// One parsed server-sent-event chunk per item; ends at `data: [DONE]` or EOF.
pub(crate) type ChatCompletionStream =
    Pin<Box<dyn Stream<Item = Result<Value, LocalOpenAiClientError>> + Send>>;

async fn next_chat_chunk(
    mut state: StreamingChatState,
) -> Option<(Result<Value, LocalOpenAiClientError>, StreamingChatState)> {
    loop {
        if state.stream_finished {
            return None;
        }
        let mut line = Vec::new();
        let read_count = match state.reader.read_until(b'\n', &mut line).await {
            Ok(read_count) => read_count,
            Err(read_error) => {
                state.stream_finished = true;
                return Some((
                    Err(LocalOpenAiClientError::Stream(format!(
                        "the streaming chat response should remain readable: {read_error}"
                    ))),
                    state,
                ));
            }
        };
        if read_count == 0 {
            state.stream_finished = true;
            if line.is_empty() {
                return None;
            }
            // A final line without a trailing newline still carries a complete payload.
            return match parse_sse_line(&String::from_utf8_lossy(&line)) {
                SseLineOutcome::Chunk(chunk) => Some((Ok(chunk), state)),
                SseLineOutcome::Done | SseLineOutcome::Ignore => None,
                SseLineOutcome::InvalidPayload(payload) => Some((
                    Err(LocalOpenAiClientError::Stream(format!(
                        "the final SSE payload should be valid JSON: {payload}"
                    ))),
                    state,
                )),
            };
        }
        match parse_sse_line(&String::from_utf8_lossy(&line)) {
            SseLineOutcome::Chunk(chunk) => return Some((Ok(chunk), state)),
            SseLineOutcome::Done => {
                state.stream_finished = true;
                return None;
            }
            SseLineOutcome::Ignore => continue,
            SseLineOutcome::InvalidPayload(payload) => {
                state.stream_finished = true;
                return Some((
                    Err(LocalOpenAiClientError::Stream(format!(
                        "the SSE payload should be valid JSON: {payload}"
                    ))),
                    state,
                ));
            }
        }
    }
}

#[derive(Debug)]
enum SseLineOutcome {
    Chunk(Value),
    Done,
    Ignore,
    InvalidPayload(String),
}

/// Parses one server-sent-event line into a chat chunk.
///
/// Only `data:` lines carry payloads; comments, `event:` lines, and blank
/// lines are ignored. `data: [DONE]` terminates the stream.
fn parse_sse_line(line: &str) -> SseLineOutcome {
    let trimmed_line = line.trim_end_matches(['\r', '\n']);
    let Some(payload) = trimmed_line.strip_prefix("data:") else {
        return SseLineOutcome::Ignore;
    };
    let payload = payload.trim_start();
    if payload == "[DONE]" {
        return SseLineOutcome::Done;
    }
    if payload.is_empty() {
        return SseLineOutcome::Ignore;
    }
    match serde_json::from_str::<Value>(payload) {
        Ok(chunk) => SseLineOutcome::Chunk(chunk),
        Err(parse_error) => SseLineOutcome::InvalidPayload(parse_error.to_string()),
    }
}

#[derive(Debug)]
pub(crate) enum LocalOpenAiClientError {
    Connection(String),
    Request(String),
    HttpStatus { status: String, body: String },
    Stream(String),
}

impl fmt::Display for LocalOpenAiClientError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Connection(detail) => write!(formatter, "connection failure: {detail}"),
            Self::Request(detail) => write!(formatter, "request failure: {detail}"),
            Self::HttpStatus { status, body } => {
                write!(formatter, "non-success HTTP response {status}: {body}")
            }
            Self::Stream(detail) => write!(formatter, "stream failure: {detail}"),
        }
    }
}

impl std::error::Error for LocalOpenAiClientError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_parse_a_data_payload_line_into_a_chunk() {
        let outcome =
            parse_sse_line("data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\r\n");
        assert!(
            matches!(&outcome, SseLineOutcome::Chunk(chunk) if chunk["choices"][0]["delta"]["content"] == "hello"),
            "the data payload should parse into its chunk: {outcome:?}"
        );
    }

    #[test]
    fn should_treat_the_done_sentinel_as_stream_end() {
        assert!(
            matches!(parse_sse_line("data: [DONE]\n"), SseLineOutcome::Done),
            "the [DONE] sentinel should terminate the stream"
        );
    }

    #[test]
    fn should_ignore_event_comment_and_blank_lines() {
        assert!(matches!(parse_sse_line(""), SseLineOutcome::Ignore));
        assert!(matches!(
            parse_sse_line("event: message\n"),
            SseLineOutcome::Ignore
        ));
        assert!(matches!(
            parse_sse_line(": keep-alive comment\n"),
            SseLineOutcome::Ignore
        ));
        assert!(matches!(parse_sse_line("data:\n"), SseLineOutcome::Ignore));
    }

    #[test]
    fn should_report_an_invalid_payload_with_its_parse_error() {
        assert!(
            matches!(
                parse_sse_line("data: {not json}\n"),
                SseLineOutcome::InvalidPayload(_)
            ),
            "a malformed data payload should surface its parse error"
        );
    }
}
