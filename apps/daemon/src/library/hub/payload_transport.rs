//! Streaming request and response contracts for model payload bytes.

use std::{future::Future, pin::Pin, time::Duration};

use bytes::Bytes;
use futures_util::Stream;

use super::HubTransportError;

pub type HubPayloadFuture<'a> =
    Pin<Box<dyn Future<Output = Result<HubPayloadResponse, HubTransportError>> + Send + 'a>>;
pub type HubPayloadByteStream =
    Pin<Box<dyn Stream<Item = Result<Bytes, HubTransportError>> + Send>>;

/// Payload transport remains separate because metadata is deliberately buffered and payloads are not.
pub trait HubPayloadTransport: Send + Sync {
    fn execute_payload(&self, request: HubPayloadRequest) -> HubPayloadFuture<'_>;
}

/// Immutable file request for the bytes from an optional first unread byte up
/// to an optional end.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HubPayloadRequest {
    url: String,
    resume_offset_bytes: u64,
    /// One past the last byte wanted, or `None` for the rest of the file.
    end_offset_bytes: Option<u64>,
    request_timeout: Duration,
}

/// Status and framing metadata followed by a non-buffering byte stream.
pub struct HubPayloadResponse {
    status: u16,
    content_range: Option<String>,
    content_length: Option<u64>,
    byte_stream: HubPayloadByteStream,
}

impl HubPayloadRequest {
    #[must_use]
    pub fn get(url: String, resume_offset_bytes: u64) -> Self {
        Self {
            url,
            resume_offset_bytes,
            end_offset_bytes: None,
            request_timeout: Duration::from_secs(120),
        }
    }

    /// Asks for exactly the bytes from `resume_offset_bytes` up to, but not
    /// including, `end_offset_bytes`. A response that ends where the request
    /// does leaves its connection open for the next request; hanging up on an
    /// open-ended response part-way does not, and every piece would then pay
    /// for a new connection that starts slow.
    #[must_use]
    pub fn get_range(url: String, resume_offset_bytes: u64, end_offset_bytes: u64) -> Self {
        Self {
            end_offset_bytes: Some(end_offset_bytes),
            ..Self::get(url, resume_offset_bytes)
        }
    }

    /// One past the last byte wanted, or `None` for the rest of the file.
    #[must_use]
    pub const fn end_offset_bytes(&self) -> Option<u64> {
        self.end_offset_bytes
    }

    /// The `Range` header value, or `None` when the whole file is wanted.
    #[must_use]
    pub fn range_header_value(&self) -> Option<String> {
        match self.end_offset_bytes {
            Some(end_offset_bytes) => Some(format!(
                "bytes={}-{}",
                self.resume_offset_bytes,
                end_offset_bytes.saturating_sub(1)
            )),
            None if self.resume_offset_bytes > 0 => {
                Some(format!("bytes={}-", self.resume_offset_bytes))
            }
            None => None,
        }
    }

    #[must_use]
    pub fn url(&self) -> &str {
        &self.url
    }

    #[must_use]
    pub const fn resume_offset_bytes(&self) -> u64 {
        self.resume_offset_bytes
    }

    #[must_use]
    pub const fn request_timeout(&self) -> Duration {
        self.request_timeout
    }
}

impl HubPayloadResponse {
    #[must_use]
    pub fn new(
        status: u16,
        content_range: Option<String>,
        content_length: Option<u64>,
        byte_stream: HubPayloadByteStream,
    ) -> Self {
        Self {
            status,
            content_range,
            content_length,
            byte_stream,
        }
    }

    #[must_use]
    pub const fn status(&self) -> u16 {
        self.status
    }

    #[must_use]
    pub fn content_range(&self) -> Option<&str> {
        self.content_range.as_deref()
    }

    #[must_use]
    pub const fn content_length(&self) -> Option<u64> {
        self.content_length
    }

    pub(crate) fn into_byte_stream(self) -> HubPayloadByteStream {
        self.byte_stream
    }
}

#[cfg(test)]
mod payload_request_tests {
    use super::HubPayloadRequest;

    const PAYLOAD_URL: &str = "https://huggingface.co/owner/name/resolve/revision/file";

    #[test]
    fn should_ask_for_an_exact_range_with_an_inclusive_last_byte() {
        let request = HubPayloadRequest::get_range(PAYLOAD_URL.to_owned(), 100, 300);

        assert_eq!(
            request.range_header_value().as_deref(),
            Some("bytes=100-299")
        );
        assert_eq!(request.end_offset_bytes(), Some(300));
    }

    #[test]
    fn should_ask_for_an_exact_range_even_from_byte_zero() {
        let request = HubPayloadRequest::get_range(PAYLOAD_URL.to_owned(), 0, 32);

        assert_eq!(request.range_header_value().as_deref(), Some("bytes=0-31"));
    }

    #[test]
    fn should_ask_for_the_rest_of_the_file_or_send_no_range_without_an_end() {
        let resumed_request = HubPayloadRequest::get(PAYLOAD_URL.to_owned(), 100);
        let whole_file_request = HubPayloadRequest::get(PAYLOAD_URL.to_owned(), 0);

        assert_eq!(
            resumed_request.range_header_value().as_deref(),
            Some("bytes=100-")
        );
        assert_eq!(whole_file_request.range_header_value(), None);
        assert_eq!(whole_file_request.end_offset_bytes(), None);
    }
}
