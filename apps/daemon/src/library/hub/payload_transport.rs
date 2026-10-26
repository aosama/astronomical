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

/// Immutable file request with an optional first unread byte.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HubPayloadRequest {
    url: String,
    resume_offset_bytes: u64,
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
            request_timeout: Duration::from_secs(120),
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
