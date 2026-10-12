//! Network-neutral request and bounded response contracts for Hub metadata calls.

use std::{
    collections::BTreeMap,
    error::Error,
    fmt::{Display, Formatter},
    future::Future,
    pin::Pin,
    time::Duration,
};

pub(crate) const MAXIMUM_RESPONSE_BODY_BYTES: usize = 1_000_000;
pub(crate) const MAXIMUM_RESPONSE_BODY_CHUNKS: usize = 1_024;
const MAXIMUM_SELECTED_HEADER_COUNT: usize = 16;
const MAXIMUM_HEADER_NAME_BYTES: usize = 64;
pub(crate) const MAXIMUM_HEADER_VALUE_BYTES: usize = 8_192;

/// Boxed future keeps the transport object-safe without imposing an async-trait dependency.
pub type HubTransportFuture<'a> =
    Pin<Box<dyn Future<Output = Result<HubHttpResponse, HubTransportError>> + Send + 'a>>;

/// Narrow asynchronous transport implemented by production HTTP and hermetic scripts.
pub trait HubTransport: Send + Sync {
    fn execute(&self, request: HubHttpRequest) -> HubTransportFuture<'_>;
}

/// Method vocabulary intentionally limited to the metadata client's current read-only surface.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HubHttpMethod {
    Get,
}

/// Complete metadata request that a production transport can map without domain knowledge.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HubHttpRequest {
    method: HubHttpMethod,
    url: String,
    headers: [(&'static str, &'static str); 3],
    metadata_timeout: Duration,
}

/// Bounded transport response retaining only selected string headers and body chunks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HubHttpResponse {
    status: u16,
    selected_headers: BTreeMap<String, String>,
    body_chunks: Vec<Vec<u8>>,
    body_byte_count: usize,
}

/// Failure while constructing a response at the untrusted transport boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HubHttpResponseError {
    InvalidStatus,
    TooManyHeaders,
    InvalidHeader,
    DuplicateHeader,
    TooManyBodyChunks,
    BodyTooLarge,
}

/// Transport-owned failure without exposing an HTTP implementation in the domain owner.
#[derive(Debug)]
pub struct HubTransportError {
    message: String,
    source: Option<Box<dyn Error + Send + Sync>>,
}

impl HubHttpRequest {
    #[must_use]
    pub fn metadata_get(url: String) -> Self {
        Self {
            method: HubHttpMethod::Get,
            url,
            headers: [
                ("Accept", "application/json"),
                ("Accept-Encoding", "identity"),
                (
                    "User-Agent",
                    concat!("Astronomical/", env!("CARGO_PKG_VERSION")),
                ),
            ],
            metadata_timeout: Duration::from_secs(30),
        }
    }

    #[must_use]
    pub const fn method(&self) -> HubHttpMethod {
        self.method
    }

    #[must_use]
    pub fn url(&self) -> &str {
        &self.url
    }

    #[must_use]
    pub fn headers(&self) -> &[(&'static str, &'static str)] {
        &self.headers
    }

    #[must_use]
    pub const fn metadata_timeout(&self) -> Duration {
        self.metadata_timeout
    }
}

impl HubHttpMethod {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Get => "GET",
        }
    }
}

impl HubHttpResponse {
    pub fn try_new(
        status: u16,
        selected_headers: impl IntoIterator<Item = (String, String)>,
        body_chunks: impl IntoIterator<Item = Vec<u8>>,
    ) -> Result<Self, HubHttpResponseError> {
        if !(100..=599).contains(&status) {
            return Err(HubHttpResponseError::InvalidStatus);
        }

        let mut bounded_headers = BTreeMap::new();
        for (header_name, header_value) in selected_headers {
            if bounded_headers.len() >= MAXIMUM_SELECTED_HEADER_COUNT {
                return Err(HubHttpResponseError::TooManyHeaders);
            }
            let normalized_header_name = header_name.to_ascii_lowercase();
            if !is_valid_header_name(&normalized_header_name)
                || !is_valid_header_value(&header_value)
            {
                return Err(HubHttpResponseError::InvalidHeader);
            }
            if bounded_headers
                .insert(normalized_header_name, header_value)
                .is_some()
            {
                return Err(HubHttpResponseError::DuplicateHeader);
            }
        }

        let mut bounded_body_chunks = Vec::new();
        let mut body_byte_count = 0_usize;
        for body_chunk in body_chunks {
            if bounded_body_chunks.len() >= MAXIMUM_RESPONSE_BODY_CHUNKS {
                return Err(HubHttpResponseError::TooManyBodyChunks);
            }
            body_byte_count = body_byte_count
                .checked_add(body_chunk.len())
                .ok_or(HubHttpResponseError::BodyTooLarge)?;
            if body_byte_count > MAXIMUM_RESPONSE_BODY_BYTES {
                return Err(HubHttpResponseError::BodyTooLarge);
            }
            bounded_body_chunks.push(body_chunk);
        }

        Ok(Self {
            status,
            selected_headers: bounded_headers,
            body_chunks: bounded_body_chunks,
            body_byte_count,
        })
    }

    #[must_use]
    pub const fn status(&self) -> u16 {
        self.status
    }

    #[must_use]
    pub fn selected_header(&self, header_name: &str) -> Option<&str> {
        self.selected_headers
            .get(&header_name.to_ascii_lowercase())
            .map(String::as_str)
    }

    #[must_use]
    pub const fn body_byte_count(&self) -> usize {
        self.body_byte_count
    }

    #[must_use]
    pub fn body_bytes(&self) -> Vec<u8> {
        let mut complete_body = Vec::with_capacity(self.body_byte_count);
        for body_chunk in &self.body_chunks {
            complete_body.extend_from_slice(body_chunk);
        }
        complete_body
    }
}

impl Display for HubHttpResponseError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::InvalidStatus => "Hub response has an invalid HTTP status",
            Self::TooManyHeaders => "Hub response has too many selected headers",
            Self::InvalidHeader => "Hub response has an invalid selected header",
            Self::DuplicateHeader => "Hub response has a duplicate selected header",
            Self::TooManyBodyChunks => "Hub response has too many body chunks",
            Self::BodyTooLarge => "Hub response exceeds the metadata body byte limit",
        })
    }
}

impl Error for HubHttpResponseError {}

impl HubTransportError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            source: None,
        }
    }

    #[must_use]
    pub fn with_source(
        message: impl Into<String>,
        source: impl Error + Send + Sync + 'static,
    ) -> Self {
        Self {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }
}

impl Display for HubTransportError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for HubTransportError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source
            .as_deref()
            .map(|source| source as &(dyn Error + 'static))
    }
}

fn is_valid_header_name(header_name: &str) -> bool {
    !header_name.is_empty()
        && header_name.len() <= MAXIMUM_HEADER_NAME_BYTES
        && header_name
            .bytes()
            .all(|character| character.is_ascii_alphanumeric() || character == b'-')
}

fn is_valid_header_value(header_value: &str) -> bool {
    header_value.len() <= MAXIMUM_HEADER_VALUE_BYTES
        && header_value
            .bytes()
            .all(|character| character == b' ' || character.is_ascii_graphic())
}
