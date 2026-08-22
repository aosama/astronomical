//! Rustls-only production transport for bounded Hugging Face metadata requests.

use std::time::Duration;

use futures_util::StreamExt;
use reqwest::{
    Client, Url,
    header::{ACCEPT_ENCODING, CONTENT_RANGE, LINK, RANGE, USER_AGENT},
    redirect,
};
use thiserror::Error;

use super::hub_transport::{
    HubHttpMethod, HubHttpRequest, HubHttpResponse, HubTransport, HubTransportError,
    HubTransportFuture, MAXIMUM_HEADER_VALUE_BYTES, MAXIMUM_RESPONSE_BODY_BYTES,
    MAXIMUM_RESPONSE_BODY_CHUNKS,
};
use super::{HubPayloadFuture, HubPayloadRequest, HubPayloadResponse, HubPayloadTransport};

const HUGGING_FACE_HOST: &str = "huggingface.co";
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const READ_TIMEOUT: Duration = Duration::from_secs(10);
// Payload files can be multi-gigabyte; a per-read idle gap on a content delivery
// network can briefly exceed the metadata budget. The per-read timeout bounds
// stalls without killing a legitimately long stream.
const PAYLOAD_READ_TIMEOUT: Duration = Duration::from_secs(30);
const MAXIMUM_REDIRECT_COUNT: usize = 5;

/// HTTPS-only reqwest transport with bounded Hugging Face redirects and response buffering.
#[derive(Clone, Debug)]
pub struct ReqwestHubTransport {
    metadata_client: Client,
    payload_client: Client,
}

/// Failure while constructing the production Hub HTTP client.
#[derive(Debug, Error)]
#[error("failed to construct the Hugging Face HTTP client: {source}")]
pub struct ReqwestHubTransportBuildError {
    #[source]
    source: reqwest::Error,
}

impl ReqwestHubTransport {
    pub fn production() -> Result<Self, ReqwestHubTransportBuildError> {
        let redirect_policy = redirect::Policy::custom(|attempt| {
            if attempt.previous().len() >= MAXIMUM_REDIRECT_COUNT {
                return attempt.error("Hugging Face redirect limit exceeded");
            }
            let original_url = attempt.previous().first();
            if is_trusted_hugging_face_url(attempt.url())
                && original_url.is_some_and(|original_url| {
                    attempt.url().path() == original_url.path()
                        && attempt.url().query() == original_url.query()
                })
            {
                attempt.follow()
            } else {
                attempt.error("Hugging Face redirect left the trusted HTTPS origin")
            }
        });
        let metadata_client = Client::builder()
            .https_only(true)
            .redirect(redirect_policy)
            .connect_timeout(CONNECT_TIMEOUT)
            .read_timeout(READ_TIMEOUT)
            .build()
            .map_err(|source| ReqwestHubTransportBuildError { source })?;
        // Public immutable payload endpoints legitimately redirect to provider-controlled content
        // delivery hosts, so payload redirects preserve HTTPS without imposing the metadata origin.
        let payload_redirect_policy = redirect::Policy::custom(|attempt| {
            if attempt.previous().len() >= MAXIMUM_REDIRECT_COUNT {
                return attempt.error("Hugging Face payload redirect limit exceeded");
            }
            if is_safe_https_url(attempt.url()) {
                attempt.follow()
            } else {
                attempt.error("Hugging Face payload redirect must remain on HTTPS")
            }
        });
        let payload_client = Client::builder()
            .https_only(true)
            .redirect(payload_redirect_policy)
            .connect_timeout(CONNECT_TIMEOUT)
            .read_timeout(PAYLOAD_READ_TIMEOUT)
            .build()
            .map_err(|source| ReqwestHubTransportBuildError { source })?;
        Ok(Self {
            metadata_client,
            payload_client,
        })
    }
}

impl HubTransport for ReqwestHubTransport {
    fn execute(&self, request: HubHttpRequest) -> HubTransportFuture<'_> {
        Box::pin(async move {
            let request_url = Url::parse(request.url()).map_err(|source| {
                HubTransportError::with_source("Hub request URL is invalid", source)
            })?;
            if !is_trusted_hugging_face_url(&request_url) {
                return Err(HubTransportError::new(
                    "Hub request must use the trusted Hugging Face HTTPS origin",
                ));
            }
            let mut request_builder = match request.method() {
                HubHttpMethod::Get => self.metadata_client.get(request_url),
            }
            .timeout(request.metadata_timeout());
            for (header_name, header_value) in request.headers() {
                request_builder = request_builder.header(*header_name, *header_value);
            }

            let response = request_builder
                .send()
                .await
                .map_err(|source| HubTransportError::with_source("Hub request failed", source))?;
            let status = response.status().as_u16();
            let selected_headers = selected_response_headers(&response)?;
            if response
                .content_length()
                .is_some_and(|content_length| content_length > MAXIMUM_RESPONSE_BODY_BYTES as u64)
            {
                return Err(HubTransportError::new(
                    "Hub response exceeds the metadata body byte limit",
                ));
            }

            let mut body_chunks = Vec::new();
            let mut body_byte_count = 0_usize;
            let mut response_stream = response.bytes_stream();
            while let Some(body_chunk) = response_stream.next().await {
                let body_chunk = body_chunk.map_err(|source| {
                    HubTransportError::with_source("Hub response body read failed", source)
                })?;
                body_byte_count = body_byte_count
                    .checked_add(body_chunk.len())
                    .ok_or_else(|| HubTransportError::new("Hub response body size overflowed"))?;
                if body_byte_count > MAXIMUM_RESPONSE_BODY_BYTES
                    || body_chunks.len() >= MAXIMUM_RESPONSE_BODY_CHUNKS
                {
                    return Err(HubTransportError::new(
                        "Hub response exceeds the metadata body limit",
                    ));
                }
                body_chunks.push(body_chunk.to_vec());
            }

            HubHttpResponse::try_new(status, selected_headers, body_chunks).map_err(|source| {
                HubTransportError::with_source("Hub response failed bounded validation", source)
            })
        })
    }
}

impl HubPayloadTransport for ReqwestHubTransport {
    fn execute_payload(&self, request: HubPayloadRequest) -> HubPayloadFuture<'_> {
        Box::pin(async move {
            let request_url = Url::parse(request.url()).map_err(|source| {
                HubTransportError::with_source("Hub payload URL is invalid", source)
            })?;
            if !is_trusted_hugging_face_url(&request_url) {
                return Err(HubTransportError::new(
                    "Hub payload must start at the trusted Hugging Face HTTPS origin",
                ));
            }
            // Per-read timeouts on the client builder bound idle gaps without killing a
            // multi-gigabyte transfer that legitimately streams for several minutes.
            let mut request_builder = self
                .payload_client
                .get(request_url)
                .header(ACCEPT_ENCODING, "identity")
                .header(
                    USER_AGENT,
                    concat!("Astronomical/", env!("CARGO_PKG_VERSION")),
                );
            if request.resume_offset_bytes() > 0 {
                request_builder = request_builder
                    .header(RANGE, format!("bytes={}-", request.resume_offset_bytes()));
            }
            let response = request_builder.send().await.map_err(|source| {
                HubTransportError::with_source("Hub payload request failed", source)
            })?;
            let status = response.status().as_u16();
            let content_length = response.content_length();
            let content_range = response
                .headers()
                .get(CONTENT_RANGE)
                .map(|header| {
                    header.to_str().map(str::to_owned).map_err(|source| {
                        HubTransportError::with_source(
                            "Hub payload Content-Range is not valid text",
                            source,
                        )
                    })
                })
                .transpose()?;
            let byte_stream = response.bytes_stream().map(|payload_bytes| {
                payload_bytes.map_err(|source| {
                    HubTransportError::with_source("Hub payload body read failed", source)
                })
            });
            Ok(HubPayloadResponse::new(
                status,
                content_range,
                content_length,
                Box::pin(byte_stream),
            ))
        })
    }
}

fn selected_response_headers(
    response: &reqwest::Response,
) -> Result<Vec<(String, String)>, HubTransportError> {
    let mut combined_link_header = String::new();
    for link_header in response.headers().get_all(LINK) {
        let link_header = link_header.to_str().map_err(|source| {
            HubTransportError::with_source("Hub Link header is not valid text", source)
        })?;
        let separator_bytes = usize::from(!combined_link_header.is_empty());
        let combined_length = combined_link_header
            .len()
            .checked_add(separator_bytes)
            .and_then(|length| length.checked_add(link_header.len()))
            .ok_or_else(|| HubTransportError::new("Hub Link header size overflowed"))?;
        if combined_length > MAXIMUM_HEADER_VALUE_BYTES {
            return Err(HubTransportError::new(
                "Hub Link header exceeds the metadata header limit",
            ));
        }
        if separator_bytes == 1 {
            combined_link_header.push(',');
        }
        combined_link_header.push_str(link_header);
    }
    if combined_link_header.is_empty() {
        Ok(Vec::new())
    } else {
        Ok(vec![("link".to_owned(), combined_link_header)])
    }
}

fn is_trusted_hugging_face_url(url: &Url) -> bool {
    is_safe_https_url(url) && url.host_str() == Some(HUGGING_FACE_HOST)
}

fn is_safe_https_url(url: &Url) -> bool {
    url.scheme() == "https"
        && url.host_str().is_some()
        && url.port().is_none()
        && url.username().is_empty()
        && url.password().is_none()
}
