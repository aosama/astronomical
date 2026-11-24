//! Reads a public Hugging Face repository over HTTPS. The listing is fetched once
//! and every later read names the commit it returned, so one inspection can
//! never mix files from two revisions if the repository moves underneath it.

use std::time::Duration;

use reqwest::header::{CONTENT_RANGE, RANGE, USER_AGENT};
use reqwest::{Client, Response, StatusCode, Url, redirect};
use serde::Deserialize;

use crate::artifact_source::{ArtifactFile, ArtifactListing};
use crate::error::VisionTowerError;

const HUGGING_FACE_ORIGIN: &str = "https://huggingface.co";
const DEFAULT_REVISION: &str = "main";
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const READ_TIMEOUT: Duration = Duration::from_secs(30);
const MAXIMUM_REDIRECT_COUNT: usize = 5;
const MAXIMUM_LISTING_BYTES: u64 = 16 * 1024 * 1024;
const CLIENT_NAME: &str = concat!("Astronomical/", env!("CARGO_PKG_VERSION"));

#[derive(Debug)]
pub struct HuggingFaceSource {
    client: Client,
    repository_id: String,
    listing: ArtifactListing,
}

#[derive(Deserialize)]
struct ModelReply {
    sha: String,
    siblings: Vec<SiblingReply>,
    #[serde(rename = "cardData")]
    card_data: Option<CardReply>,
    pipeline_tag: Option<String>,
}

#[derive(Deserialize)]
struct SiblingReply {
    rfilename: String,
    size: Option<u64>,
}

#[derive(Deserialize)]
struct CardReply {
    license: Option<serde_json::Value>,
}

impl HuggingFaceSource {
    pub async fn open(
        repository_id: &str,
        revision: Option<&str>,
    ) -> Result<Self, VisionTowerError> {
        let client = build_client()?;
        let requested_revision = revision.unwrap_or(DEFAULT_REVISION);
        let listing_url = listing_url(repository_id, requested_revision)?;
        let listing_reply = send_request(&client, &listing_url, None).await?;
        match listing_reply.status() {
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN | StatusCode::NOT_FOUND => {
                return Err(VisionTowerError::RepositoryNotFound {
                    repository_id: repository_id.to_owned(),
                    revision: requested_revision.to_owned(),
                });
            }
            StatusCode::OK => {}
            other_status => return Err(unexpected_status(&listing_url, other_status)),
        }
        let listing_bytes =
            read_bounded_body(listing_reply, &listing_url, MAXIMUM_LISTING_BYTES).await?;
        Ok(Self {
            client,
            repository_id: repository_id.to_owned(),
            listing: listing_from_reply(repository_id, &listing_bytes)?,
        })
    }

    #[must_use]
    pub const fn listing(&self) -> &ArtifactListing {
        &self.listing
    }

    pub async fn read_whole_file(
        &self,
        relative_path: &str,
        maximum_bytes: u64,
    ) -> Result<Vec<u8>, VisionTowerError> {
        let file_url = self.pinned_file_url(relative_path)?;
        let file_reply = send_request(&self.client, &file_url, None).await?;
        if file_reply.status() != StatusCode::OK {
            return Err(unexpected_status(&file_url, file_reply.status()));
        }
        read_bounded_body(file_reply, &file_url, maximum_bytes).await
    }

    pub async fn read_byte_range(
        &self,
        relative_path: &str,
        start_byte: u64,
        byte_count: u64,
    ) -> Result<Vec<u8>, VisionTowerError> {
        if byte_count == 0 {
            return Ok(Vec::new());
        }
        let file_url = self.pinned_file_url(relative_path)?;
        let last_byte = start_byte + byte_count - 1;
        let range_header = format!("bytes={start_byte}-{last_byte}");
        let range_reply = send_request(&self.client, &file_url, Some(&range_header)).await?;
        // A server that ignores the range would start sending the whole weight
        // file, which is the download this crate exists to avoid.
        if range_reply.status() != StatusCode::PARTIAL_CONTENT {
            return Err(VisionTowerError::RangeNotHonored {
                url: file_url.to_string(),
                description: format!("status {}", range_reply.status().as_u16()),
            });
        }
        let content_range = match range_reply.headers().get(CONTENT_RANGE) {
            Some(header_value) => header_value.to_str().unwrap_or_default().to_owned(),
            None => String::new(),
        };
        if !content_range_matches(&content_range, start_byte, last_byte) {
            return Err(VisionTowerError::RangeNotHonored {
                url: file_url.to_string(),
                description: format!("Content-Range was {content_range:?}"),
            });
        }
        let range_bytes = read_bounded_body(range_reply, &file_url, byte_count).await?;
        if range_bytes.len() as u64 != byte_count {
            return Err(VisionTowerError::RangeNotHonored {
                url: file_url.to_string(),
                description: format!("{} of {byte_count} bytes arrived", range_bytes.len()),
            });
        }
        Ok(range_bytes)
    }

    fn pinned_file_url(&self, relative_path: &str) -> Result<Url, VisionTowerError> {
        let pinned_revision = self
            .listing
            .pinned_revision
            .as_deref()
            .unwrap_or(DEFAULT_REVISION);
        resolve_url(&self.repository_id, pinned_revision, relative_path)
    }
}

fn build_client() -> Result<Client, VisionTowerError> {
    // The workspace builds reqwest with `rustls-no-provider`, so clients refuse to
    // build until a provider exists. Installing twice is a harmless no-op.
    let _ = rustls::crypto::ring::default_provider().install_default();
    let built_client = Client::builder()
        .https_only(true)
        .redirect(redirect::Policy::limited(MAXIMUM_REDIRECT_COUNT))
        .connect_timeout(CONNECT_TIMEOUT)
        .read_timeout(READ_TIMEOUT)
        .build();
    match built_client {
        Ok(client) => Ok(client),
        Err(build_error) => Err(VisionTowerError::HttpClientUnavailable {
            cause: build_error.to_string(),
        }),
    }
}

async fn send_request(
    client: &Client,
    url: &Url,
    range_header: Option<&str>,
) -> Result<Response, VisionTowerError> {
    let mut request_builder = client.get(url.clone()).header(USER_AGENT, CLIENT_NAME);
    if let Some(range_header) = range_header {
        request_builder = request_builder.header(RANGE, range_header);
    }
    match request_builder.send().await {
        Ok(reply) => Ok(reply),
        Err(send_error) => Err(request_failed(url, &send_error)),
    }
}

/// Streams with a cap, because a declared length can lie and an unbounded read of
/// a mistaken URL could be many gigabytes.
async fn read_bounded_body(
    mut reply: Response,
    url: &Url,
    maximum_bytes: u64,
) -> Result<Vec<u8>, VisionTowerError> {
    let mut body_bytes = Vec::new();
    loop {
        let body_piece = match reply.chunk().await {
            Ok(body_piece) => body_piece,
            Err(read_error) => return Err(request_failed(url, &read_error)),
        };
        let Some(body_piece) = body_piece else {
            return Ok(body_bytes);
        };
        if (body_bytes.len() + body_piece.len()) as u64 > maximum_bytes {
            return Err(VisionTowerError::ReplyTooLarge {
                url: url.to_string(),
                maximum_bytes,
            });
        }
        body_bytes.extend_from_slice(&body_piece);
    }
}

fn request_failed(url: &Url, request_error: &reqwest::Error) -> VisionTowerError {
    VisionTowerError::HttpRequestFailed {
        url: url.to_string(),
        cause: request_error.to_string(),
    }
}

fn unexpected_status(url: &Url, status: StatusCode) -> VisionTowerError {
    VisionTowerError::UnexpectedHttpStatus {
        url: url.to_string(),
        status: status.as_u16(),
    }
}

fn listing_url(repository_id: &str, revision: &str) -> Result<Url, VisionTowerError> {
    let mut url = url_from_segments(&["api", "models"], repository_id, &["revision", revision])?;
    url.set_query(Some("blobs=true"));
    Ok(url)
}

/// Built by path segment so a file name can never change the URL's structure.
fn resolve_url(
    repository_id: &str,
    revision: &str,
    relative_path: &str,
) -> Result<Url, VisionTowerError> {
    let mut trailing_segments = vec!["resolve", revision];
    for path_segment in relative_path.split('/') {
        trailing_segments.push(path_segment);
    }
    url_from_segments(&[], repository_id, &trailing_segments)
}

fn url_from_segments(
    leading_segments: &[&str],
    repository_id: &str,
    trailing_segments: &[&str],
) -> Result<Url, VisionTowerError> {
    let mut path_segments = leading_segments.to_vec();
    for repository_segment in repository_id.split('/') {
        path_segments.push(repository_segment);
    }
    path_segments.extend_from_slice(trailing_segments);
    let mut url = Url::parse(HUGGING_FACE_ORIGIN).expect("the origin is a valid URL");
    url.path_segments_mut()
        .expect("an HTTPS origin has path segments")
        .extend(path_segments);
    Ok(url)
}

/// Accepts `bytes START-END/TOTAL` where the total may be `*`.
fn content_range_matches(content_range: &str, start_byte: u64, last_byte: u64) -> bool {
    let Some(range_text) = content_range.strip_prefix("bytes ") else {
        return false;
    };
    let Some((served_range, _total_size)) = range_text.split_once('/') else {
        return false;
    };
    served_range == format!("{start_byte}-{last_byte}")
}

fn listing_from_reply(
    repository_id: &str,
    listing_bytes: &[u8],
) -> Result<ArtifactListing, VisionTowerError> {
    let model_reply: ModelReply = match serde_json::from_slice(listing_bytes) {
        Ok(model_reply) => model_reply,
        Err(parse_error) => {
            return Err(VisionTowerError::RepositoryListingMalformed {
                description: parse_error.to_string(),
            });
        }
    };
    let mut files = Vec::with_capacity(model_reply.siblings.len());
    for sibling in model_reply.siblings {
        if is_safe_relative_path(&sibling.rfilename) {
            files.push(ArtifactFile {
                relative_path: sibling.rfilename,
                byte_count: sibling.size,
            });
        }
    }
    Ok(ArtifactListing {
        display_name: repository_id.to_owned(),
        origin_repository: Some(repository_id.to_owned()),
        pinned_revision: Some(model_reply.sha),
        license: license_text(model_reply.card_data),
        pipeline_tag: model_reply.pipeline_tag,
        files,
    })
}

/// A model card's license is usually one string, and sometimes a list.
fn license_text(card_data: Option<CardReply>) -> Option<String> {
    let license_value = card_data?.license?;
    match license_value {
        serde_json::Value::String(license) => Some(license),
        serde_json::Value::Array(licenses) => {
            let mut license_names = Vec::with_capacity(licenses.len());
            for license in &licenses {
                license_names.push(license.as_str()?);
            }
            Some(license_names.join(", "))
        }
        _ => None,
    }
}

/// Repository listings are outside data, so a name that could climb out of the
/// repository's path is dropped rather than requested.
fn is_safe_relative_path(relative_path: &str) -> bool {
    for path_segment in relative_path.split('/') {
        if path_segment.is_empty() || path_segment == "." || path_segment == ".." {
            return false;
        }
    }
    true
}

#[cfg(test)]
#[path = "hugging_face_source_tests.rs"]
mod tests;
