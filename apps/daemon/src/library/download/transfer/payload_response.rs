//! Immutable provider URL construction and ranged-response framing validation.

use reqwest::Url;

use super::DownloadPayloadTransferError;

const HUGGING_FACE_ORIGIN: &str = "https://huggingface.co";

pub(super) fn payload_url(
    repository_id: &str,
    revision: &str,
    relative_path: &str,
) -> Result<String, DownloadPayloadTransferError> {
    let mut url = Url::parse(HUGGING_FACE_ORIGIN)
        .map_err(|_| DownloadPayloadTransferError::InvalidJobState)?;
    let mut path_segments = url
        .path_segments_mut()
        .map_err(|_| DownloadPayloadTransferError::InvalidJobState)?;
    for repository_component in repository_id.split('/') {
        path_segments.push(repository_component);
    }
    path_segments.push("resolve").push(revision);
    for file_component in relative_path.split('/') {
        path_segments.push(file_component);
    }
    drop(path_segments);
    Ok(url.into())
}

/// Accepts the three honest answers to a request for the bytes from
/// `resume_offset_bytes` up to `end_offset_bytes` of a file: exactly that
/// range; the rest of the file from the same start, which servers that ignore
/// an upper bound send, and of which the caller consumes only its share; or
/// the whole file with `200` when the request started at byte zero.
pub(super) fn validate_payload_response(
    status: u16,
    content_range: Option<&str>,
    content_length: Option<u64>,
    (resume_offset_bytes, end_offset_bytes): (u64, u64),
    expected_file_bytes: u64,
) -> Result<(), DownloadPayloadTransferError> {
    let remaining_file_bytes = expected_file_bytes - resume_offset_bytes;
    if resume_offset_bytes == 0 && status == 200 && content_range.is_none() {
        return require_content_length(content_length, remaining_file_bytes);
    }
    if status != 206 {
        return Err(DownloadPayloadTransferError::InvalidRangeResponse);
    }
    let exact_range = format!(
        "bytes {resume_offset_bytes}-{}/{expected_file_bytes}",
        end_offset_bytes - 1
    );
    let rest_of_file_range = format!(
        "bytes {resume_offset_bytes}-{}/{expected_file_bytes}",
        expected_file_bytes - 1
    );
    if content_range == Some(exact_range.as_str()) {
        return require_content_length(content_length, end_offset_bytes - resume_offset_bytes);
    }
    if content_range == Some(rest_of_file_range.as_str()) {
        return require_content_length(content_length, remaining_file_bytes);
    }
    Err(DownloadPayloadTransferError::InvalidRangeResponse)
}

/// A declared length must match the bytes the response's range promises; an
/// undeclared one is checked later against the bytes that actually arrive.
fn require_content_length(
    content_length: Option<u64>,
    expected_response_bytes: u64,
) -> Result<(), DownloadPayloadTransferError> {
    if content_length.is_some_and(|declared_bytes| declared_bytes != expected_response_bytes) {
        return Err(DownloadPayloadTransferError::InvalidPayloadLength);
    }
    Ok(())
}

#[cfg(test)]
#[path = "payload_response_tests.rs"]
mod payload_response_tests;
