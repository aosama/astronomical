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

pub(super) fn validate_payload_response(
    status: u16,
    content_range: Option<&str>,
    content_length: Option<u64>,
    resume_offset_bytes: u64,
    expected_file_bytes: u64,
) -> Result<(), DownloadPayloadTransferError> {
    let expected_transfer_bytes = expected_file_bytes - resume_offset_bytes;
    if content_length.is_some_and(|length| length != expected_transfer_bytes) {
        return Err(DownloadPayloadTransferError::InvalidPayloadLength);
    }
    if resume_offset_bytes == 0 && status == 200 && content_range.is_none() {
        return Ok(());
    }
    if status != 206 {
        return Err(DownloadPayloadTransferError::InvalidRangeResponse);
    }
    let expected_range = format!(
        "bytes {}-{}/{}",
        resume_offset_bytes,
        expected_file_bytes - 1,
        expected_file_bytes
    );
    if content_range != Some(expected_range.as_str()) {
        return Err(DownloadPayloadTransferError::InvalidRangeResponse);
    }
    Ok(())
}
