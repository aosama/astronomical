//! Shared fixtures keep payload acceptance cases focused on externally visible transfer behavior.

use std::{collections::VecDeque, path::Path, sync::Mutex};

use astronomical_supervisor::{
    DownloadJob, DownloadPublicationRefresh, HubPayloadFuture, HubPayloadRequest,
    HubPayloadResponse, HubPayloadTransport, HubTransportError,
    SupervisorPerformanceAttributionLog,
};
use bytes::Bytes;
use futures_util::stream;
use sha1::Sha1;
use sha2::{Digest, Sha256};
use tempfile::TempDir;

use super::{RELATIVE_PATH, REPOSITORY_ID, REVISION};

pub(super) struct ScriptedPayloadTransport {
    requests: Mutex<Vec<HubPayloadRequest>>,
    responses: Mutex<VecDeque<HubPayloadResponse>>,
}

impl ScriptedPayloadTransport {
    pub(super) fn new(responses: impl IntoIterator<Item = HubPayloadResponse>) -> Self {
        Self {
            requests: Mutex::new(Vec::new()),
            responses: Mutex::new(responses.into_iter().collect()),
        }
    }

    pub(super) fn requests(&self) -> Vec<HubPayloadRequest> {
        self.requests
            .lock()
            .expect("request lock should remain available")
            .clone()
    }
}

impl HubPayloadTransport for ScriptedPayloadTransport {
    fn execute_payload(&self, request: HubPayloadRequest) -> HubPayloadFuture<'_> {
        Box::pin(async move {
            self.requests
                .lock()
                .map_err(|_| HubTransportError::new("request lock was poisoned"))?
                .push(request);
            self.responses
                .lock()
                .map_err(|_| HubTransportError::new("response lock was poisoned"))?
                .pop_front()
                .ok_or_else(|| HubTransportError::new("unexpected payload request"))
        })
    }
}

#[derive(Default)]
pub(super) struct RecordingRefresh(Mutex<Vec<std::path::PathBuf>>);

impl RecordingRefresh {
    pub(super) fn refreshed_directories(&self) -> Vec<std::path::PathBuf> {
        self.0
            .lock()
            .expect("refresh lock should remain available")
            .clone()
    }
}

impl DownloadPublicationRefresh for RecordingRefresh {
    fn refresh(
        &self,
        published_directory: &Path,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.0
            .lock()
            .map_err(|_| "refresh lock was poisoned")?
            .push(published_directory.to_path_buf());
        Ok(())
    }
}

pub(super) struct FailingRefresh;

impl DownloadPublicationRefresh for FailingRefresh {
    fn refresh(
        &self,
        _published_directory: &Path,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Err("scripted discovery refresh failure".into())
    }
}

pub(super) fn payload_response<const CHUNK_COUNT: usize>(
    status: u16,
    content_range: Option<&str>,
    payload_chunks: [&[u8]; CHUNK_COUNT],
) -> HubPayloadResponse {
    let content_length = payload_chunks
        .iter()
        .map(|payload_chunk| payload_chunk.len() as u64)
        .sum();
    let owned_payload_chunks = payload_chunks
        .into_iter()
        .map(Bytes::copy_from_slice)
        .collect::<Vec<_>>();
    HubPayloadResponse::new(
        status,
        content_range.map(str::to_owned),
        Some(content_length),
        Box::pin(stream::iter(owned_payload_chunks.into_iter().map(Ok))),
    )
}

pub(super) fn sha256_job(
    state: &str,
    bytes_on_disk: u64,
    digest: String,
    error_code: Option<&str>,
) -> DownloadJob {
    parse_job(state, bytes_on_disk, "sha256", &digest, error_code)
}

pub(super) fn git_blob_job(digest: String) -> DownloadJob {
    parse_job("paused", 0, "git_blob_sha1", &digest, None)
}

fn parse_job(
    state: &str,
    bytes_on_disk: u64,
    algorithm: &str,
    digest: &str,
    error_code: Option<&str>,
) -> DownloadJob {
    parse_job_with_size(state, bytes_on_disk, 16, algorithm, digest, error_code)
}

pub(super) fn parse_job_with_size(
    state: &str,
    bytes_on_disk: u64,
    expected_bytes: u64,
    algorithm: &str,
    digest: &str,
    error_code: Option<&str>,
) -> DownloadJob {
    let error_json = error_code.map_or_else(|| "null".to_owned(), |error| format!("\"{error}\""));
    DownloadJob::parse_json(&format!(
        "{{\"huggingface_id\":\"{REPOSITORY_ID}\",\"revision\":\"{REVISION}\",\"state\":\"{state}\",\"bytes_completed\":{bytes_on_disk},\"bytes_total\":{expected_bytes},\"current_file_relative_path\":null,\"files\":[{{\"relative_path\":\"{RELATIVE_PATH}\",\"expected_bytes\":{expected_bytes},\"expected_digest\":{{\"algorithm\":\"{algorithm}\",\"hex\":\"{digest}\"}},\"bytes_on_disk\":{bytes_on_disk}}}],\"error_code\":{error_json},\"updated_at\":100}}"
    ))
    .expect("fixture job should be valid")
}

pub(super) fn sha256_hex(payload: &[u8]) -> String {
    lowercase_hex(Sha256::digest(payload).as_ref())
}

pub(super) fn git_blob_sha1_hex(payload: &[u8]) -> String {
    let mut digest = Sha1::new();
    digest.update(format!("blob {}\0", payload.len()).as_bytes());
    digest.update(payload);
    lowercase_hex(digest.finalize().as_ref())
}

fn lowercase_hex(digest_bytes: &[u8]) -> String {
    const HEX_CHARACTERS: &[u8; 16] = b"0123456789abcdef";
    let mut hexadecimal_digest = String::with_capacity(digest_bytes.len() * 2);
    for digest_byte in digest_bytes {
        hexadecimal_digest.push(HEX_CHARACTERS[(digest_byte >> 4) as usize] as char);
        hexadecimal_digest.push(HEX_CHARACTERS[(digest_byte & 0x0f) as usize] as char);
    }
    hexadecimal_digest
}

pub(super) fn staged_file_path(models_directory: &Path) -> std::path::PathBuf {
    models_directory.join(format!(".incomplete/{REPOSITORY_ID}/{RELATIVE_PATH}"))
}

pub(super) fn disabled_attribution(
    test_directory: &TempDir,
) -> SupervisorPerformanceAttributionLog {
    SupervisorPerformanceAttributionLog::open(test_directory.path(), false)
        .expect("disabled attribution should construct")
}
