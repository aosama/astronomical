//! Shared fixtures keep payload acceptance cases focused on externally visible transfer behavior.

use std::{
    collections::{BTreeMap, VecDeque},
    path::Path,
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

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

pub(super) const RELATIVE_PATH: &str = "weights/romeo-and-juliet.txt";
pub(super) const REPOSITORY_ID: &str = "astronomical-test/example-qwen";
pub(super) const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

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

pub(super) type ScriptedResponseFactory = Box<dyn Fn() -> HubPayloadResponse + Send + Sync>;

/// Serves one scripted response factory per relative file path and tracks how many payload
/// requests are concurrently in flight, so a journey can prove its parallelism stayed bounded.
pub(super) struct PathKeyedPayloadTransport {
    requests: Mutex<Vec<HubPayloadRequest>>,
    active_request_count: AtomicUsize,
    maximum_active_request_count: AtomicUsize,
    response_factories_by_relative_path: BTreeMap<String, ScriptedResponseFactory>,
}

impl PathKeyedPayloadTransport {
    pub(super) fn new(
        response_factories_by_relative_path: impl IntoIterator<Item = (String, ScriptedResponseFactory)>,
    ) -> Self {
        Self {
            requests: Mutex::new(Vec::new()),
            active_request_count: AtomicUsize::new(0),
            maximum_active_request_count: AtomicUsize::new(0),
            response_factories_by_relative_path: response_factories_by_relative_path
                .into_iter()
                .collect(),
        }
    }

    pub(super) fn requests(&self) -> Vec<HubPayloadRequest> {
        self.requests
            .lock()
            .expect("request lock should remain available")
            .clone()
    }

    pub(super) fn maximum_active_request_count(&self) -> usize {
        self.maximum_active_request_count.load(Ordering::Acquire)
    }
}

impl HubPayloadTransport for PathKeyedPayloadTransport {
    fn execute_payload(&self, request: HubPayloadRequest) -> HubPayloadFuture<'_> {
        Box::pin(async move {
            self.requests
                .lock()
                .map_err(|_| HubTransportError::new("request lock was poisoned"))?
                .push(request.clone());
            let observed_active_count =
                self.active_request_count.fetch_add(1, Ordering::AcqRel) + 1;
            self.maximum_active_request_count
                .fetch_max(observed_active_count, Ordering::AcqRel);
            // A brief overlap window lets concurrently launched transfers register before any of
            // them completes, so the recorded maximum reflects the scheduler's true in-flight width.
            tokio::time::sleep(Duration::from_millis(20)).await;
            self.active_request_count.fetch_sub(1, Ordering::AcqRel);
            let relative_path = payload_relative_path(request.url())?;
            let response_factory = self
                .response_factories_by_relative_path
                .get(&relative_path)
                .ok_or_else(|| {
                    HubTransportError::new("unexpected payload request for this file")
                })?;
            Ok(response_factory())
        })
    }
}

pub(super) fn payload_relative_path_for_request(payload_url: &str) -> String {
    let revision_marker = format!("/resolve/{REVISION}/");
    let (_, relative_path) = payload_url
        .split_once(&revision_marker)
        .expect("payload request URL should address the scripted revision");
    relative_path.to_owned()
}

fn payload_relative_path(payload_url: &str) -> Result<String, HubTransportError> {
    let revision_marker = format!("/resolve/{REVISION}/");
    let (_, relative_path) = payload_url
        .split_once(&revision_marker)
        .ok_or_else(|| HubTransportError::new("payload URL must address the scripted revision"))?;
    Ok(relative_path.to_owned())
}

/// Complete remaining payload after an interrupted prefix, framed as a valid ranged response.
pub(super) fn resume_payload_response(
    complete_payload: &[u8],
    resume_offset_bytes: u64,
) -> HubPayloadResponse {
    let remaining_payload = &complete_payload[resume_offset_bytes as usize..];
    let content_range = format!(
        "bytes {resume_offset_bytes}-{last_byte_index}/{total_bytes}",
        last_byte_index = complete_payload.len() - 1,
        total_bytes = complete_payload.len()
    );
    payload_response(206, Some(&content_range), [remaining_payload])
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

/// Durable paused job with one file per provided relative path, sized to its payload.
pub(super) fn multi_file_paused_job(file_payloads: &[(&str, &[u8])]) -> DownloadJob {
    let mut files_json = String::new();
    let mut bytes_total = 0_u64;
    for (file_index, (relative_path, payload_bytes)) in file_payloads.into_iter().enumerate() {
        if file_index > 0 {
            files_json.push(',');
        }
        bytes_total += payload_bytes.len() as u64;
        files_json.push_str(&format!(
            "{{\"relative_path\":\"{relative_path}\",\"expected_bytes\":{expected_bytes},\"expected_digest\":{{\"algorithm\":\"sha256\",\"hex\":\"{digest}\"}},\"bytes_on_disk\":0}}",
            expected_bytes = payload_bytes.len(),
            digest = sha256_hex(&payload_bytes),
        ));
    }
    DownloadJob::parse_json(&format!(
        "{{\"huggingface_id\":\"{REPOSITORY_ID}\",\"revision\":\"{REVISION}\",\"state\":\"paused\",\"bytes_completed\":0,\"bytes_total\":{bytes_total},\"current_file_relative_path\":null,\"files\":[{files_json}],\"error_code\":null,\"updated_at\":100}}"
    ))
    .expect("multi-file fixture job should be valid")
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

pub(super) fn staged_file_path_for_relative_path(
    models_directory: &Path,
    relative_path: &str,
) -> std::path::PathBuf {
    models_directory
        .join(".incomplete")
        .join(REPOSITORY_ID)
        .join(relative_path)
}

pub(super) fn disabled_attribution(
    test_directory: &TempDir,
) -> SupervisorPerformanceAttributionLog {
    SupervisorPerformanceAttributionLog::open(test_directory.path(), false)
        .expect("disabled attribution should construct")
}
