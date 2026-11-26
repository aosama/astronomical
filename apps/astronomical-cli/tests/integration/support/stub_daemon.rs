//! Configurable stub daemon speaking the real framed protocol over a real
//! unix socket, shared by the one-shot verb journey tests.
//!
//! It mirrors the daemon's lifecycle surface: status, installed models,
//! catalog, download start/status, default-model persistence, and scripted
//! chat or embeddings results. One connection serves a scripted exchange
//! sequence, like the real daemon.

mod stub_connection;

use super::stub_decisions::scripted_decisions_response;
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};

use decision_contract::{DecisionFailure, QuestionOutcome};
use ipc_protocol::{
    ChatGenerationSettings, ChatImageInput, ChatMessage, DaemonDownloadJob, DaemonRequest,
    DaemonResponse, DaemonWorkerStatus, EmbeddingsFailureReason, GeneratedImage,
    ImageGenerateRequest, ImageGenerationFailureReason, ImageGenerationResultMetadata,
    ModelLoadInFlight, ProtocolWriter,
};
use stub_connection::{StubDaemonState, serve_stub_connection};

pub use super::stub_inventory::{StubCatalogEntry, StubInstalledModel};

/// Scripted result of an image request.
pub enum StubImageOutcome {
    /// The daemon answers with these PNG bytes.
    Completed(Vec<u8>),
    /// The daemon refuses before admission, with this reason.
    Rejected(String),
    /// The worker fails the image after admission.
    Failed(ImageGenerationFailureReason),
    /// The daemon accepts the request and never answers (a wedged worker).
    Stalls,
}

/// Scripted result of an embeddings request.
pub enum StubEmbeddingsOutcome {
    Completed,
    ContextLengthExceeded,
}

/// Scripted result of a decision request.
pub enum StubDecisionsOutcome {
    /// The daemon answers with these outcomes, one per question.
    Completed(Vec<QuestionOutcome>),
    /// The worker fails the request after admission.
    Failed(DecisionFailure),
    /// The daemon refuses before admission, with this reason.
    Rejected(String),
}

/// Scripted result of a speech request.
pub enum StubSpeechOutcome {
    /// The daemon answers with these samples, split over two audio frames.
    Completed(Vec<u8>),
    /// The worker fails the request after admission.
    Failed(speech_contract::SpeechFailure),
}

/// Scripted behaviour of the stub daemon for one test.
pub struct StubDaemonConfig {
    pub worker_status: DaemonWorkerStatus,
    pub installed_models: Vec<StubInstalledModel>,
    pub catalog_entries: Vec<StubCatalogEntry>,
    /// Replaces `catalog_entries` for `Catalog` requests once a download has
    /// been started — how the daemon reports a just-finished download.
    pub catalog_entries_after_download: Option<Vec<StubCatalogEntry>>,
    pub default_model_id: Option<String>,
    pub chat_fragments: Vec<String>,
    /// How long a `ChatGenerate` stays silent before its first frame, standing in
    /// for a model load inside the real daemon.
    pub chat_first_frame_delay: Duration,
    /// FIFO of load reports for `Status` requests, where `None` means the load has
    /// ended; the last entry repeats once the scripted list is exhausted. Empty means
    /// nothing is loading.
    pub model_load_reports: Vec<Option<ModelLoadInFlight>>,
    /// FIFO of progress counts for `Status` requests; the last entry repeats the way
    /// a stuck worker's count stays put. Empty reports zero.
    pub request_progress_counts: Vec<u64>,
    /// Which `Status` requests, counted from zero across every connection, are read
    /// and never answered.
    pub hung_status_requests: std::ops::Range<usize>,
    pub neural_engine_prefill: Option<ipc_protocol::NeuralEnginePrefillStatus>,
    /// The drafter status every `Status` reports.
    pub draft_model: Option<ipc_protocol::DraftModelStatus>,
    /// What every `ModelUnload` answers; an unload names the resident model.
    pub model_unload_status: ipc_protocol::DaemonModelUnloadStatus,
    pub embeddings_outcome: StubEmbeddingsOutcome,
    /// How long an `EmbedGenerate` stays silent before its one frame, standing in for
    /// a queue or a model load inside the real daemon.
    pub embeddings_first_frame_delay: Duration,
    pub embedding_vector: Vec<f32>,
    /// What the daemon does with a `DecideGenerate` request.
    pub decisions_outcome: StubDecisionsOutcome,
    /// Captures the most recent `DecideGenerate` request the CLI sent, so a
    /// test can assert the state and questions that crossed the socket.
    pub decide_request_capture: Arc<std::sync::Mutex<Option<DaemonRequest>>>,
    /// What the daemon does with a `SpeakGenerate` request.
    pub speech_outcome: StubSpeechOutcome,
    /// Captures the most recent `SpeakGenerate` request the CLI sent.
    pub speak_request_capture: Arc<std::sync::Mutex<Option<DaemonRequest>>>,
    /// What the daemon does with an `ImageGenerate` request.
    pub image_outcome: StubImageOutcome,
    /// Captures the most recent `ImageGenerate` request the CLI sent, so a
    /// test can assert exactly what crossed the socket.
    pub image_request_capture: Arc<std::sync::Mutex<Option<DaemonRequest>>>,
    /// Captures the images the CLI attached to the most recent `ChatGenerate`,
    /// when a test supplies one. Shared across connections by the Arc clone.
    pub image_capture: Arc<std::sync::Mutex<Option<Vec<ChatImageInput>>>>,
    /// Captures the full ordered message list the CLI sent on the most recent
    /// `ChatGenerate`, so a test can assert a system message was prepended.
    pub messages_capture: Arc<std::sync::Mutex<Option<Vec<ChatMessage>>>>,
    /// Captures the generation settings the CLI sent on the most recent
    /// `ChatGenerate`, so a test can assert a thinking budget was carried.
    pub settings_capture: Arc<std::sync::Mutex<Option<ChatGenerationSettings>>>,
    /// Captures the raw schema text the CLI sent on the most recent
    /// `ChatGenerate`, so a test can assert the schema crossed the boundary.
    pub schema_capture: Arc<std::sync::Mutex<Option<String>>>,
    /// FIFO of jobs for `DownloadStatus` requests; the last entry repeats
    /// once the scripted list is exhausted. `None` means no active job.
    pub download_jobs: Vec<Option<DaemonDownloadJob>>,
    /// The download list sent with each `DownloadStatus`, in step with `download_jobs`.
    pub download_waiting_huggingface_ids: Vec<Vec<String>>,
}

impl Default for StubDaemonConfig {
    fn default() -> Self {
        StubDaemonConfig {
            worker_status: DaemonWorkerStatus::Ready,
            installed_models: Vec::new(),
            catalog_entries: Vec::new(),
            catalog_entries_after_download: None,
            default_model_id: None,
            chat_fragments: Vec::new(),
            chat_first_frame_delay: Duration::ZERO,
            model_load_reports: Vec::new(),
            request_progress_counts: Vec::new(),
            hung_status_requests: 0..0,
            neural_engine_prefill: None,
            draft_model: None,
            model_unload_status: ipc_protocol::DaemonModelUnloadStatus::Unloaded,
            embeddings_outcome: StubEmbeddingsOutcome::Completed,
            embeddings_first_frame_delay: Duration::ZERO,
            embedding_vector: vec![0.25, -0.5],
            decisions_outcome: StubDecisionsOutcome::Completed(Vec::new()),
            decide_request_capture: Arc::new(std::sync::Mutex::new(None)),
            speech_outcome: StubSpeechOutcome::Completed(vec![0, 0, 0, 64]),
            speak_request_capture: Arc::new(std::sync::Mutex::new(None)),
            image_outcome: StubImageOutcome::Completed(Vec::new()),
            image_request_capture: Arc::new(std::sync::Mutex::new(None)),
            download_jobs: Vec::new(),
            download_waiting_huggingface_ids: Vec::new(),
            image_capture: Arc::new(std::sync::Mutex::new(None)),
            messages_capture: Arc::new(std::sync::Mutex::new(None)),
            settings_capture: Arc::new(std::sync::Mutex::new(None)),
            schema_capture: Arc::new(std::sync::Mutex::new(None)),
        }
    }
}

impl StubDaemonConfig {
    /// The model the daemon reports as resident: the first resident
    /// installed model, mirroring the real daemon's status frame.
    fn resident_model_id(&self) -> Option<String> {
        self.installed_models
            .iter()
            .find(|installed_model| installed_model.is_resident)
            .map(|installed_model| installed_model.model_id.clone())
    }
}

/// The next scripted load report across every connection, repeating the last one
/// the way a stuck load stays reported.
fn next_scripted_load_report(
    model_load_reports: &[Option<ModelLoadInFlight>],
    model_load_report_index: &Mutex<usize>,
) -> Option<ModelLoadInFlight> {
    let mut report_index = model_load_report_index
        .lock()
        .expect("the stub report index lock should stay live");
    let scripted_report = model_load_reports
        .get(*report_index)
        .or(model_load_reports.last())
        .cloned()
        .flatten();
    *report_index += 1;
    scripted_report
}

/// The next scripted progress count across every connection.
fn next_scripted_progress_count(
    request_progress_counts: &[u64],
    request_progress_count_index: &Mutex<usize>,
) -> u64 {
    let mut count_index = request_progress_count_index
        .lock()
        .expect("the stub count index lock should stay live");
    let scripted_count = request_progress_counts
        .get(*count_index)
        .or(request_progress_counts.last())
        .copied()
        .unwrap_or(0);
    *count_index += 1;
    scripted_count
}

/// Numbers this `Status` request across every connection and, when it is scripted to
/// hang, outlasts any test bound so the client's own deadline is what ends the wait.
async fn wait_out_hung_status_request(
    hung_status_requests: &std::ops::Range<usize>,
    status_request_count: &Mutex<usize>,
) {
    let status_request_index = {
        let mut request_count = status_request_count
            .lock()
            .expect("the stub status count lock should stay live");
        let status_request_index = *request_count;
        *request_count += 1;
        status_request_index
    };
    if hung_status_requests.contains(&status_request_index) {
        tokio::time::sleep(Duration::from_secs(3_600)).await;
    }
}

/// Binds the socket and serves scripted exchanges until aborted.
pub fn spawn_stub_daemon(
    socket_path: PathBuf,
    config: StubDaemonConfig,
) -> tokio::task::JoinHandle<()> {
    // Bind before spawning so the client can never race an unbound socket.
    let std_listener = std::os::unix::net::UnixListener::bind(&socket_path)
        .expect("the stub daemon should bind the test socket");
    std_listener
        .set_nonblocking(true)
        .expect("the stub listener should accept non-blocking mode");
    let unix_listener = tokio::net::UnixListener::from_std(std_listener)
        .expect("the stub listener should register with the runtime");
    let state = Arc::new(StubDaemonState {
        resident_model_id: config.resident_model_id(),
        catalog_entries_after_download: config.catalog_entries_after_download.clone(),
        default_model_id: Mutex::new(config.default_model_id.clone()),
        download_job_index: Mutex::new(0usize),
        download_started: Mutex::new(false),
        model_load_report_index: Mutex::new(0usize),
        request_progress_count_index: Mutex::new(0usize),
        status_request_count: Mutex::new(0usize),
        config,
    });
    tokio::spawn(async move {
        loop {
            let Ok((connection_stream, _peer_address)) = unix_listener.accept().await else {
                return;
            };
            tokio::spawn(serve_stub_connection(connection_stream, Arc::clone(&state)));
        }
    })
}

/// The frame the stub answers an image request with, or `None` when the
/// scripted worker never answers: the stalled case waits out an hour (far
/// beyond any test bound) so the client's own deadline is what ends it.
async fn scripted_image_response(
    image_outcome: &StubImageOutcome,
    image_request: ImageGenerateRequest,
) -> Option<DaemonResponse> {
    match image_outcome {
        StubImageOutcome::Completed(png_bytes) => Some(DaemonResponse::ImageGenerationCompleted {
            model: image_request.model,
            image: GeneratedImage {
                mime_type: "image/png".to_owned(),
                encoded_bytes: png_bytes.clone(),
            },
            metadata: ImageGenerationResultMetadata {
                width_pixels: image_request.width_pixels,
                height_pixels: image_request.height_pixels,
                steps: 4,
                guidance_thousandths: 1_000,
                seed: image_request.seed.unwrap_or(7),
                elapsed_millis: 1_234,
            },
        }),
        StubImageOutcome::Rejected(reason) => Some(DaemonResponse::GenerationRejected {
            reason: reason.clone(),
        }),
        StubImageOutcome::Failed(failure_reason) => Some(DaemonResponse::ImageGenerationFailed {
            reason: failure_reason.clone(),
        }),
        StubImageOutcome::Stalls => {
            tokio::time::sleep(std::time::Duration::from_secs(3_600)).await;
            None
        }
    }
}

/// The frames a helper-model request is answered with: the audio frames and
/// the terminal one for speech, else the one terminal frame.
async fn scripted_helper_model_frames(
    config: &StubDaemonConfig,
    resident_model_id: Option<String>,
    helper_model_request: DaemonRequest,
) -> Vec<DaemonResponse> {
    let DaemonRequest::SpeakGenerate(speak_request) = &helper_model_request else {
        return vec![
            scripted_helper_model_response(config, resident_model_id, helper_model_request).await,
        ];
    };
    let answering_model_id = speak_request.model.clone().unwrap_or_default();
    let seed = speak_request.seed.unwrap_or(7);
    let voice_name = speak_request
        .voice_name
        .clone()
        .unwrap_or_else(|| "af_heart".to_owned());
    *config
        .speak_request_capture
        .lock()
        .expect("the stub speak capture should stay live") = Some(helper_model_request);
    scripted_speech_frames(&config.speech_outcome, answering_model_id, voice_name, seed)
}

fn scripted_speech_frames(
    speech_outcome: &StubSpeechOutcome,
    answering_model_id: String,
    voice_name: String,
    seed: u64,
) -> Vec<DaemonResponse> {
    let pcm16_bytes = match speech_outcome {
        StubSpeechOutcome::Failed(reason) => {
            return vec![DaemonResponse::SpeechFailed {
                reason: reason.clone(),
            }];
        }
        StubSpeechOutcome::Completed(pcm16_bytes) => pcm16_bytes,
    };
    let (first_half, second_half) = pcm16_bytes.split_at(pcm16_bytes.len() / 4 * 2);
    vec![
        DaemonResponse::SpeechAudio {
            pcm16_bytes: first_half.to_vec(),
        },
        DaemonResponse::SpeechAudio {
            pcm16_bytes: second_half.to_vec(),
        },
        DaemonResponse::SpeechCompleted {
            model: answering_model_id,
            metadata: ipc_protocol::SpeechResultMetadata {
                sample_rate_hertz: 24_000,
                sample_count: (pcm16_bytes.len() / 2) as u64,
                segment_count: 1,
                cut_short_segment_count: 0,
                voice_name,
                seed,
            },
        },
    ]
}

async fn send_frames_and_close(
    mut protocol_writer: ProtocolWriter<impl tokio::io::AsyncWrite + Unpin>,
    daemon_responses: Vec<DaemonResponse>,
) {
    for daemon_response in &daemon_responses {
        protocol_writer
            .send_daemon_response(daemon_response)
            .await
            .expect("the stub helper-model frame should transmit");
    }
    let _ = protocol_writer.close().await;
}

/// The one terminal frame for a request a helper model answers in one step:
/// an embed request after its scripted delay, or a decide request, which is
/// also captured so a test can read what crossed the socket.
async fn scripted_helper_model_response(
    config: &StubDaemonConfig,
    resident_model_id: Option<String>,
    helper_model_request: DaemonRequest,
) -> DaemonResponse {
    let requested_model_id = match &helper_model_request {
        DaemonRequest::EmbedGenerate { model, .. }
        | DaemonRequest::DecideGenerate { model, .. } => model.clone(),
        _ => None,
    };
    let answering_model_id = requested_model_id
        .or(resident_model_id)
        .expect("the completed stub always has a model");
    if matches!(helper_model_request, DaemonRequest::DecideGenerate { .. }) {
        *config
            .decide_request_capture
            .lock()
            .expect("the stub decide capture should stay live") = Some(helper_model_request);
        return scripted_decisions_response(&config.decisions_outcome, answering_model_id);
    }
    tokio::time::sleep(config.embeddings_first_frame_delay).await;
    match config.embeddings_outcome {
        StubEmbeddingsOutcome::Completed => DaemonResponse::EmbeddingsCompleted {
            model: answering_model_id,
            vectors: vec![config.embedding_vector.clone()],
            input_token_counts: vec![3],
        },
        StubEmbeddingsOutcome::ContextLengthExceeded => DaemonResponse::EmbeddingsFailed {
            reason: EmbeddingsFailureReason::ContextLengthExceeded {
                actual_total_context_tokens: 5_000,
                maximum_context_tokens: 2_048,
            },
        },
    }
}

/// One scripted download job for the stub's `DownloadStatus` sequence.
pub fn stub_download_job(
    huggingface_id: &str,
    state: &str,
    bytes_completed: u64,
    bytes_total: u64,
    error: Option<&str>,
) -> DaemonDownloadJob {
    DaemonDownloadJob {
        huggingface_id: huggingface_id.to_owned(),
        state: state.to_owned(),
        bytes_completed,
        bytes_total,
        error: error.map(str::to_owned),
    }
}
