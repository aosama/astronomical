//! Configurable stub daemon speaking the real framed protocol over a real
//! unix socket, shared by the one-shot verb journey tests.
//!
//! It mirrors the daemon's lifecycle surface: status, installed models,
//! catalog, download start/status, default-model persistence, and scripted
//! chat or embeddings results. One connection serves a scripted exchange
//! sequence, like the real daemon.

use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};

use astronomical_decision_contract::{DecisionFailure, QuestionOutcome};
use astronomical_ipc_protocol::{
    ChatGenerationSettings, ChatImageInput, ChatMessage, DAEMON_APPLICATION_NAME,
    DAEMON_PROTOCOL_VERSION, DaemonCatalogEntry, DaemonDownloadJob, DaemonListedModel,
    DaemonRequest, DaemonResponse, DaemonWorkerStatus, EmbeddingsFailureReason, GeneratedImage,
    ImageGenerateRequest, ImageGenerationFailureReason, ImageGenerationResultMetadata,
    ModelLoadInFlight, ProtocolReader, ProtocolWriter,
};

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
    Failed(astronomical_speech_contract::SpeechFailure),
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
    pub neural_engine_prefill: Option<astronomical_ipc_protocol::NeuralEnginePrefillStatus>,
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
#[expect(clippy::too_many_lines)]
pub fn spawn_stub_daemon(
    socket_path: PathBuf,
    config: StubDaemonConfig,
) -> tokio::task::JoinHandle<()> {
    let resident_model_id = config.resident_model_id();
    let catalog_entries_after_download = Arc::new(config.catalog_entries_after_download.clone());
    let default_model_id = Arc::new(Mutex::new(config.default_model_id.clone()));
    let download_job_index = Arc::new(Mutex::new(0usize));
    let download_started = Arc::new(Mutex::new(false));
    let model_load_report_index = Arc::new(Mutex::new(0usize));
    let request_progress_count_index = Arc::new(Mutex::new(0usize));
    let status_request_count = Arc::new(Mutex::new(0usize));
    // Bind before spawning so the client can never race an unbound socket.
    let std_listener = std::os::unix::net::UnixListener::bind(&socket_path)
        .expect("the stub daemon should bind the test socket");
    std_listener
        .set_nonblocking(true)
        .expect("the stub listener should accept non-blocking mode");
    let unix_listener = tokio::net::UnixListener::from_std(std_listener)
        .expect("the stub listener should register with the runtime");
    let config = Arc::new(config);
    tokio::spawn(async move {
        loop {
            let Ok((connection_stream, _peer_address)) = unix_listener.accept().await else {
                return;
            };
            let config = Arc::clone(&config);
            let resident_model_id = resident_model_id.clone();
            let catalog_entries_after_download = Arc::clone(&catalog_entries_after_download);
            let default_model_id = Arc::clone(&default_model_id);
            let download_job_index = Arc::clone(&download_job_index);
            let download_started = Arc::clone(&download_started);
            let model_load_report_index = Arc::clone(&model_load_report_index);
            let request_progress_count_index = Arc::clone(&request_progress_count_index);
            let status_request_count = Arc::clone(&status_request_count);
            tokio::spawn(async move {
                let (read_half, write_half) = connection_stream.into_split();
                let mut protocol_reader = ProtocolReader::new(read_half);
                let mut protocol_writer = ProtocolWriter::new(write_half);
                loop {
                    let Some(daemon_request) = protocol_reader.next_daemon_request().await.expect(
                        "the stub daemon request read should not fail at the transport layer",
                    ) else {
                        return;
                    };
                    match daemon_request {
                        DaemonRequest::Handshake => {
                            protocol_writer
                                .send_daemon_response(&DaemonResponse::HandshakeAccepted {
                                    protocol_version: DAEMON_PROTOCOL_VERSION,
                                    application_name: DAEMON_APPLICATION_NAME.to_owned(),
                                })
                                .await
                                .expect("the stub handshake should transmit");
                        }
                        DaemonRequest::Status => {
                            wait_out_hung_status_request(
                                &config.hung_status_requests,
                                &status_request_count,
                            )
                            .await;
                            let current_default_model_id = default_model_id
                                .lock()
                                .expect("the stub default lock should stay live")
                                .clone();
                            let model_load_in_flight = next_scripted_load_report(
                                &config.model_load_reports,
                                &model_load_report_index,
                            );
                            protocol_writer
                                .send_daemon_response(&DaemonResponse::Status {
                                    worker_status: config.worker_status,
                                    ready_model_id: resident_model_id.clone(),
                                    default_model_id: current_default_model_id,
                                    model_load_in_flight,
                                    request_progress_count: next_scripted_progress_count(
                                        &config.request_progress_counts,
                                        &request_progress_count_index,
                                    ),
                                    neural_engine_prefill: config.neural_engine_prefill.clone(),
                                })
                                .await
                                .expect("the stub status should transmit");
                        }
                        DaemonRequest::ModelsList => {
                            let listed_models: Vec<DaemonListedModel> = config
                                .installed_models
                                .iter()
                                .map(|installed_model| DaemonListedModel {
                                    model_id: installed_model.model_id.clone(),
                                    family: "stub".to_owned(),
                                    context_window: installed_model.context_window,
                                    supports_embeddings: installed_model.supports_embeddings,
                                    supports_decisions: installed_model.supports_decisions,
                                    supports_speech: installed_model.supports_speech,
                                    supports_image_generation: installed_model
                                        .supports_image_generation,
                                    is_resident: installed_model.is_resident,
                                    size_bytes: 1_000_000_000,
                                })
                                .collect();
                            protocol_writer
                                .send_daemon_response(&DaemonResponse::ModelsList {
                                    models: listed_models,
                                })
                                .await
                                .expect("the stub model list should transmit");
                        }
                        DaemonRequest::Catalog => {
                            let after_download_active = *download_started
                                .lock()
                                .expect("the stub download flag should stay live");
                            let catalog_entries = if after_download_active {
                                catalog_entries_after_download.as_ref().clone().expect(
                                    "a catalog flip must be scripted when a download starts",
                                )
                            } else {
                                config.catalog_entries.clone()
                            };
                            let entries: Vec<DaemonCatalogEntry> = catalog_entries
                                .into_iter()
                                .map(|catalog_entry| {
                                    let model_id = astronomical_config::leaf_model_id(
                                        &catalog_entry.huggingface_id,
                                    )
                                    .to_owned();
                                    DaemonCatalogEntry {
                                        huggingface_id: catalog_entry.huggingface_id,
                                        model_id,
                                        family: "stub".to_owned(),
                                        // ~2 GB keeps rendered sizes at two digits so
                                        // list-output assertions stay readable.
                                        approximate_size_bytes: 2_000_000_000,
                                        ready_on_this_mac: catalog_entry.ready_on_this_mac,
                                        requestable_model_id: catalog_entry.requestable_model_id,
                                        download_state: catalog_entry.download_state,
                                        context_window: catalog_entry.context_window,
                                        supports_reasoning: false,
                                        supports_vision: false,
                                        supports_tool_calls: false,
                                        supports_image_generation: catalog_entry
                                            .supports_image_generation,
                                        supports_embeddings: catalog_entry.supports_embeddings,
                                        supports_decisions: catalog_entry.supports_decisions,
                                        supports_speech: catalog_entry.supports_speech,
                                    }
                                })
                                .collect();
                            protocol_writer
                                .send_daemon_response(&DaemonResponse::Catalog { entries })
                                .await
                                .expect("the stub catalog should transmit");
                        }
                        DaemonRequest::DownloadStart { model_id } => {
                            *download_started
                                .lock()
                                .expect("the stub download flag should stay live") = true;
                            protocol_writer
                                .send_daemon_response(&DaemonResponse::DownloadStarted {
                                    huggingface_id: model_id,
                                })
                                .await
                                .expect("the stub download start should transmit");
                        }
                        DaemonRequest::DownloadStatus => {
                            // The guard must drop before the send await so the task stays Send.
                            let scripted_job = {
                                let mut job_index = download_job_index
                                    .lock()
                                    .expect("the stub job index should stay live");
                                let scripted_job = if *job_index < config.download_jobs.len() {
                                    config.download_jobs[*job_index].clone()
                                } else {
                                    config.download_jobs.last().cloned().unwrap_or(None)
                                };
                                *job_index += 1;
                                scripted_job
                            };
                            protocol_writer
                                .send_daemon_response(&DaemonResponse::DownloadStatus {
                                    job: scripted_job,
                                })
                                .await
                                .expect("the stub download status should transmit");
                        }
                        DaemonRequest::DefaultModelSet { model_id } => {
                            *default_model_id
                                .lock()
                                .expect("the stub default lock should stay live") =
                                Some(model_id.clone());
                            protocol_writer
                                .send_daemon_response(&DaemonResponse::DefaultModelSet {
                                    default_model_id: model_id,
                                })
                                .await
                                .expect("the stub default set should transmit");
                        }
                        DaemonRequest::ChatGenerate {
                            messages,
                            settings,
                            schema_json,
                            ..
                        } => {
                            // Record the messages and settings the CLI sent so a journey
                            // test can assert they crossed the boundary.
                            if let Ok(mut guard) = config.messages_capture.lock() {
                                *guard = Some(messages.clone());
                            }
                            if let Ok(mut guard) = config.settings_capture.lock() {
                                *guard = Some(settings.clone());
                            }
                            if let Ok(mut guard) = config.schema_capture.lock() {
                                *guard = schema_json.clone();
                            }
                            // The user's images may sit after a system message, so find
                            // the first user message rather than assuming it is first.
                            if let Some(user_images) =
                                messages.iter().find_map(|message| match message {
                                    ChatMessage::User { images, .. } => Some(images.clone()),
                                    _ => None,
                                })
                                && let Ok(mut guard) = config.image_capture.lock()
                            {
                                *guard = Some(user_images);
                            }
                            tokio::time::sleep(config.chat_first_frame_delay).await;
                            for chat_fragment in config.chat_fragments.iter() {
                                protocol_writer
                                    .send_daemon_response(&DaemonResponse::ChatGenerationText {
                                        text: chat_fragment.clone(),
                                    })
                                    .await
                                    .expect("the stub fragment should transmit");
                            }
                            protocol_writer
                                .send_daemon_response(
                                    &DaemonResponse::ChatGenerationCompleted {
                                        prompt_token_count: 7,
                                        generated_token_count: 2,
                                        reasoning_token_count: 0,
                                        cached_token_count: 0,
                                        reason:
                                            astronomical_ipc_protocol::ChatGenerationCompletionReason::EndOfSequence,
                                    },
                                )
                                .await
                                .expect("the stub completion frame should transmit");
                            let _ = protocol_writer.close().await;
                            return;
                        }
                        helper_model_request @ (DaemonRequest::EmbedGenerate { .. }
                        | DaemonRequest::DecideGenerate { .. }
                        | DaemonRequest::SpeakGenerate(_)) => {
                            let helper_model_frames = scripted_helper_model_frames(
                                &config,
                                resident_model_id.clone(),
                                helper_model_request,
                            )
                            .await;
                            send_frames_and_close(protocol_writer, helper_model_frames).await;
                            return;
                        }
                        DaemonRequest::ImageGenerate(image_request) => {
                            *config
                                .image_request_capture
                                .lock()
                                .expect("the stub image capture should stay live") =
                                Some(DaemonRequest::ImageGenerate(image_request.clone()));
                            let Some(image_response) =
                                scripted_image_response(&config.image_outcome, image_request).await
                            else {
                                return;
                            };
                            protocol_writer
                                .send_daemon_response(&image_response)
                                .await
                                .expect("the stub image frame should transmit");
                            let _ = protocol_writer.close().await;
                            return;
                        }
                    }
                }
            });
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
            metadata: astronomical_ipc_protocol::SpeechResultMetadata {
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

/// The one terminal frame the scripted outcome stands for.
fn scripted_decisions_response(
    decisions_outcome: &StubDecisionsOutcome,
    answering_model_id: String,
) -> DaemonResponse {
    match decisions_outcome {
        StubDecisionsOutcome::Completed(question_outcomes) => DaemonResponse::DecisionsCompleted {
            model: answering_model_id,
            question_outcomes: question_outcomes.clone(),
            input_token_count: 42,
            elapsed_millis: 7,
        },
        StubDecisionsOutcome::Failed(failure_reason) => DaemonResponse::DecisionsFailed {
            reason: failure_reason.clone(),
        },
        StubDecisionsOutcome::Rejected(reason) => DaemonResponse::GenerationRejected {
            reason: reason.clone(),
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
